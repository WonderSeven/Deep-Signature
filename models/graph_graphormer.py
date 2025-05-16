import pdb
from re import split

import torch
import torch.nn as nn
from mmtf.utils.codec_utils import add_header
from torch_geometric.data import Data, Batch
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn import DenseGraphConv, GCNConv, dense_mincut_pool
from models.local_aggregation import *
from models.temporal_aggregation import *
from models.readout import BinaryClassifier
from models.graphormer_modules import GraphormerModule
from .baselines import AbstractMethod


class GraphGraphormer(AbstractMethod):
    def __init__(self, spatial_in_dim, spatial_out_dim, spatial_hidden_dim, temporal_out_dim, num_clusters, local_mode):
        super(GraphGraphormer, self).__init__(spatial_in_dim, spatial_out_dim, temporal_out_dim, num_clusters)
        self.spatial_hidden_dim = spatial_hidden_dim
        self.local_mode = local_mode
        self._build()

    def _build(self):
        self.local_aggregator = FrameAggregator(self.spatial_in_dim, self.spatial_out_dim,
                                                hidden_channels=self.spatial_hidden_dim,
                                                num_clusters=self.num_clusters,
                                                mode=self.local_mode)
        self.graphormer_module = GraphormerModule(num_layers=3,
                                                 input_node_dim=self.spatial_hidden_dim,
                                                 node_dim=128,
                                                 input_edge_dim=1,
                                                 output_dim=self.spatial_out_dim,
                                                 edge_dim=128,
                                                 n_heads=4,
                                                 ff_dim=256,
                                                 max_in_degree=5,
                                                 max_out_degree=5,
                                                 max_path_distance=5,
                                                 )
        self.energy_regressor = nn.Linear(self.spatial_hidden_dim, self.spatial_in_dim)
        self.classifier = BinaryClassifier(self.spatial_out_dim, reduction='mean')

    def forward(self, data: Data):
        traj = data.pos

        if self.local_mode == 'head_and_tail':
            traj = torch.stack([traj[:, 0, ...], traj[:, -1, ...]], dim=1)
        elif self.local_mode == 'head':
            traj = traj[:, 0, ...].unsqueeze(1)
        elif self.local_mode == 'tail':
            traj = traj[:, -1, ...].unsqueeze(1)
        else:
            raise NotImplementedError

        batch_size, atom_nums = traj.shape[0], traj.shape[2]

        edge_index = data.edge_index.reshape(data.edge_index.shape[0], batch_size, -1)
        for idx in range(batch_size):
            edge_index[:, idx, :] += (idx * atom_nums - idx)

        edge_index = edge_index.reshape(edge_index.size(0), -1)  # [2, 33216]
        batch = data.batch.unsqueeze(-1).repeat(1, atom_nums).reshape(-1)
        data.pos, data.edge_index, data.batch, data.ptr = traj.reshape(-1, traj.shape[-1]), edge_index, batch, data.ptr * atom_nums

        hidden_x, mc_loss, o_loss, agg_matrix, adj = self.local_aggregator(data)  # [48, 25, 20]

        pred_e = self.energy_regressor(hidden_x).squeeze(-1)

        # Clustering
        if traj.dim() == 3:
            # traj: [48, 50, 100] -> [48, 50, 25]
            agg_traj = torch.einsum('btn, btnm->btm', traj, agg_matrix.detach())
        elif traj.dim() == 4:
            agg_traj = torch.einsum('btnd, btnm->btmd', traj, agg_matrix.detach())
        else:
            raise NotImplementedError

        # Energy prediction loss
        local_pred_loss = F.l1_loss(pred_e, agg_traj)
        split_adj = torch.split(adj, 1)
        edge_index = []
        for idx, cur_adj in enumerate(split_adj):
            cur_edge_index = torch.tensor(cur_adj.squeeze(0).nonzero(), dtype=torch.long)
            cur_edge_index += idx * self.num_clusters
            edge_index.append(cur_edge_index)
        edge_index = torch.cat(edge_index)

        edge_attr = torch.ones([edge_index.shape[0], 1]).float()

        batch = torch.arange(batch_size).unsqueeze(-1).repeat(1, self.num_clusters).reshape(-1)
        ptr = torch.arange(batch_size + 1) * self.num_clusters
        cg_data = Batch(pos=hidden_x.view(-1, hidden_x.shape[-1]), edge_index=edge_index, edge_attr=edge_attr,
                        y=data.y, batch=batch, ptr=ptr)

        # pdb.set_trace()
        out = self.graphormer_module(cg_data)
        pred_y = self.classifier(out).unsqueeze(-1)
        local_pred_loss = torch.tensor(0.)
        mc_loss = torch.tensor(0.)
        o_loss = torch.tensor(0.)
        return pred_y, local_pred_loss, mc_loss, o_loss


class FrameAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, num_clusters, mode='head_and_tail', hidden_channels=32):
        super(FrameAggregator, self).__init__()
        self.mode = mode
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # num_nodes = ceil(0.5 * average_nodes)
        self.pool1 = nn.Linear(hidden_channels, 2 * num_clusters)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        # TODO: multi-level aggregation
        # num_nodes = ceil(0.5 * average_nodes)
        self.pool2 = nn.Linear(hidden_channels, num_clusters)

        self.conv3 = DenseGraphConv(hidden_channels, out_channels)

    def forward(self, data):
        """
        :param x0: [48, 50, 100]
        :param adj: [48, 25, 25]
        :return:
        """
        batch_size = data.y.shape[0]
        time_steps = 1
        x = data.pos.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        x = self.conv1(x, edge_index, edge_attr).relu()
        x, mask = to_dense_batch(x, data.batch)
        adj = to_dense_adj(edge_index, data.batch)

        s1 = self.pool1(x) # .softmax(dim=-1) # .tanh()
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s1)

        x = self.conv2(x, adj).relu()
        s2 = self.pool2(x) # .softmax(dim=-1) # .tanh()
        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s2)

        x = self.conv3(x, adj)
        x = x.view(batch_size, time_steps, *x.shape[1:])

        s1, s2 = torch.softmax(s1, dim=-1), torch.softmax(s2, dim=-1)
        agg_matrix = torch.bmm(s1, s2)
        agg_matrix = agg_matrix.view(batch_size, time_steps, *agg_matrix.shape[1:])

        return x, mc1 + mc2, o1 + o2, agg_matrix, adj
