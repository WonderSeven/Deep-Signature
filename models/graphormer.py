import pdb

import torch
import torch.nn as nn
from click.core import batch
from torch_geometric.data import Data
from torch_geometric.nn.pool import global_mean_pool
from models.local_aggregation import *
from models.temporal_aggregation import *
from models.readout import BinaryClassifier
from models.graphormer_modules import GraphormerModule
from .baselines import AbstractMethod


class Graphormer(AbstractMethod):
    def __init__(self, spatial_in_dim, spatial_out_dim, spatial_hidden_dim, temporal_out_dim, num_clusters, local_mode):
        super(Graphormer, self).__init__(spatial_in_dim, spatial_out_dim, temporal_out_dim, num_clusters)
        self.spatial_hidden_dim = spatial_hidden_dim
        self.local_mode = local_mode
        self._build()

    def _build(self):
        self.graphormer_module = GraphormerModule(num_layers=3,
                                                  input_node_dim=self.spatial_in_dim,
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

        if batch_size == 1:
            data.pos, data.edge_attr = traj[0, 0], data.edge_attr[0]
        else:
            edge_index = data.edge_index.reshape(data.edge_index.shape[0], batch_size, -1)
            for idx in range(batch_size):
                edge_index[:, idx, :] += (idx * atom_nums - idx)

            edge_index = edge_index.reshape(edge_index.size(0), -1) # [2, 33216]
            traj = traj.reshape(-1, traj.shape[-1]) # [22144, 3]
            batch = data.batch.unsqueeze(-1).repeat(1, atom_nums).reshape(-1)

            data.pos, data.edge_index, data.batch, data.ptr = traj, edge_index, batch, data.ptr * atom_nums

        out = self.graphormer_module(data)
        out = global_mean_pool(out, data.batch)

        pred_y = self.classifier(out.unsqueeze(1)) #.unsqueeze(0)

        local_pred_loss = torch.tensor(0.)
        mc_loss = torch.tensor(0.)
        o_loss = torch.tensor(0.)

        return pred_y, local_pred_loss, mc_loss, o_loss
