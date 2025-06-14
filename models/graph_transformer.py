import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from .deep_signature import LocalAggregator
from .temporal_aggregation import *
from .readout import BinaryClassifier
from .baselines import AbstractMethod


class GraphTransformer(AbstractMethod):
    def __init__(self, spatial_in_dim, spatial_out_dim, spatial_hidden_dim, temporal_out_dim, num_clusters):
        super(GraphTransformer, self).__init__(spatial_in_dim, spatial_out_dim, temporal_out_dim, num_clusters)
        self.spatial_hidden_dim = spatial_hidden_dim
        self._build()

    def _build(self):
        self.local_aggregator = LocalAggregator(self.spatial_in_dim, self.spatial_out_dim,
                                                hidden_channels=self.spatial_hidden_dim,
                                                num_clusters=self.num_clusters)
        self.temporal_aggregator = TransformerAggregator(self.num_clusters * self.spatial_in_dim, self.temporal_out_dim,
                                                         hidden_channels=self.temporal_out_dim)
        self.energy_regressor = nn.Linear(self.spatial_hidden_dim, self.spatial_in_dim)
        self.classifier = BinaryClassifier(self.temporal_out_dim, reduction='mean')

    def forward(self, data: Data):
        """
        :param data:
        :return:
        """
        hidden_x, mc_loss, o_loss, agg_matrix = self.local_aggregator(data)
        pred_e = self.energy_regressor(hidden_x).squeeze(-1)

        traj, batch = data.pos, data.batch
        traj, mask = to_dense_batch(traj, data.batch)
        traj = traj.transpose(1, 2)

        # Clustering
        if traj.dim() == 3:
            agg_traj = torch.einsum('btn, bnm->btm', traj, agg_matrix.detach())
        elif traj.dim() == 4:
            agg_traj = torch.einsum('btnd, bnm->btmd', traj, agg_matrix.detach())
        else:
            raise NotImplementedError

        # Energy prediction loss
        local_pred_loss = F.l1_loss(pred_e, agg_traj, reduction='mean')

        # Temporal aggregation
        hidden_x = self.temporal_aggregator(pred_e)

        pred_y = self.classifier(hidden_x)

        return pred_y, local_pred_loss, mc_loss, o_loss



class TransformerAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32, trans_num_layers=2):
        super(TransformerAggregator, self).__init__()
        self.hidden_channels = hidden_channels

        self.bi_lstm_layer = nn.LSTM(in_channels, hidden_channels, 1, bidirectional=True, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels,
                                                   nhead=5,
                                                   dim_feedforward=1024,
                                                   dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_num_layers)
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        """
        :param x: [48, 50, 25, 20]
        :return:
        """
        batch_size, time_steps, num_nodes = x.shape[:3]

        x = x.permute(1, 0, 2, 3)
        x = x.contiguous().view(time_steps, batch_size, -1)
        out = self.transformer(x)
        out = out.mean(dim=0)
        # out = self.norm(out)
        out = self.linear(out).unsqueeze(1)
        return out