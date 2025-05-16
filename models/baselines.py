import os
import abc
import pdb

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from models.local_aggregation import *
from models.temporal_aggregation import *
from models.readout import BinaryClassifier


class AbstractMethod(nn.Module, abc.ABC):
    def __init__(self, spatial_in_dim, spatial_out_dim, temporal_out_dim, num_clusters):
        super().__init__()
        self.spatial_in_dim = spatial_in_dim
        self.spatial_out_dim = spatial_out_dim
        self.temporal_out_dim = temporal_out_dim
        self.num_clusters = num_clusters

    @abc.abstractmethod
    def _build(self):
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class FrameLSTM(AbstractMethod):
    def __init__(self, spatial_in_dim, spatial_out_dim, spatial_hidden_dim, temporal_out_dim, num_clusters, local_mode):
        super(FrameLSTM, self).__init__(spatial_in_dim, spatial_out_dim, temporal_out_dim, num_clusters)
        self.spatial_hidden_dim = spatial_hidden_dim
        self.local_mode = local_mode
        self._build()

    def _build(self):
        self.local_aggregator = LocalAggregator(self.spatial_in_dim, self.spatial_out_dim,
                                                hidden_channels=self.spatial_hidden_dim,
                                                num_clusters=self.num_clusters)
        self.energy_regressor = nn.Linear(self.spatial_hidden_dim, self.spatial_in_dim)
        self.temporal_aggregator = LSTMAggregator(self.spatial_in_dim, self.temporal_out_dim,
                                                  hidden_channels=self.temporal_out_dim)
        self.classifier = BinaryClassifier(self.temporal_out_dim, reduction='mean')

    def forward(self, data: Data):
        """
        DataBatch(edge_index=[2, 29796], edge_attr=[29796, 1], y=[16, 1], pos=[19864, 100, 3], batch=[19864], ptr=[17])
        :param data:
        :return:
        """
        if self.local_mode == 'head_and_tail':
            data.pos = torch.stack([data.pos[:, 0, :], data.pos[:, -1, :]], dim=1)
        elif self.local_mode == 'head':
            data.pos = data.pos[:, 0, :].unsqueeze(1)
        elif self.local_mode == 'tail':
            data.pos = data.pos[:, -1, :].unsqueeze(1)
        else:
            raise NotImplementedError

        hidden_x, mc_loss, o_loss, agg_matrix = self.local_aggregator(data) # [16, 1, 50, 10], [16, 1384, 50]
        pred_e = self.energy_regressor(hidden_x).squeeze(-1) # [16, 1, 50, 3]

        traj, batch = data.pos, data.batch
        traj, mask = to_dense_batch(traj, data.batch) # [16, 1384, 100, 3]
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

