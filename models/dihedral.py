import os
import pdb

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Data
from models.readout import BinaryClassifier
from .baselines import AbstractMethod


class DihedralPCA(AbstractMethod):
    def __init__(self, spatial_in_dim, spatial_out_dim, spatial_hidden_dim, temporal_out_dim, num_clusters):
        super(DihedralPCA, self).__init__(spatial_in_dim, spatial_out_dim, temporal_out_dim, num_clusters)
        self.spatial_hidden_dim = spatial_hidden_dim
        self._build()

    def _build(self):
        # self.local_aggregator = FrameAggregator(self.spatial_in_dim, self.spatial_out_dim,
        #                                         hidden_channels=self.spatial_hidden_dim,
        #                                         num_clusters=self.num_clusters,
        #                                         mode=self.local_mode)
        # self.energy_regressor = nn.Linear(self.spatial_hidden_dim, self.spatial_in_dim)
        # self.temporal_aggregator = LSTMAggregator(self.spatial_hidden_dim, self.temporal_out_dim, hidden_channels=self.temporal_out_dim)
        self.classifier = BinaryClassifier(self.temporal_out_dim, reduction='mean')

    def forward(self, data: Data):
        dih, A = data.dih, data.A

        pdb.set_trace()
        torch.pca_lowrank(dih.cpu())

        hidden_x, mc_loss, o_loss, agg_matrix = self.local_aggregator(traj, A)  # [48, 25, 20]
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

        # Temporal aggregation
        hidden_x = self.temporal_aggregator(hidden_x)

        pred_y = self.classifier(hidden_x).squeeze(-1)

        return pred_y, local_pred_loss, mc_loss, o_loss
