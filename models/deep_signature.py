import os
import pdb
import math
import signatory
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch

from .baselines import AbstractMethod
from .local_aggregation import LocalAggregator
from models.temporal_aggregation import SubSignatureAggregator
from models.readout import BinaryClassifier


class DeepSignature(AbstractMethod):
    def __init__(self, spatial_in_dim, spatial_out_dim, spatial_hidden_dim, temporal_out_dim, num_clusters, signature_depth):
        super(DeepSignature, self).__init__(spatial_in_dim, spatial_out_dim, temporal_out_dim, num_clusters)
        self.spatial_hidden_dim = spatial_hidden_dim
        self.signature_depth = signature_depth
        self._build()

    def _build(self):
        self.local_aggregator = LocalAggregator(self.spatial_in_dim, self.spatial_out_dim,
                                                hidden_channels=self.spatial_hidden_dim,
                                                num_clusters=self.num_clusters)
        self.temporal_aggregator = SubSignatureAggregator(self.spatial_in_dim, self.temporal_out_dim,
                                                          self.num_clusters, signature_depth=self.signature_depth)
        self.energy_regressor = nn.Linear(self.spatial_hidden_dim, self.spatial_in_dim)
        self.classifier = BinaryClassifier(self.temporal_out_dim, reduction='none')

    def forward(self, data: Data):
        """
        :param data: DataBatch(edge_index=[2, 29820], edge_attr=[29820, 1], y=[16, 1], pos=[19880, 100, 3], batch=[19880], ptr=[17])
        :return:
        """
        hidden_x, mc_loss, o_loss, agg_matrix = self.local_aggregator(data)  # hidden_x:[16, 100, 50, 20] agg_matrix:[16, 1384, 50]
        pred_e = self.energy_regressor(hidden_x).squeeze(-1)

        traj, batch = data.pos, data.batch
        traj, mask = to_dense_batch(traj, data.batch)  # [16, 1384, 100, 3]
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

