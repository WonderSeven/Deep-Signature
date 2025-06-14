# Spectral Clustering with Graph Neural Networks for Graph Pooling
import torch
import torch.nn as nn
from torch.nn import functional as F
from .modules import DenseGraphConv, dense_mincut_pool
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch


class LocalAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, num_clusters, hidden_channels=32):
        super(LocalAggregator, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = nn.Linear(hidden_channels, 8 * num_clusters)
        # multi-level aggregation
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        self.pool2 = nn.Linear(hidden_channels, 4 * num_clusters)
        # multi-level aggregation
        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
        self.pool3 = nn.Linear(hidden_channels, num_clusters)

        self.conv4 = DenseGraphConv(hidden_channels, out_channels)
        self.activation = nn.ELU()

    def forward(self, data):
        """
        :param x: [48, 50, 100]
        :param adj: [48, 25, 25]
        :return:
        """
        x = data.pos.float().transpose(0, 1)
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        x = self.conv1(x, edge_index, edge_attr).relu()
        x = x.transpose(0, 1)
        x, mask = to_dense_batch(x, data.batch)
        adj = to_dense_adj(edge_index, data.batch)
        x = x.transpose(1, 2)
        s1 = self.pool1(x).mean(1).tanh()
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s1, mask)

        x = self.conv2(x, adj)
        s2 = self.pool2(x).mean(1).tanh()
        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s2)

        x = self.conv3(x, adj)
        s3 = self.pool3(x).mean(1).tanh()
        x, adj, mc3, o3 = dense_mincut_pool(x, adj, s3)

        x = self.conv4(x, adj)

        # Ensure the consistency for pooling in dense_mincut_pool
        s1, s2, s3 = torch.softmax(s1, dim=-1), torch.softmax(s2, dim=-1), torch.softmax(s3, dim=-1)

        agg_matrix = torch.bmm(s1, s2)
        agg_matrix = torch.bmm(agg_matrix, s3)

        return x, mc1 + mc2 + mc3, o1 + o2 + o3, agg_matrix


class FrameAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, num_clusters, hidden_channels=32):
        super(FrameAggregator, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = nn.Linear(hidden_channels, 2 * num_clusters)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        # multi-level aggregation
        self.pool2 = nn.Linear(hidden_channels, num_clusters)

        self.conv3 = DenseGraphConv(hidden_channels, out_channels)

    def forward(self, data):
        """
        :param x0: [48, 50, 100]
        :param adj: [48, 25, 25]
        :return:
        """
        x = data.pos.float().transpose(0, 1)
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        x = self.conv1(x, edge_index, edge_attr).relu()
        x = x.transpose(0, 1)
        x, mask = to_dense_batch(x, data.batch)
        adj = to_dense_adj(edge_index, data.batch)
        x = x.transpose(1, 2)
        s1 = self.pool1(x).mean(1).tanh()
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s1, mask)

        x = self.conv2(x, adj).relu()
        s2 = self.pool2(x).mean(1).tanh()
        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s2)

        x = self.conv3(x, adj)

        s1, s2 = torch.softmax(s1, dim=-1), torch.softmax(s2, dim=-1)
        agg_matrix = torch.bmm(s1, s2)

        return x, mc1 + mc2, o1 + o2, agg_matrix

