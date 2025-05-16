# https://github.com/leffff/graphormer-pyg/blob/main/graphormer
from __future__ import annotations

import pdb
from typing import Tuple
from typing import Tuple, Dict, List, Union

import networkx as nx
import torch
from sympy.codegen.ast import continue_
from torch import nn
from torch.multiprocessing import spawn
from torch_geometric.utils import degree
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import to_networkx


# ------------------------------------------ model.py ---------------------------------------------
class GraphormerModule(nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_node_dim: int,
                 node_dim: int,
                 input_edge_dim: int,
                 edge_dim: int,
                 output_dim: int,
                 n_heads: int,
                 ff_dim: int,
                 max_in_degree: int,
                 max_out_degree: int,
                 max_path_distance: int):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance

        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                n_heads=self.n_heads,
                ff_dim=self.ff_dim,
                max_path_distance=self.max_path_distance) for _ in range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def forward(self, data: Union[Data]) -> torch.Tensor:
        """
        # batch_size=1 data DataBatch(x=[5, 9], edge_index=[2, 8], edge_attr=[8, 3], smiles=[1], y=[1, 1], batch=[5], ptr=[2])
        # batch_size=10 DataBatch(x=[128, 9], edge_index=[2, 258], edge_attr=[258, 3], smiles=[10], y=[10, 1], batch=[128], ptr=[11])
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        # pdb.set_trace()
        # TODO: data.x -> data.pos
        x = data.pos.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()

        if type(data) == Data:
            ptr = None
            node_paths, edge_paths = shortest_path_distance(data)
        else:
            ptr = data.ptr
            node_paths, edge_paths = batched_shortest_path_distance(data)

        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(x, node_paths)

        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)

        x = self.node_out_lin(x)

        return x


# --------------------------- utils.py -------------------------------
def decrease_to_max_value(x, max_value):
    x[x > max_value] = max_value
    return x

# --------------------------- layers.py -------------------------------
class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """
        num_nodes = x.shape[0]

        in_degree = decrease_to_max_value(degree(index=edge_index[1], num_nodes=num_nodes).long(),
                                          self.max_in_degree - 1)
        out_degree = decrease_to_max_value(degree(index=edge_index[0], num_nodes=num_nodes).long(),
                                           self.max_out_degree - 1)

        x += self.z_in[in_degree] + self.z_out[out_degree]

        return x


class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.max_path_distance = max_path_distance

        self.b = nn.Parameter(torch.randn(self.max_path_distance))

    def forward(self, x: torch.Tensor, paths) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param paths: pairwise node paths
        :return: torch.Tensor, spatial Encoding matrix
        """
        spatial_matrix = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        for src in paths:
            for dst in paths[src]:
                len(paths[src][dst])
                spatial_matrix[src][dst] = self.b[min(len(paths[src][dst]), self.max_path_distance) - 1]

        return spatial_matrix


def dot_product(x1, x2) -> torch.Tensor:
    return (x1 * x2).sum(dim=1)


class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(torch.randn(self.max_path_distance, self.edge_dim))

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_paths) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param edge_paths: pairwise node paths in edge indexes
        :return: torch.Tensor, Edge Encoding matrix
        """
        cij = torch.zeros((x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        for src in edge_paths:
            for dst in edge_paths[src]:
                path_ij = edge_paths[src][dst][:self.max_path_distance]
                weight_inds = [i for i in range(len(path_ij))]
                cij[src][dst] = dot_product(self.edge_vector[weight_inds], edge_attr[path_ij]).mean()

        cij = torch.nan_to_num(cij)
        return cij


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr=None) -> torch.Tensor:
        """
        :param query: node feature matrix
        :param key: node feature matrix
        :param value: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        batch_mask_neg_inf = torch.full(size=(x.shape[0], x.shape[0]), fill_value=-1e6).to(
            next(self.parameters()).device)
        batch_mask_zeros = torch.zeros(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        # OPTIMIZE: get rid of slices: rewrite to torch
        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)
            batch_mask_zeros += 1
        else:
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1

        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        c = self.edge_encoding(x, edge_attr, edge_paths)
        a = self.compute_a(key, query, ptr)
        a = (a + b + c) * batch_mask_neg_inf
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros
        x = softmax.mm(value)
        return x

    def compute_a(self, key, query, ptr=None):
        if type(ptr) == type(None):
            a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        else:
            a = torch.zeros((query.shape[0], query.shape[0]), device=key.device)
            for i in range(len(ptr) - 1):
                a[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = query[ptr[i]:ptr[i + 1]].mm(
                    key[ptr[i]:ptr[i + 1]].transpose(0, 1)) / query.size(-1) ** 0.5

        return a


# FIX: PyG attention instead of regular attention, due to specificity of GNNs
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths,
                ptr) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat([
                attention_head(x, edge_attr, b, edge_paths, ptr) for attention_head in self.heads
            ], dim=-1)
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, ff_dim, max_path_distance):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )
        self.ln_1 = nn.LayerNorm(self.node_dim)
        self.ln_2 = nn.LayerNorm(self.node_dim)
        self.ff = nn.Sequential(
                    nn.Linear(self.node_dim, self.ff_dim),
                    nn.GELU(),
                    nn.Linear(self.ff_dim, self.node_dim)
        )


    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch,
                edge_paths,
                ptr) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new


# -------------------------------------------- functional.py --------------------------------------------
def floyd_warshall_source_to_all(G, source, cutoff=None):
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))

    edges = {edge: i for i, edge in enumerate(G.edges())}

    level = 0  # the current level
    nextlevel = {source: 1}  # list of nodes to check at next level
    node_paths = {source: [source]}  # paths dictionary  (paths to key from source)
    edge_paths = {source: []}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    nextlevel[w] = 1

        level = level + 1

        if (cutoff is not None and cutoff <= level):
            break

    return node_paths, edge_paths


def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths


def shortest_path_distance(data: Data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths


def batched_shortest_path_distance(data) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    all_subgraph_data = data.to_data_list()
    batch_size = len(all_subgraph_data)

    edge_index = data.edge_index.reshape(data.edge_index.shape[0], batch_size, -1)
    for idx in range(batch_size):
        all_subgraph_data[idx].edge_index = edge_index[:, idx, :]
        all_subgraph_data[idx].num_nodes = 0 # data.num_nodes // batch_size

    # original command
    graphs = [to_networkx(sub_data) for sub_data in all_subgraph_data]

    # relabeled_graphs = []
    # shift = 0
    # for i in range(len(graphs)):
    #     num_nodes = graphs[i].number_of_nodes()
    #     relabeled_graphs.append(nx.relabel_nodes(graphs[i], {i: i + shift for i in range(num_nodes)}))
    #     shift += num_nodes

    relabeled_graphs = graphs

    # pdb.set_trace()

    paths = [all_pairs_shortest_path(G) for G in relabeled_graphs]
    node_paths = {}
    edge_paths = {}

    for path in paths:
        for k, v in path[0].items():
            node_paths[k] = v
        for k, v in path[1].items():
            edge_paths[k] = v

    return node_paths, edge_paths




