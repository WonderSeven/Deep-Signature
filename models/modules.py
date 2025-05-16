import pdb
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear


class DenseGraphConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GraphConv`."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'add',
        bias: bool = True,
    ):
        assert aggr in ['add', 'mean', 'max']
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.lin_rel = Linear(in_channels, out_channels, bias=bias)
        self.lin_root = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(self, x: Tensor, adj: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        B, T, N, C = x.size()

        if self.aggr == 'add':
            out = torch.einsum('btjk,bij->btik', x, adj)
        elif self.aggr == 'mean':
            out = torch.einsum('btjk,bij->btik', x, adj)
            out = out / adj.sum(dim=-1, keepdim=True).clamp_(min=1).unsqueeze(1).repeat(1, T, 1, 1)
        elif self.aggr == 'max':
            out = x.unsqueeze(-2).repeat(1, 1, 1, N, 1)
            adj = adj.unsqueeze(1).unsqueeze(-1).expand(B, T, N, N, C)
            out[adj == 0] = float('-inf')
            out = out.max(dim=-3)[0]
            out[out == float('-inf')] = 0.
        else:
            raise NotImplementedError

        out = self.lin_rel(out)
        out = out + self.lin_root(x)

        if mask is not None:
            out = out * mask.view(-1, N, 1).to(x.dtype)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')


def dense_mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s
    # x:[16, 2, 1384, 10] adj:[16, 1384, 1384], s1:[16, 1384, 100], mask:[16, 1384]
    (batch_size, time_stamps, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        # mask = mask.view(batch_size, time_stamps, num_nodes, 1).to(x.dtype)
        # x, s = x * mask, s * mask
        mask = mask.unsqueeze(-1).to(x.dtype)
        s = s * mask
        mask = mask.unsqueeze(1).repeat(1, time_stamps, 1, 1)
        x = x * mask

    out = torch.einsum('btik,bij->btjk', x, s) # [48, 25, 16, 10]
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss


def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum('ijj->i', x)


def _rank3_diag(x: Tensor) -> Tensor:
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))

    return out
