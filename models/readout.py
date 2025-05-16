import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self, in_channesl, reduction='mean'):
        super(BinaryClassifier, self).__init__()
        self.reduction = reduction
        self.linear = nn.Linear(in_channesl, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.reduction == 'mean':
            x = x.mean(dim=-2)
        elif self.reduction == 'sum':
            x = x.sum(dim=-2)

        return self.linear(x).sigmoid()


# class BinaryClassifier(nn.Module):
#     def __init__(self, in_channesl, reduction='mean'):
#         super(BinaryClassifier, self).__init__()
#         self.reduction = reduction
#         self.linear = nn.Linear(in_channesl, 2)
#         # self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         if self.reduction == 'mean':
#             x = x.mean(dim=1)
#         elif self.reduction == 'sum':
#             x = x.sum(dim=1)
#
#         return self.linear(x)