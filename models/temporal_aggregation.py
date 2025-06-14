import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import signatory


class LSTMAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super(LSTMAggregator, self).__init__()
        self.hidden_channels = hidden_channels
        self.bi_lstm_layer = nn.LSTM(in_channels, hidden_channels, 1, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_channels * 2, out_channels)
        self.norm = nn.LayerNorm(hidden_channels * 2) # signature_dim

    def forward(self, x):
        """
        :param x: [48, 50, 25, 20]
        :return:
        """
        batch_size, time_steps, num_nodes = x.shape[:3]
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(batch_size * num_nodes, *x.shape[2:])

        out, _ = self.bi_lstm_layer(x)
        frontal_last = out[:, -1, :self.hidden_channels]
        backward_last = out[:, 0, self.hidden_channels:]
        out = torch.cat((frontal_last, backward_last), dim=-1)
        # out = self.norm(out)
        out = self.linear(out)
        out = out.view(batch_size, num_nodes, -1)
        return out


class SubSignatureAggregator(nn.Module):
    """
    Temporal aggregation the trajectory with Signature.
    """
    def __init__(self, in_channels, out_channels, num_clusters, signature_depth=2, num_segments=4, dropout=0.2):
        super(SubSignatureAggregator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_clusters = num_clusters
        self.signature_depth = signature_depth
        self.num_segments = num_segments
        self.dropout = dropout
        signature_dim = int(math.pow(num_clusters * in_channels, signature_depth+1) / (num_clusters*in_channels - 1)) - 1
        self.signature_dim = signature_dim // 2  # 1136275
        linear_hidden_dim = 256

        self.lstm_hidden_dim = 256
        self.lstm_layer = nn.LSTM(self.signature_dim, self.lstm_hidden_dim, 1, bidirectional=False, batch_first=True)

        self.linear = nn.Sequential(nn.Linear(self.lstm_hidden_dim, self.out_channels),
                                    nn.Tanh())
        self.norm = nn.LayerNorm(self.lstm_hidden_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        batch_size, time_span, n_nodes = x.shape[:3]

        # Flatten the xyz
        if x.dim() == 4:
            x = x.contiguous().view(batch_size * self.num_segments, time_span // self.num_segments, -1)

        path_sig = signatory.logsignature(x, self.signature_depth) # 2:[16, 11325]  3:[16, 1136275]
        path_sig = path_sig.view(batch_size, self.num_segments, -1)
        path_sig, _ = self.lstm_layer(path_sig)
        path_sig = self.norm(path_sig)

        out = self.linear(path_sig[:, -1, :])
        out = self.dropout(out)

        return out