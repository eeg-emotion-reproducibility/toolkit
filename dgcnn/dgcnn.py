"""DGCNN (Song et al., "EEG emotion recognition using dynamical graph
convolutional neural networks", IEEE Trans. Affective Computing, 2018).

Graph convolution with a learnable adjacency matrix over the electrodes and a
Chebyshev polynomial expansion of order ``num_layers``. The building blocks
follow the TorchEEG implementation of DGCNN.

Input:  [batch, num_electrodes, in_channels]  (in_channels = frequency bands)
Output: [batch, num_classes] logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        return out


class Linear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def normalize_A(A: torch.Tensor, symmetry: bool = False) -> torch.Tensor:
    """Normalize adjacency matrix with D^{-1/2} A D^{-1/2}."""
    A = F.relu(A)
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt(d + 1e-10)
    D = torch.diag_embed(d)
    L = torch.matmul(torch.matmul(D, A), D)
    return L


def generate_cheby_adj(A: torch.Tensor, num_layers: int) -> list:
    """Generate Chebyshev polynomial supports [T_0, T_1, ..., T_k]."""
    support = []
    for i in range(num_layers):
        if i == 0:
            support.append(torch.eye(A.shape[1]).to(A.device))
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support


class Chebynet(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, out_channels: int):
        super(Chebynet, self).__init__()
        self.num_layers = num_layers
        self.gc1 = nn.ModuleList()
        for i in range(num_layers):
            self.gc1.append(GraphConvolution(in_channels, out_channels))

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        adj = generate_cheby_adj(L, self.num_layers)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result = result + self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


class DGCNN(nn.Module):
    """DGCNN with a learnable adjacency matrix.

    Args:
        in_channels (int): features per electrode (number of DE bands).
        num_electrodes (int): number of EEG channels.
        num_layers (int): Chebyshev order K.
        hid_channels (int): graph-convolution output dimension.
        num_classes (int): number of output classes.
        dropout (float): dropout before the output layer.
    """

    def __init__(self,
                 in_channels: int = 5,
                 num_electrodes: int = 32,
                 num_layers: int = 2,
                 hid_channels: int = 64,
                 num_classes: int = 2,
                 dropout: float = 0.5):
        super(DGCNN, self).__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_layers = num_layers
        self.num_classes = num_classes

        # The creation order of the sub-modules fixes the order in which the
        # random initialisers draw numbers; it is kept identical to the
        # implementation that produced the reported results.
        self.layer1 = Chebynet(in_channels, num_layers, hid_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = Linear(num_electrodes * hid_channels, 64)
        self.fc2 = Linear(64, num_classes)

        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes))
        nn.init.xavier_normal_(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = self.dropout(F.relu(self.fc1(result)))
        result = self.fc2(result)
        return result
