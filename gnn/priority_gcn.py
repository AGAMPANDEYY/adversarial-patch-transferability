# gnn/priority_gcn.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PriorityGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        # two GCN layers + final MLP to scalar
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin   = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        # produce one score per node
        return self.lin(h).squeeze(-1)  # [num_nodes]
