import torch, itertools
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class NodeGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=16):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

class EdgeGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=16):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.mlp =  nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), 
            nn.ReLU(), nn.Linear(hidden_dim, 1))
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        if batch is not None:
            x_arr = []
            for b in batch.unique():
                curr_mask = batch==b
                aff_mat = torch.matmul(x[curr_mask], x[curr_mask].T)
                curr_x_arr = []
                for i, j in itertools.combinations(range(curr_mask.sum().item()), 2):
                    curr_x_arr.append(aff_mat[i, j])
                x_arr += curr_x_arr
        else:
            aff_mat = torch.matmul(x, x.T)
            x_arr = []
            for i, j in itertools.combinations(range(x.shape[0]), 2):
                x_arr.append(aff_mat[i, j])
        x = torch.stack(x_arr).reshape(-1, 1)
        return torch.sigmoid(x)