import torch, itertools
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, GPSConv
from torch_geometric.utils import to_dense_batch

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()

        # Add the first layer (input to first hidden layer)
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())

        # Add the hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(nn.ReLU())

        # Add the output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class NodeGCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=16):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, data, x_label="x", edge_index_label="edge_index"):
        x, edge_index = data[x_label], data[edge_index_label]
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
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, data, x_label="x", edge_index_label="edge_index", decode_index=None):
        x, edge_index = data[x_label], data[edge_index_label]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        if decode_index is not None:
            x = self.cos(x[decode_index[0, :]], x[decode_index[1, :]])
        return torch.sigmoid(x)
    

class NodeSAGE(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=128):
        super().__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_dim, project=True)
        self.conv2 = SAGEConv(hidden_dim, 1, project=True)

    def forward(self, data, x_label="x", edge_index_label="edge_index"):
        x, edge_index = data[x_label], data[edge_index_label]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x) 
    
class EdgeSAGE(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=128):
        super().__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, data, decode_index=None, x_label="x", edge_index_label="edge_index"):
        x, edge_index = data[x_label], data[edge_index_label]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        if decode_index is not None:
            x = self.cos(x[decode_index[0, :]], x[decode_index[1, :]])
            
        return torch.sigmoid(x)