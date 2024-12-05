import torch, itertools
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, GPSConv
from torch_geometric.utils import to_dense_batch
from models.StaticGNN import MLP

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, softmax=True):
        hidden = self.initHidden()
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        if softmax:
            output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(next(self.parameters()).device)

class NodeSAGERNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=128, output_size=3):
        super().__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_dim, project=True)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, project=True)
        self.rnn = RNN(input_size=hidden_dim, hidden_size=hidden_dim, output_size=output_size)

    def forward(self, data, hidden=None, x_label="x", edge_index_label="edge_index"):
        x, edge_index = data[x_label], data[edge_index_label]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return self.rnn(x, hidden)
    
    def initHidden(self):
        return self.rnn.initHidden()
    
class NodeSAGELSTM(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=128, output_size=3):
        super().__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_dim, project=True)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, project=True)
        self.rnn = nn.LSTM(input_size=num_node_features, hidden_size=hidden_dim, proj_size=output_size)

    def forward(self, data, hidden=None, cell=None, x_label="x", edge_index_label="edge_index"):
        x, edge_index = data[x_label], data[edge_index_label]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        breakpoint()
        return self.rnn(x, hidden, cell)

class EdgeSAGERNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=128):
        super().__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.rnn = RNN(input_size=hidden_dim, hidden_size=hidden_dim, output_size=hidden_dim)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, data, hidden=None, decode_index=None, x_label="x", edge_index_label="edge_index"):
        x, edge_index = data[x_label], data[edge_index_label]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x, hidden = self.rnn(x, hidden, softmax=False)
        if decode_index is not None:
            x = self.cos(x[decode_index[0, :]], x[decode_index[1, :]])
        return torch.sigmoid(x), hidden

    def initHidden(self):
        return self.rnn.initHidden()
    

    
class SAGELSTM(torch.nn.Module):
    def __init__(self, num_nodes, num_X_features, hidden_dim=128, num_y_categories=3):
        super().__init__()
        self.y_conv1 = SAGEConv(num_X_features+1, hidden_dim) # adding Y and I as feature
        self.y_conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.a_conv1 = SAGEConv(num_nodes, hidden_dim) # adding Y and I as feature
        self.a_conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.y_rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim)
        self.y_t0_mlp = MLP(input_size=hidden_dim, hidden_sizes=[], output_size=num_y_categories)
        self.y_t1_mlp = MLP(input_size=hidden_dim, hidden_sizes=[], output_size=num_y_categories)
        self.a_rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim)
        self.mlp = MLP(input_size=num_X_features, hidden_sizes=[hidden_dim], output_size=hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, data, y_hidden=None, a_hidden=None,
                x_label="X", a_label="A_curr", t_label="T", y_label="Y_curr", 
                edge_index_label="edge_index"):
        X, edge_index = data[x_label], data[edge_index_label]
        A, T, Y = data[a_label], data[t_label], data[y_label]
        X_id = torch.eye(A.shape[0]).to(X.device)
        
        X_y = torch.hstack((X, Y))
        X_y = self.y_conv1(X_y, edge_index)
        X_y = F.relu(X_y)
        X_y = F.dropout(X_y, training=self.training)
        X_y = self.y_conv2(X_y, edge_index)

        # predict y
        y_score, y_hidden= self.y_rnn(X_y, y_hidden)
        y_t0_score = self.softmax(self.y_t0_mlp(y_score))
        y_t1_score = self.softmax(self.y_t1_mlp(y_score))
        y_score = y_t0_score * (1-T) + y_t1_score * T
        
        # predict a
        X_a = self.a_conv1(X_id, edge_index)
        X_a = F.relu(X_a)
        X_a = F.dropout(X_a, training=self.training)
        X_a = self.y_conv2(X_a, edge_index)
        X_a, a_hidden= self.a_rnn(X_a, a_hidden)
        X_a = X_a * (self.mlp(X) * (torch.argmax(y_score, axis=1)==2).reshape(-1, 1) + \
            torch.ones_like(X_a).to(X_a.device) * (torch.argmax(y_score, axis=1)!=2).reshape(-1, 1))
        X_a = F.normalize(X_a, p=2, dim=1)
        a_score = X_a @ X_a.T
        a_score = a_score

        return y_score, a_score, y_hidden, a_hidden
    
    def flatten_parameters(self):
        self.y_rnn.flatten_parameters()
        self.a_rnn.flatten_parameters()