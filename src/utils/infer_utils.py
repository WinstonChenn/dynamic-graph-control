import torch, itertools
from copy import deepcopy
import numpy as np
from torch_geometric.data import Data
from generation.sis import DeterministicSIS, get_edge_index, get_node_pred_feature_matrix
from utils import train_utils


def nx2pyg(G, device="cpu", pred_edge=False):
    edge_index_torch = torch.from_numpy(np.array(get_edge_index(G)).astype(int).T).to(device)
    if pred_edge:
        x_node_torch = torch.from_numpy(np.identity(len(G.nodes)).astype(np.float32)).to(device)
        decode_index = torch.from_numpy(np.array(list(itertools.permutations(G.nodes, 2))).astype(int).T).to(device)
        return train_utils.PairData(x=x_node_torch, edge_index=edge_index_torch, decode_index=decode_index)
    else:
        x_node_torch = torch.from_numpy(get_node_pred_feature_matrix(G).astype(np.float32)).to(device)
        return Data(x=x_node_torch, edge_index=edge_index_torch)
    
def pairs2adj(index_pairs, values):
    adj_mat = np.zeros((index_pairs.max()+1, index_pairs.max()+1))
    for idx, (i, j) in enumerate(index_pairs.T):
        adj_mat[i, j] = values[idx]
    assert np.all(np.diagonal(adj_mat) == 0)
    assert np.all(adj_mat[~np.eye(adj_mat.shape[0], dtype=bool)] != 0)
    return adj_mat

def forecast(G, node_model, edge_model, device="cpu", num_times=2):
    node_scores_list, adj_mats_list = [], []
    G_curr = deepcopy(G)
    for _ in range(num_times):
        # predict next node labels
        G_node_data = nx2pyg(G_curr, device, pred_edge=False)
        node_scores = node_model(G_node_data).detach().cpu().numpy().reshape(-1)
        node_scores_list.append(node_scores)
        node_pred = np.round(node_scores)
        # predict next edge labels
        G_edge_data = nx2pyg(G_curr, device, pred_edge=True)
        edge_scores = edge_model(G_edge_data, decode_index=G_edge_data.decode_index).detach().cpu().numpy().reshape(-1)
        adj_mat = pairs2adj(G_edge_data.decode_index, edge_scores)
        adj_mats_list.append(adj_mat)
        edge_pred = np.round(edge_scores)
        edge_list = G_edge_data.decode_index.detach().cpu().numpy().astype(int)[:, edge_pred==1].T
        # create next predicted graph
        G_next = deepcopy(G_curr)
        G_next.graph["t"] = G_next.graph["t"] + 1
        G_next.clear_edges()
        for edge in edge_list: 
            G_next.add_edge(*edge)
        for i in range(len(node_pred)):
            if node_pred[i] == 0:
                G_next.nodes[i]["state"] = "S"
            else:
                if G_curr.nodes[i]["state"] == "S":
                    G_next.nodes[i]["state"] = f"I-{G_next.graph["t"]}"
                elif G_curr.nodes[i]["state"].startswith("I") or G_curr.nodes[i]["state"].startswith("Q"):
                    G_next.nodes[i]["state"] = G_curr.nodes[i]["state"]
                else: raise Exception()
        G_curr = deepcopy(G_next)
    return np.stack(node_scores_list), np.stack(adj_mats_list)