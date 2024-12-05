import torch, itertools
from copy import deepcopy
import numpy as np
from torch_geometric.data import Data
from generation.sis import DeterministicSIS, get_edge_index, get_X_matrix, get_A_matrix, \
    get_Y_vector
from utils import train_utils


def nx2pyg(G, device="cpu", pred_edge=False, dynamic=False, intervention=[]):
    if dynamic:
        X_curr = torch.from_numpy(get_X_matrix(G, time=True).astype(np.float32)).to(device)
        A_curr = torch.from_numpy(get_A_matrix(G).astype(np.float32)).to(device)
        edge_index = torch.from_numpy(np.array(get_edge_index(G)).astype(int).T).to(device)
        Y_curr = torch.from_numpy(get_Y_vector(G).reshape(-1, 1).astype(int)).to(device)
        T_curr = torch.zeros((X_curr.shape[0], 1)).to(device)
        T_curr[intervention, 0] = 1

        return Data(X=X_curr, edge_index=edge_index, A_curr=A_curr, Y_curr=Y_curr, T=T_curr)


    else:
        edge_index_torch = torch.from_numpy(np.array(get_edge_index(G)).astype(int).T).to(device)
        if pred_edge:
            x_node_torch = torch.from_numpy(np.identity(len(G.nodes)).astype(np.float32)).to(device)
            decode_index = torch.from_numpy(np.array(list(itertools.permutations(G.nodes, 2))).astype(int).T).to(device)
            return train_utils.PairData(x=x_node_torch, edge_index=edge_index_torch, decode_index=decode_index)
        else:
            x_node_torch = torch.from_numpy(get_X_matrix(G).astype(np.float32)).to(device)
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
        for i in range(len(node_pred)):
            if node_pred[i] == 0:
                G_next.nodes[i]["state"] = "S"
            else:
                if G_curr.nodes[i]["state"] == "S":
                    G_next.nodes[i]["state"] = f"I-{G_next.graph["t"]}"
                elif G_curr.nodes[i]["state"].startswith("I") or G_curr.nodes[i]["state"].startswith("Q"):
                    G_next.nodes[i]["state"] = G_curr.nodes[i]["state"]
                else: raise Exception()
        for edge in edge_list: 
            if not (G_next.nodes[edge[0]]["state"].startswith("Q") or 
                G_next.nodes[edge[1]]["state"].startswith("Q")):
                    G_next.add_edge(*edge)
        G_curr = deepcopy(G_next)
    return np.stack(node_scores_list), np.stack(adj_mats_list)

def dynamic_forecast(G, model, device="cpu", num_times=2, intervention=[], num_mc=1, 
        input_label_dict={"x_label":"X", "y_label":"Y_curr", "edge_index_label":"edge_index"}):
    node_scores_lists, adj_scores_lists = [], []
    for _ in range(num_mc):
        node_scores_list, adj_scores_list = [], []
        G_curr = deepcopy(G)
        y_hidden, a_hidden = None, None
        for num in range(num_times):
            # predict next node labels
            data = nx2pyg(G_curr, device, dynamic=True, intervention=intervention if num == 0 else [])
            y_score, a_score, y_hidden, a_hidden = model(data, y_hidden=y_hidden, a_hidden=a_hidden, **input_label_dict)
            a_score = torch.sigmoid(a_score)
            node_scores = y_score.detach().cpu().numpy()
            node_scores_list.append(node_scores)
            # sample node predictions
            adj_scores = a_score.detach().cpu().numpy()
            if num_mc == 1: 
                node_pred = np.argmax(node_scores, axis=1)
                adj_pred = np.round(adj_scores)
            else: 
                node_cumsum = np.cumsum(node_scores, axis=1)
                node_pred = (np.random.rand(node_scores.shape[0])[:, None] < node_cumsum).argmax(axis=1)
                adj_pred = np.random.binomial(n=1, p=adj_scores)
            
            # predict next edge labels
            adj_scores_list.append(adj_scores)
            edge_list = []
            for u, v in itertools.permutations(G.nodes, 2):
                if adj_pred[u, v] == 1: edge_list.append((u, v))
            
            # create next predicted graph
            G_next = deepcopy(G_curr)
            G_next.graph["t"] = G_next.graph["t"] + 1
            G_next.clear_edges()
            for i in range(len(node_pred)):
                if node_pred[i] == 0:
                    if G_curr.nodes[i]["state"].startswith("S"):
                        G_next.nodes[i]["state"] = G_curr.nodes[i]["state"]
                    elif G_curr.nodes[i]["state"].startswith("I"):
                        G_next.nodes[i]["state"] = f"S-{G_next.graph["t"]}"
                    elif G_curr.nodes[i]["state"].startswith("Q"):
                        G_next.nodes[i]["state"] = f"S-{G_next.graph["t"]}"
                elif node_pred[i] == 1:
                    if G_curr.nodes[i]["state"].startswith("S"):
                        G_next.nodes[i]["state"] = f"I-{G_next.graph["t"]}"
                    elif G_curr.nodes[i]["state"].startswith("I"):
                        G_next.nodes[i]["state"] = G_curr.nodes[i]["state"]
                    elif G_curr.nodes[i]["state"].startswith("Q"):
                        G_next.nodes[i]["state"] = G_curr.nodes[i]["state"]
                elif node_pred[i] == 2:
                    if G_curr.nodes[i]["state"].startswith("I"):
                        G_next.nodes[i]["state"] = G_curr.nodes[i]["state"].replace("I", "Q")
                    elif G_curr.nodes[i]["state"].startswith("Q"):
                        G_next.nodes[i]["state"] = G_curr.nodes[i]["state"]
                    elif G_curr.nodes[i]["state"].startswith("S"):
                        G_next.nodes[i]["state"] = G_curr.nodes[i]["state"]
            for edge in edge_list: 
                if not (G_next.nodes[edge[0]]["state"].startswith("Q") or 
                    G_next.nodes[edge[1]]["state"].startswith("Q")):
                        G_next.add_edge(*edge)
            G_curr = deepcopy(G_next)
        node_scores_list = np.stack(node_scores_list)
        adj_scores_list = np.stack(adj_scores_list)
        node_scores_lists.append(node_scores_list)
        adj_scores_lists.append(adj_scores_list)
    return np.stack(node_scores_lists).mean(axis=0), np.stack(adj_scores_lists).mean(axis=0)
