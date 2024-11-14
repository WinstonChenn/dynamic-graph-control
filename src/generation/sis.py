import sys, math, itertools, random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class DeterministicSIS(): 
    def __init__(self, num_nodes, seed=42,
        lat_dim=2, edge_thresh=0.55, int_param=(1,1),
        init_inf_prop=0.1, inf_thresh=0.2, max_inf_days=10,
        inf_param=(1,1), sus_param=(1,1), rec_param=(1,1)):
        '''
            basic parameters: 
                num_nodes: number of nodes
                seed: random seed
            edge generation parameters:
                lat_dim: dynamic node latent feature dimension
                edge_thresh: edge generation threshold
                int_param: Beta distribution parameter for intervenablness
            state generation parameters:
                init_inf_prop: initial infection proportion
                inf_thresh: infection pressure threshold
                max_inf_days: maximum possible number of infected days
                inf_param: Beta distribution parameter for infectiousnes
                sus_param: Beta distribution parameter for susceptibilty
                rec_param: Beta distribution parameter for recoverability
        '''
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        # basic graph parameters
        self.num_nodes = num_nodes
        self.seed = seed
        # edge generation parameters
        self.lat_dim = lat_dim
        self.edge_thresh = edge_thresh
        self.int_param = int_param
        # node state generation parameters
        self.init_inf_prop = init_inf_prop
        self.inf_thresh = inf_thresh
        self.max_inf_days = max_inf_days
        self.inf_param = inf_param
        self.sus_param = sus_param
        self.rec_param = rec_param

        # timestep counter
        self.t = 0
        # initialize graph
        self.G = nx.Graph()
        self.G.graph["t"] = self.t
        # initialize node features
        sus_list = []
        for i in range(self.num_nodes):
            inf = np.random.beta(*self.inf_param)
            sus = np.random.beta(*self.sus_param)
            rec = np.random.beta(*self.rec_param)
            int = np.random.beta(*self.int_param)
            num_lat = np.random.randint(low=2, high=6)
            lats = np.random.normal(loc=0, scale=1, size=(num_lat, self.lat_dim))
            self.G.add_node(i, state="S", inf=inf, sus=sus, rec=rec, int=int, 
                            lats=lats, curr_lat=lats[0])
            sus_list.append(sus)
        # initialize node state
        infected_idx = np.argsort(sus_list)[::-1][:round(self.num_nodes*self.init_inf_prop)]
        for i in infected_idx: self.G.nodes[i]["state"] = f"I-{self.t}"
        
        # initialize node edges
        for i, j in itertools.combinations(self.G.nodes, 2):
            affinity = sigmoid(self.get_edge_afinity_scores(i, j))
            if affinity > self.edge_thresh: self.G.add_edge(i, j)
    
    def intervene(self, nodes):
        for i in nodes:
            if self.G.nodes[i]['state'].startswith("I-"):
                self.G.nodes[i]['state'] = self.G.nodes[i]['state'].replace("I", "Q")

    def update(self):
        self.t += 1
        self.G.graph["t"] = self.t
        for i in self.G.nodes:
            # update susceptible nodes
            if self.G.nodes[i]["state"] == "S":
                total_inf = 0
                for j in self.G.neighbors(i):
                    if self.G.nodes[j]["state"].startswith("I"):
                        total_inf+=self.G.nodes[j]["inf"]
                if total_inf * self.G.nodes[i]["sus"] > self.inf_thresh:
                    self.G.nodes[i]["state"] = f"I-{self.t}"
            # update infected nodes
            elif self.G.nodes[i]['state'].startswith("I-") or \
                 self.G.nodes[i]['state'].startswith("Q-"):
                inf_time = self.t - int(self.G.nodes[i]['state'].split("-")[1])
                assert inf_time > 0
                rec_time = (1-self.G.nodes[i]["rec"]) * self.max_inf_days
                if inf_time >= rec_time:
                    self.G.nodes[i]['state'] = "S"
        
        # update dynamic features
        for i in self.G.nodes:
            self.G.nodes[i]["curr_lat"] = self.G.nodes[i]["lats"][self.t%self.G.nodes[i]["lats"].shape[0]]

        # update dynamic edges
        for i, j in itertools.combinations(self.G.nodes, 2):
            affinity = sigmoid(self.get_edge_afinity_scores(i, j))
            if self.G.nodes[i]["state"].startswith("Q-"):
                affinity *= (1-self.G.nodes[i]["int"])
            if self.G.nodes[j]["state"].startswith("Q-"):
                affinity *= (1-self.G.nodes[j]["int"])
            if self.G.has_edge(i, j) and affinity <= self.edge_thresh:
                self.G.remove_edge(i, j)
            elif not self.G.has_edge(i, j) and affinity > self.edge_thresh: 
                self.G.add_edge(i, j)

    ### Utils ###
    def get_rec_times(self):
        rec_times = []
        for i in self.G.nodes:
            rec_times.append(self.d*(1-self.G.nodes[i]["rec"]))
        return rec_times
    
    def get_edge_afinity_scores(self, i, j):
        return get_edge_afinity_scores(self.G, i, j)

    def get_num_nodes_in_state(self, state):
        count = 0
        for i in self.G.nodes:
            if self.G.nodes[i]['state'].startswith(state):
                count += 1
        return count

def get_edge_afinity_scores(G, i, j):
    lat_i = G.nodes[i]["curr_lat"]
    lat_j = G.nodes[j]["curr_lat"]
    return np.dot(lat_i, lat_j) / np.sqrt(len(lat_i))

def get_all_affinity_scores(G):
        affinity_scores = []
        for i, j in itertools.combinations(G.nodes, 2):
            affinity = get_edge_afinity_scores(G, i, j)
            affinity_scores.append(affinity)
        return affinity_scores
    
def get_num_nodes_in_state(G, state):
    count = 0
    for i in G.nodes:
        if G.nodes[i]['state'].startswith(state):
            count += 1
    return count

def get_unintervened_node_index(G):
    idx = []
    for i in G.nodes:
        if not G.nodes[i]["state"].startswith("Q"):
            idx.append(i)
    return idx

def get_node_feature(G, i):
    feat = [G.nodes[i]["inf"], G.nodes[i]["sus"], 
            G.nodes[i]["rec"], G.nodes[i]["int"]]
    if G.nodes[i]["state"].startswith("S"):
        feat += [0]
    else:
        feat += [G.graph["t"] - int(G.nodes[i]["state"].split("-")[1]) + 1]
    return feat

def get_node_feature_matrix(G):
    features = []
    for i in G.nodes:
        features.append(np.array(get_node_feature(G, i)))
    return np.stack(features)

def get_node_latent_feature_matrix(G):
    features = []
    for i in G.nodes:
        features.append(np.array(G.nodes[i]["x"]))
    return np.stack(features)

def get_node_label_vector(G):
    labels = []
    for i in G.nodes:
        if G.nodes[i]["state"].startswith("S"):
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)

def get_edge_index(G):
    edge_index = list(G.edges)
    edge_index += [(e[1], e[0]) for e in edge_index] 
    return edge_index

def get_decode_index_label(G):
    decode_index, labels = [], []
    for i, j in itertools.permutations(G.nodes, 2):
        labels.append(int(G.has_edge(i, j)))
        decode_index.append((i, j))
    return decode_index, labels

def get_edge_feature_matrix(G):
    features = []
    for i, j in itertools.combinations(G.nodes, 2):
        features.append(np.array(get_node_feature(G, i) + 
            get_node_feature(G, j)))
    return np.stack(features)
        