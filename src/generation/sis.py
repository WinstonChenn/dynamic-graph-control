import sys, math, itertools, random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class DeterministicSIS(): 
    def __init__(self, num_nodes, seed=42,
        lat_dim=2, edge_thresh=0.55, int_param=None,
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
            if self.int_param is None: int = 0.9
            else: int = np.random.beta(*self.int_param)
            num_lat = np.random.randint(low=2, high=6)
            lats = np.random.normal(loc=0, scale=1, size=(num_lat, self.lat_dim))
            self.G.add_node(i, state="S", inf=inf, sus=sus, rec=rec, int=int, lats=lats)
            sus_list.append(sus)
        # initialize node state
        infected_idx = np.argsort(sus_list)[::-1][:round(self.num_nodes*self.init_inf_prop)]
        for i in infected_idx: self.G.nodes[i]["state"] = f"I-{self.t}"
        # initialize node edges
        self.update_edges()
    
    def intervene(self, nodes):
        for i in nodes:
            if self.G.nodes[i]['state'].startswith("I-"):
                self.G.nodes[i]['state'] = self.G.nodes[i]['state'].replace("I", "Q")

    def update(self):
        self.t += 1
        self.G.graph["t"] = self.t
        self.update_nodes()
        self.update_edges()
    
    def update_nodes(self):
        for i in self.G.nodes:
            # update susceptible nodes
            if self.G.nodes[i]["state"] == "S":
                # count total number of infectiousness from infected neighbors
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

    def update_edges(self):
        self.G.clear_edges()
        for i in self.G.nodes:
            curr_lat_idx = self.t%self.G.nodes[i]["lats"].shape[0]
            self.G.nodes[i]["curr_lat"] = self.G.nodes[i]["lats"][curr_lat_idx]
        node_lats = get_all_node_attribute(self.G, feature_name='curr_lat')
        node_pairs = np.stack(list(itertools.combinations(self.G.nodes, 2)))
        aff_scores = np.sum(node_lats[node_pairs[:, 0]]*node_lats[node_pairs[:, 1]], axis=1)
        # compute affinity score modification based on whether node is in Q state
        node_in_q = np.array([s.startswith("Q-") for s in get_all_node_attribute(self.G, feature_name='state')]).astype(int)
        node_int = get_all_node_attribute(self.G, feature_name='int')
        node_aff_mod = (1-node_int*node_in_q)
        aff_scores = sigmoid(aff_scores * node_aff_mod[node_pairs[:, 0]] * node_aff_mod[node_pairs[:, 1]])
        self.G.graph['aff_scores'] = aff_scores
        pos_pairs = node_pairs[aff_scores > self.edge_thresh]
        self.G.add_edges_from(pos_pairs)

### Utils ###
def get_all_node_attribute(G, feature_name):
    node_features = []
    for i in sorted(G.nodes):
        node_features.append(G.nodes[i][feature_name])
    return np.stack(node_features)

def get_nodes_in_state(G, state, invert=False):
    nodes = []
    for i in G.nodes:
        condition = G.nodes[i]['state'].startswith(state)
        if invert: condition = not condition
        if condition: nodes.append(i)
    return nodes

def get_node_pred_feature(G, i):
    feat = [G.nodes[i]["inf"], G.nodes[i]["sus"], 
            G.nodes[i]["rec"], G.nodes[i]["int"]]
    if G.nodes[i]["state"].startswith("S"):
        feat += [0]
    else:
        feat += [G.graph["t"] - int(G.nodes[i]["state"].split("-")[1]) + 1]
    return feat

def get_node_pred_feature_matrix(G):
    features = []
    for i in G.nodes:
        features.append(np.array(get_node_pred_feature(G, i)))
    return np.stack(features)

def get_node_pred_label_vector(G):
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

def get_all_decode_index_label(G):
    decode_index, labels = [], []
    for i, j in itertools.permutations(G.nodes, 2):
        labels.append(int(G.has_edge(i, j)))
        decode_index.append((i, j))
    return decode_index, labels
        