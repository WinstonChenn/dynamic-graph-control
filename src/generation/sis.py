import sys, math, itertools, warnings
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class DeterministicSIS(): 
    def __init__(self, n, p, gamma=0.1, d=10, tau=0.5, delta=0.5, inf_alpha=1, inf_beta=1, sus_alpha=1, sus_beta=1, 
                 rec_alpha=1, rec_beta=1, int_alpha=1, int_beta=1):
        '''
            inputs: 
                n: number of nodes
                p: dynamic node feature dimension
                gamma: initial infected proportion
                d: maximum possible number of infected days
                tau: infection pressure threshold
                delta: edge generation threshold
                inf_alpha, inf_beta: Beta distribution parameter for infectiousnes
                sus_alpha, sus_beta: Beta distribution parameter for susceptibilty
                rec_alpha, rec_beta: Beta distribution parameter for recoverability
                int_alpha, int_beta: Beta distribution parameter for intervenablness
        '''
        self.n = n
        self.p = p
        self.gamma = gamma
        self.d = d
        self.tau = tau
        self.delta = delta

        self.inf_alpha, self.inf_beta = inf_alpha, inf_beta
        self.sus_alpha, self.sus_beta = sus_alpha, sus_beta
        self.rec_alpha, self.rec_beta = rec_alpha, rec_beta
        self.int_alpha, self.int_beta = int_alpha, int_beta

        # time counter
        self.t = 0

        # dynamic feature (x) update function
        self.W = np.random.uniform(low=-1, high=1, size=(p,p))
        # self.W = np.random.normal(loc=-1, scale=1, size=(p,p))
        # self.W = self.W / self.W.sum(axis=1)[:, np.newaxis]
        
        # initialize graph
        self.G = nx.Graph()
        # initialize node features
        sus_list = []
        for i in range(n):
            inf = np.random.beta(self.inf_alpha, self.inf_beta)
            sus = np.random.beta(self.sus_alpha, self.sus_beta)
            rec = np.random.beta(self.rec_alpha, self.rec_beta)
            int = np.random.beta(self.int_alpha, self.int_beta)
            # x = np.random.normal(loc=0, scale=1, size=p)
            x = np.random.uniform(low=-np.pi*2, high=np.pi*2, size=p)
            self.G.add_node(i, state="S", inf=inf, sus=sus, rec=rec, int=int, x=x)
            sus_list.append(sus)
        # initialize node state
        infected_idx = np.argsort(sus_list)[::-1][:round(n*self.gamma)]
        for i in infected_idx:
            self.G.nodes[i]["state"] = f"I-{self.t}"
        
        # initialize node edges
        for i, j in itertools.combinations(self.G.nodes, 2):
            affinity = sigmoid(self.get_edge_afinity_scores(i, j))
            if affinity > delta: self.G.add_edge(i, j)
    
    def intervene(self, nodes):
        for i in nodes:
            if self.G.nodes[i]['state'].startswith("I-"):
                self.G.nodes[i]['state'] = self.G.nodes[i]['state'].replace("I", "Q")

    def update(self):
        self.t += 1
        for i in self.G.nodes:
            # update susceptible nodes
            if self.G.nodes[i]["state"] == "S":
                total_inf = 0
                for j in self.G.neighbors(i):
                    if self.G.nodes[j]["state"].startswith("I"):
                        total_inf+=self.G.nodes[j]["inf"]
                if total_inf * self.G.nodes[i]["sus"] > self.tau:
                    self.G.nodes[i]["state"] = f"I-{self.t}"
            # update infected nodes
            elif self.G.nodes[i]['state'].startswith("I-") or \
                 self.G.nodes[i]['state'].startswith("Q-"):
                inf_time = self.t - int(self.G.nodes[i]['state'].split("-")[1])
                assert inf_time > 0
                rec_time = (1-self.G.nodes[i]["rec"]) * self.d
                if inf_time >= rec_time:
                    self.G.nodes[i]['state'] = "S"
        
        # update dynamic features
        for i in self.G.nodes:
            new_x = np.pi*2*np.sin(self.G.nodes[i]["x"])
            self.G.nodes[i]["x"] = new_x

        # update dynamic edges
        for i, j in itertools.combinations(self.G.nodes, 2):
            affinity = sigmoid(self.get_edge_afinity_scores(i, j))
            # print(f"edge: {i}-{j}, affinity: {affinity}, {np.dot(self.G.nodes[i]["x"], self.G.nodes[j]["x"])}")
            if self.G.has_edge(i, j):
                if affinity <= self.delta: self.G.remove_edge(i, j)
            elif affinity > self.delta: self.G.add_edge(i, j)
    

    ### Utils ###
    def get_rec_times(self):
        rec_times = []
        for i in self.G.nodes:
            rec_times.append(self.d*(1-self.G.nodes[i]["rec"]))
        return rec_times
    
    def get_edge_afinity_scores(self, i, j):
            x_i, x_j = self.G.nodes[i]["x"], self.G.nodes[j]["x"]
            return np.dot(x_i, x_j)/(np.linalg.norm(x_i)*np.linalg.norm(x_j))
            # return np.dot(x_i, x_j)/np.sqrt(self.p)
    
    # /np.sqrt(self.p)

    def get_all_affinity_scores(self):
        affinity_scores = []
        for i, j in itertools.combinations(self.G.nodes, 2):
            affinity = self.get_edge_afinity_scores(i, j)
            affinity_scores.append(affinity)
        return affinity_scores

    def get_num_nodes_in_state(self, state):
        count = 0
        for i in self.G.nodes:
            if self.G.nodes[i]['state'].startswith(state):
                count += 1
        return count



        