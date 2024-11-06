import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def smooth(signal, window_size=10):
    """Convolve two 1D arrays with 'same' mode and no boundary effects."""

    # Calculate padding needed to avoid boundary effects
    kernel = [1/window_size] * window_size
    kernel_size = len(kernel)
    padding = kernel_size // 2

    # Pad the signal with the mean value to minimize edge effects
    padded_signal = np.pad(signal, padding, mode='reflect') 

    # Perform the convolution
    result = np.convolve(padded_signal, kernel, mode='valid')

    return result

def plot_SIS_graph(SIS, path=None, pos=None, fig=None, ax=None):
    node_color = []
    for i in SIS.G.nodes:
        if SIS.G.nodes[i]["state"] == "S":
            node_color.append("green")
        elif SIS.G.nodes[i]["state"].startswith("I"):
            node_color.append("red")
        elif SIS.G.nodes[i]["state"].startswith("Q"):
            node_color.append("yellow")
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    res = nx.draw_networkx(SIS.G, node_color=node_color, pos=pos, ax=ax)
    
    # compute graph summaries
    num_sus = SIS.get_num_nodes_in_state("S")
    num_inf = SIS.get_num_nodes_in_state("I")
    num_int = SIS.get_num_nodes_in_state("Q")
    num_edge = SIS.G.number_of_edges()
    num_comb = len(list(itertools.combinations(SIS.G.nodes, 2)))
    fig.suptitle(f"time={SIS.t}\nnodes: #S={num_sus}, #inf={num_inf}, #int={num_int}"+
                 f"\nedges: #connection={num_edge}/{num_comb}")
    fig.tight_layout()
    if path: fig.savefig(path)
    plt.close()
    return res

def plot_affinity_distribution(SIS, path):
    affinities = sigmoid(np.array(SIS.get_all_affinity_scores()))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(affinities)
    fig.savefig(path)
