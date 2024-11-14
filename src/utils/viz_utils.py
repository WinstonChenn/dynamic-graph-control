import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from generation.sis import get_num_nodes_in_state, get_all_affinity_scores

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

def plot_SIS_graph(G, path=None, pos=None, fig=None, ax=None):
    node_color = []
    for i in G.nodes:
        if G.nodes[i]["state"] == "S":
            node_color.append("green")
        elif G.nodes[i]["state"].startswith("I"):
            node_color.append("red")
        elif G.nodes[i]["state"].startswith("Q"):
            node_color.append("yellow")
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    res = nx.draw_networkx(G, node_color=node_color, pos=pos, ax=ax)
    
    # compute graph summaries
    num_sus = get_num_nodes_in_state(G, "S")
    num_inf = get_num_nodes_in_state(G, "I")
    num_int = get_num_nodes_in_state(G, "Q")
    num_edge = G.number_of_edges()
    num_comb = len(list(itertools.combinations(G.nodes, 2)))
    fig.suptitle(f"nodes: #S={num_sus}, #inf={num_inf}, #int={num_int}"+
                 f"\nedges: #connection={num_edge}/{num_comb}")
    fig.tight_layout()
    if path: fig.savefig(path)
    plt.close()
    return res

def plot_affinity_distribution(G, path=None, fig=None, ax=None):
    affinities = sigmoid(np.array(get_all_affinity_scores(G)))
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.hist(affinities)
    if path is not None:
        fig.savefig(path)
    return ax

def animate_graphs(graphs):
    fig, ax = plt.subplots(1, 1)
    pos = nx.random_layout(graphs[0])
    def update_network(i):
        res = plot_SIS_graph(graphs[i], fig=fig, ax=ax, pos=pos)
        return res

    def animate(i):
        ax.clear()
        return update_network(i)

    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, animate, frames=len(graphs), 
                                  interval=500, repeat=False)
    return ani

def animate_affinity_distribution(graphs):
    fig, ax = plt.subplots(1, 1)
    def update_network(i):
        res = plot_affinity_distribution(graphs[i], fig=fig, ax=ax)
        return res

    def animate(i):
        ax.clear()
        return update_network(i)

    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, animate, frames=len(graphs), 
                                  interval=500, repeat=False)
    return ani