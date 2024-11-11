from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from utils import viz_utils, train_utils
from torch_geometric.loader import DataLoader

def evaluate_policy(policy, environment, num_int=3, num_times=100, verbose=True):
    model_rewards = []
    for i in tqdm(range(num_times), disable=not verbose):
        int_idx = policy(environment.G, num_int)
        environment.update()
        environment.intervene(int_idx)
        reward = environment.get_num_nodes_in_state("S") / len(environment.G.nodes)
        model_rewards.append(reward)
    return np.array(model_rewards)

def plot_learning_curves(loss_dict, eval_dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(loss_dict["train"], label="train")
    axes[0].plot(loss_dict["val"], label="val")
    axes[0].plot(loss_dict["test"], label="test")
    axes[1].plot(eval_dict["train"], label="train")
    axes[1].plot(eval_dict["val"], label="val")
    axes[1].plot(eval_dict["test"], label="test")
    axes[0].set_title("Loss")
    axes[1].set_title("AUROC")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    return fig

def plot_temporal_auroc(model, data_list, train_time, val_time, 
        y_label, x_label="x", eval_edge=False):
    _, aurocs = train_utils.eval_model(model, DataLoader(data_list), 
        y_label=y_label, x_label=x_label, eval_edge=eval_edge)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(aurocs, label="AUROC")
    ax.plot(viz_utils.smooth(aurocs, window_size=10), label="Smoothed AUROC")
    ax.axvline(x=train_time, color='y', linestyle='--', label="train cutoff")
    ax.axvline(x=val_time, color='r', linestyle='--', label="val cutoff")
    ax.legend()
    ax.set_ylabel("Node state prediction AUROC")
    ax.set_xlabel("Timestep")
    fig.tight_layout()
    return fig