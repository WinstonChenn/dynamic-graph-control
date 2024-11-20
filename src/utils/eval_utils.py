from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import viz_utils
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from generation.sis import get_nodes_in_state

def evaluate_policy(policy, environment, num_int=3, num_times=100, verbose=True):
    model_rewards = []
    for i in tqdm(range(num_times), disable=not verbose):
        int_idx = policy(environment.G, num_int)
        environment.update()
        environment.intervene(int_idx)
        reward = len(get_nodes_in_state(environment.G, "S")) / len(environment.G.nodes)
        model_rewards.append(reward)
    return np.array(model_rewards)

def plot_learning_curves(loss_dict, eval_dict, log=False, axes=None, fig=None):
    if axes is None or fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(loss_dict["train"], label="train", alpha=0.75)
    axes[0].plot(loss_dict["val"], label="val", alpha=0.75)
    axes[0].plot(loss_dict["test"], label="test", alpha=0.75)
    axes[1].plot(eval_dict["train"], label="train", alpha=0.75)
    axes[1].plot(eval_dict["val"], label="val", alpha=0.75)
    axes[1].plot(eval_dict["test"], label="test", alpha=0.75)
    axes[0].set_title("Loss")
    axes[1].set_title("AUROC")
    axes[0].legend()
    axes[1].legend()
    axes[0].set_xlabel("Number of Training Epochs")
    axes[1].set_xlabel("Number of Training Epochs")
    if log:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")
    fig.tight_layout()
    return fig

def eval_model(model, dataloader, y_label="y", x_label="x", edge_index_label="edge_index", 
               eval_edge=False, decode_index_label=None, metric="auroc", threshold=0.5):
    if eval_edge: assert decode_index_label is not None
    loss_list, eval_list = [], []
    for data in dataloader:
        if eval_edge:
            score = model(data, x_label=x_label, edge_index_label=edge_index_label, 
                         decode_index=data[decode_index_label])
        else:
            score = model(data, x_label)
        target = data[y_label]
        loss = F.binary_cross_entropy(score, target)
        loss_list.append(loss.item())
        score_np = score.detach().cpu().numpy()
        pred_np = (score_np>threshold).astype(int)
        target_np = target.detach().cpu().numpy()
        if metric == "auroc":
            eval_list.append(roc_auc_score(target_np, score_np))
        elif metric == "accuracy":
            eval_list.append(accuracy_score(target_np, pred_np))
        elif metric == "f1":
            eval_list.append(f1_score(target_np, pred_np))
        elif metric == "precision":
            eval_list.append(precision_score(target_np, pred_np))
        elif metric == "recall":
            eval_list.append(recall_score(target_np, pred_np))
    return loss_list, eval_list

def plot_temporal_eval(model, data_list, train_time=None, val_time=None, 
        y_label="y", x_label="x", edge_index_label="edge_index", 
        eval_edge=False, decode_index_label=None, y_axis_label="", metric="auroc"):
    if eval_edge: assert decode_index_label is not None
    _, evals = eval_model(model, data_list, 
        y_label=y_label, x_label=x_label, edge_index_label=edge_index_label, 
        eval_edge=eval_edge, decode_index_label=decode_index_label, metric=metric)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(evals, label=metric)
    ax.plot(viz_utils.smooth(evals, window_size=10), label=f"Smoothed {metric}")
    if train_time is not None:
        ax.axvline(x=train_time, color='y', linestyle='--', label="train cutoff")
    if val_time is not None:
        ax.axvline(x=val_time, color='r', linestyle='--', label="val cutoff")
    ax.legend()
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel("Timestep")
    fig.tight_layout()
    return fig

def confidence_interval(rewards, level=0.95):
    means = np.mean(rewards, axis=0)
    ci = level*np.std(rewards, axis=0)/np.sqrt(len(rewards))
    lower = means - ci
    higher = means + ci
    return lower, higher