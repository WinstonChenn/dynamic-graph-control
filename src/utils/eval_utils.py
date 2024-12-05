from tqdm import tqdm
import torch
import torch.nn as nn
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

def plot_dynamic_learning_curves(loss_dict, eval_dict, log=False, axes=None, fig=None):
    if axes is None or fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0][0].plot(loss_dict["train"]["Y"], label="train", alpha=0.75)
    axes[0][0].plot(loss_dict["val"]["Y"], label="val", alpha=0.75)
    axes[0][0].plot(loss_dict["test"]["Y"], label="test", alpha=0.75)

    axes[0][1].plot(eval_dict["train"]["Y"], label="train", alpha=0.75)
    axes[0][1].plot(eval_dict["val"]["Y"], label="val", alpha=0.75)
    axes[0][1].plot(eval_dict["test"]["Y"], label="test", alpha=0.75)

    axes[1][0].plot(loss_dict["train"]["A"], label="train", alpha=0.75)
    axes[1][0].plot(loss_dict["val"]["A"], label="val", alpha=0.75)
    axes[1][0].plot(loss_dict["test"]["A"], label="test", alpha=0.75)

    axes[1][1].plot(eval_dict["train"]["A"], label="train", alpha=0.75)
    axes[1][1].plot(eval_dict["val"]["A"], label="val", alpha=0.75)
    axes[1][1].plot(eval_dict["test"]["A"], label="test", alpha=0.75)
    
    axes[0][0].set_title("State Prediction Loss")
    axes[0][1].set_title("State Prediction AUROC")
    axes[1][0].set_title("Adjacency Prediction Loss")
    axes[1][1].set_title("Adjacency Prediction AUROC")
    axes[0][0].legend()
    axes[0][1].legend()
    axes[1][0].legend()
    axes[1][1].legend()
    axes[0][0].set_xlabel("Number of Training Epochs")
    axes[0][1].set_xlabel("Number of Training Epochs")
    axes[1][0].set_xlabel("Number of Training Epochs")
    axes[1][1].set_xlabel("Number of Training Epochs")
    if log:
        axes[0][0].set_yscale("log")
        axes[0][1].set_yscale("log")
        axes[1][0].set_yscale("log")
        axes[1][1].set_yscale("log")
    fig.tight_layout()
    return fig

def eval_static_model(model, dataloader, y_label="y", x_label="x", edge_index_label="edge_index", 
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

def eval_dynamic_model(model, data_list, input_label_dict={"x_label":"X", "y_label":"Y_curr", "edge_index_label":"edge_index"},
                       metric="accuracy"):
    y_loss_list, y_eval_list = [], []
    a_loss_list, a_eval_list = [], []
    y_hidden, a_hidden = None, None
    for data in data_list:
        y_score, a_score, y_hidden, a_hidden = model(data, y_hidden=y_hidden, a_hidden=a_hidden, **input_label_dict)
        if y_hidden is not None:
            y_hidden = tuple(y.detach() for y in y_hidden)
        if a_hidden is not None:
            a_hidden = tuple(a.detach() for a in a_hidden)
        y_loss = nn.CrossEntropyLoss()(y_score, data["Y_next"])
        a_loss = nn.BCEWithLogitsLoss()(a_score, data["A_next"])
        y_loss_list.append(y_loss.item())
        a_loss_list.append(a_loss.item())
        y_score_np = y_score.detach().cpu().numpy()
        y_pred_np = np.argmax(y_score_np, axis=1)
        # mask = ~torch.eye(a_score.size(0), dtype=torch.bool)
        mask = torch.triu(torch.ones_like(a_score, dtype=torch.bool), diagonal=1)
        a_score, a_target = a_score[mask], data["A_next"][mask]
        a_score_np = a_score.detach().cpu().numpy()
        a_pred_np = (a_score_np>0.5).astype(int)
        
        y_target_np = data["Y_next"].detach().cpu().numpy()
        a_target_np = a_target.detach().cpu().numpy()
        if metric == "auroc":
            y_eval_list.append(roc_auc_score(y_target_np, y_score_np, labels=[0,1,2], multi_class="ovr", average="micro"))
            a_eval_list.append(roc_auc_score(a_target_np, a_score_np, average=None))
        elif metric == "accuracy":
            y_eval_list.append(accuracy_score(y_target_np, y_pred_np))
            a_eval_list.append(accuracy_score(a_target_np, a_pred_np))
        elif metric == "f1":
            y_eval_list.append(f1_score(y_target_np, y_pred_np, labels=[0,1,2], average="micro"))
            a_eval_list.append(f1_score(a_target_np, a_pred_np))
        elif metric == "precision":
            y_eval_list.append(precision_score(y_target_np, y_pred_np, labels=[0,1,2], average="micro"))
            a_eval_list.append(precision_score(a_target_np, a_pred_np))
        elif metric == "recall":
            y_eval_list.append(recall_score(y_target_np, y_pred_np, labels=[0,1,2], average="micro"))
            a_eval_list.append(recall_score(a_target_np, a_pred_np))
        else: raise Exception(f"{metric} metric undefined")
    return y_loss_list, a_loss_list, y_eval_list, a_eval_list

def plot_temporal_eval(model, data_list, train_time=None, val_time=None, 
        y_label="y", x_label="x", edge_index_label="edge_index",
        eval_edge=False, decode_index_label=None, y_axis_label="", metric="auroc", 
        criterion=torch.nn.BCELoss()):
    if eval_edge: assert decode_index_label is not None
    _, evals = eval_static_model(model, data_list, 
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

def plot_dynamic_temporal_eval(model, data_list, train_time=None, val_time=None, 
        input_label_dict={"x_label":"X", "y_label":"Y_curr", "edge_index_label":"edge_index"}, 
        y_axis_label="", metric="auroc"):
    _, _, y_evals, a_evals = eval_dynamic_model(model, data_list, input_label_dict=input_label_dict, metric=metric)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(y_evals, label=f"{metric}")
    axes[0].plot(viz_utils.smooth(y_evals, window_size=10), label=f"{metric}")
    axes[0].set_title("Node State Preidction")
    axes[1].plot(a_evals, label=f"{metric}")
    axes[1].plot(viz_utils.smooth(a_evals, window_size=10), label=f"Smoothed {metric}")
    axes[1].set_title("Adjacency Preidction")
    if train_time is not None:
        axes[0].axvline(x=train_time, color='y', linestyle='--', label="train cutoff")
        axes[1].axvline(x=train_time, color='y', linestyle='--', label="train cutoff")
    if val_time is not None:
        axes[0].axvline(x=val_time, color='r', linestyle='--', label="val cutoff")
        axes[1].axvline(x=val_time, color='r', linestyle='--', label="val cutoff")
    axes[0].legend()
    axes[1].legend()
    axes[0].set_ylabel(y_axis_label)
    axes[1].set_ylabel(y_axis_label)
    axes[0].set_xlabel("Timestep")
    axes[1].set_xlabel("Timestep")
    fig.tight_layout()
    return fig
    

def confidence_interval(rewards, level=0.95):
    means = np.mean(rewards, axis=0)
    ci = level*np.std(rewards, axis=0)/np.sqrt(len(rewards))
    lower = means - ci
    higher = means + ci
    return lower, higher