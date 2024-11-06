import argparse, os, pickle, random
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from utils import train_utils, viz_utils, eval_utils
from generation.sis import DeterministicSIS, get_node_feature_matrix, \
    get_node_label_vector, get_edge_index, get_edge_label, get_unintervened_node_index
from models.StaticGNN import NodeGCN, EdgeGCN

def nx2pyg(G, device="cpu"):
    edge_index_torch = torch.from_numpy(np.array(get_edge_index(G)).astype(int).T).to(device)
    x_node_torch = torch.from_numpy(get_node_feature_matrix(G).astype(np.float32)).to(device)
    return Data(x=x_node_torch, edge_index=edge_index_torch)


def main(args):
    ### Set random seed ###
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ### Setup directories ###
    data_str = os.path.join(f"#nodes={args.num_nodes}", 
        f"latDim={args.lat_dim}_edgeThresh={args.edge_thresh}_intParam={args.inf_param}", 
        f"initProp={args.init_inf_prop}_infThresh={args.max_inf_days}_infParam={args.inf_param}" \
        f"susParam={args.sus_param}_recParam={args.rec_param}", 
        f"#train={args.num_train}_#val={args.num_val}_#test={args.num_test}", f"seed={args.seed}")
    data_dir = os.path.join(args.data_dir, data_str)
    os.makedirs(data_dir, exist_ok=True)
    model_str = os.path.join(f"model={args.model_name}", 
        f"#epochs={args.epochs}_batch={args.batch_size}_lr={args.lr}_l2={args.l2}")
    cp_dir = os.path.join(args.cp_dir, data_str, model_str)
    os.makedirs(cp_dir, exist_ok=True)
    policy_cp_dir = os.path.join(cp_dir, "policy", f"#int={args.num_int}")
    os.makedirs(policy_cp_dir, exist_ok=True)
    random_policy_cp_dir = os.path.join(policy_cp_dir, "random")
    os.makedirs(random_policy_cp_dir, exist_ok=True)
    model_policy_cp_dir = os.path.join(policy_cp_dir, "model")
    os.makedirs(model_policy_cp_dir, exist_ok=True)
    figure_dir = os.path.join(args.figure_dir, data_str)
    os.makedirs(figure_dir, exist_ok=True)
    data_figure_dir = os.path.join(figure_dir, "data")
    os.makedirs(data_figure_dir, exist_ok=True)
    model_figure_dir = os.path.join(figure_dir, "model", model_str)
    os.makedirs(model_figure_dir, exist_ok=True)
    
    
    ### Generate data ###
    data_path = os.path.join(data_dir, "graphs.pkl")
    if not os.path.isfile(data_path) or args.overwrite_data:
        # generate graphs
        SIS = DeterministicSIS(seed=args.seed, num_nodes=args.num_nodes, lat_dim=args.lat_dim, 
            edge_thresh=args.edge_thresh, int_param=args.int_param, init_inf_prop=args.init_inf_prop, 
            inf_thresh=args.inf_thresh, max_inf_days=args.max_inf_days, inf_param=args.inf_param,
            sus_param=args.sus_param, rec_param=args.rec_param)
        pos = nx.random_layout(SIS.G)
        graphs = []
        for i in tqdm(range(args.num_train+args.num_val+args.num_test+1)):
            assert len(np.unique(get_node_label_vector(SIS.G))) > 1
            graphs.append(SIS.G.copy())
            viz_utils.plot_SIS_graph(SIS, path=os.path.join(data_figure_dir, f"t={i}"), pos=pos)
            SIS.update()
        with open(data_path, 'wb') as f: pickle.dump(graphs, f)
    else:
        with open(data_path, 'rb') as f: graphs = pickle.load(f)

    # convert networkx to pyG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_list = []
    for i in tqdm(range(len(graphs)-1)):
        G_curr, G_next = graphs[i], graphs[i+1]
        edge_index_torch = torch.from_numpy(np.array(get_edge_index(G_curr)).astype(int).T).to(device)
        x_node_torch = torch.from_numpy(get_node_feature_matrix(G_curr).astype(np.float32)).to(device)
        y_node_torch = torch.from_numpy(np.array(get_node_label_vector(G_next)).reshape(-1, 1).astype(np.float32)).to(device)
        y_edge_torch = torch.from_numpy(np.array(get_edge_label(G_next)).reshape(-1, 1).astype(np.float32)).to(device)
        data_list.append(Data(x=x_node_torch, edge_index=edge_index_torch, y_node=y_node_torch, y_edge=y_edge_torch))

    # np.random.shuffle(data_list)
    train_data, val_data = data_list[:args.num_train], data_list[args.num_train:args.num_train+args.num_val]
    test_data = data_list[args.num_train+args.num_val:]
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    ### Training GNN ###
    if args.model_name == "GCN":
        node_model = NodeGCN(num_node_features=data_list[0].x.shape[1]).to(device)
        # edge_model = EdgeGCN(num_node_features=data_list[0].x.shape[1]).to(device)
    
    # train node prediction GNN
    node_model_path = os.path.join(cp_dir, "model.pt")
    if not os.path.isfile(node_model_path) or args.overwrite_model:
        node_model, node_loss_dict, node_eval_dict = train_utils.train_torch_model(node_model, lr=args.lr, 
            l2=args.l2, epochs=args.epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
            test_dataloader=test_dataloader, verbose=True, label="y_node")
        torch.save({"state_dict": node_model.state_dict(), "loss_dict": node_loss_dict, 
                    "eval_dict": node_eval_dict}, node_model_path)
    else:
        cp = torch.load(node_model_path, weights_only=False)
        node_model.load_state_dict(cp["state_dict"]) 
        node_loss_dict, node_eval_dict = cp["loss_dict"], cp["eval_dict"]
    # train edge prediction GNN
    # edge_model = train_utils.train_torch_model(edge_model, lr=args.lr, l2=args.l2, epochs=args.epochs, 
    #     train_dataloader=train_dataloader, val_dataloader=val_dataloader, verbose=True, label="y_edge")
    # breakpoint()

    # plot loss and eval
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(node_loss_dict["train"], label="train")
    axes[0].plot(node_loss_dict["val"], label="val")
    axes[0].plot(node_loss_dict["test"], label="test")
    axes[1].plot(node_eval_dict["train"], label="train")
    axes[1].plot(node_eval_dict["val"], label="val")
    axes[1].plot(node_eval_dict["test"], label="test")
    axes[0].set_title("Loss")
    axes[1].set_title("AUROC")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(f"seed={args.seed}")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "node_loss_eval_curve.png"))
    
    # plot auroc by time
    aurocs = []
    for data in data_list:
        node_pred = node_model(data).detach().cpu().numpy()
        node_label = data.y_node.detach().cpu().numpy()
        curr_auroc = roc_auc_score(node_label, node_pred)
        aurocs.append(curr_auroc)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(aurocs, label="AUROC")
    ax.plot(viz_utils.smooth(aurocs, window_size=10), label="Smoothed AUROC")
    ax.axvline(x=args.num_train, color='y', linestyle='--', label="train cutoff")
    ax.axvline(x=args.num_train+args.num_val, color='r', linestyle='--', label="val cutoff")
    ax.legend()
    ax.set_ylabel("Node state prediction AUROC")
    ax.set_xlabel("Timestep")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "node_auroc_by_time.png"))
    
    ## Evaluate policy ###
    def node_model_policy(G, num_int):
        G_data = nx2pyg(G, device)
        node_pred = node_model(G_data).detach().cpu().numpy().reshape(-1)
        unint_idx = np.array(get_unintervened_node_index(G))
        int_idx = unint_idx[np.argsort(node_pred[unint_idx])[::-1][:num_int]]
        return int_idx
    
    def random_policy(G, num_int):
        unint_idx = np.array(get_unintervened_node_index(G))
        int_idx = np.random.choice(unint_idx, size=num_int)
        return int_idx

    
    SIS = DeterministicSIS(seed=args.seed, num_nodes=args.num_nodes, lat_dim=args.lat_dim, 
        edge_thresh=args.edge_thresh, int_param=args.int_param, init_inf_prop=args.init_inf_prop, 
        inf_thresh=args.inf_thresh, max_inf_days=args.max_inf_days, inf_param=args.inf_param,
        sus_param=args.sus_param, rec_param=args.rec_param)

    model_rewards_list, random_rewards_list = [], []
    for i in tqdm(range(101)):
        model_reward_path = os.path.join(model_policy_cp_dir, f"rep={i}.npy")
        if not os.path.isfile(model_reward_path):
            model_rewards = eval_utils.evaluate_policy(node_model_policy, environment=deepcopy(SIS), 
                num_int=args.num_int, verbose=False)
            np.save(model_reward_path, model_rewards)
        else:
            model_rewards = np.load(model_reward_path)
        model_rewards_list.append(model_rewards)
        random_reward_path = os.path.join(random_policy_cp_dir, f"rep={i}.npy")
        if not os.path.isfile(random_reward_path):
            random_rewards = eval_utils.evaluate_policy(random_policy, environment=deepcopy(SIS), 
                num_int=args.num_int, verbose=False)
            np.save(random_reward_path, random_rewards)
        else:
            random_rewards = np.load(random_reward_path)
        random_rewards_list.append(random_rewards)
    model_rewards, random_rewards = np.stack(model_rewards_list), np.stack(random_rewards_list)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(random_rewards.mean(axis=0), label="random policy", color="tab:blue")
    ax.fill_between(x=range(random_rewards.shape[1]), y1=np.quantile(random_rewards, 0.25, axis=0), 
        y2=np.quantile(random_rewards, 0.75, axis=0), label="random policy IQR", color="tab:blue", alpha=0.15)
    ax.plot(model_rewards.mean(axis=0), label=f"{args.model_name} policy", color="tab:orange")
    ax.fill_between(x=range(model_rewards.shape[1]), y1=np.quantile(model_rewards, 0.25, axis=0), 
        y2=np.quantile(model_rewards, 0.75, axis=0), label=f"{args.model_name} policy IQR", color="tab:orange", alpha=0.15)
    ax.legend()
    ax.set_xlabel("timestep")
    ax.set_ylabel("Percent of healthy nodes")
    ax.set_ylim(0, 1)
    ax.set_xlim(-1, model_rewards.shape[1])
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "policy_eval.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--cp_dir", type=str, default="../checkpoints")
    parser.add_argument("--figure_dir", type=str, default="../figures")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    # basic data args
    parser.add_argument("--num_nodes", type=int, default=30, help="number of nodes")
    # edge generation args
    parser.add_argument("--lat_dim", type=int, default=30, help="dynamic node latent feature dimension")
    parser.add_argument("--edge_thresh", type=float, default=0.55, help="edge generation threshold")
    parser.add_argument("--int_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for intervenablness")
    # node generation args
    parser.add_argument("--init_inf_prop", type=float, default=0.1, help="initial infection proportion")
    parser.add_argument("--inf_thresh", type=float, default=0.3, help="infection pressure threshold")
    parser.add_argument("--max_inf_days", type=int, default=10, help="maximum possible number of infected days")
    parser.add_argument("--inf_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for infectiousnes")
    parser.add_argument("--sus_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for susceptibilty")
    parser.add_argument("--rec_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for recoverability")
    parser.add_argument("--overwrite_data", action="store_true")
    # sample size args
    parser.add_argument("--num_train", type=int, default=100, help="number of trainig graphs")
    parser.add_argument("--num_val", type=int, default=100, help="number of testing graphs")
    parser.add_argument("--num_test", type=int, default=100, help="number of testing graphs")
    # modeling args
    parser.add_argument("--model_name", type=str, choices=["GCN"], default="GCN")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=5e-4)
    parser.add_argument("--overwrite_model", action="store_true")
    # intervention policy args
    parser.add_argument("--num_int", type=str, default=1, help="number of intervention every timestep")
    args = parser.parse_args()

    main(args)