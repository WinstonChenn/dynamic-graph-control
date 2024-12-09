import argparse, random, os, itertools
from copy import deepcopy
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from models.StaticGNN import NodeGCN, NodeSAGE, EdgeGCN, EdgeSAGE
from generation.sis import DeterministicSIS, get_edge_index, get_X_matrix, \
    get_nodes_in_state
from utils import eval_utils, infer_utils

def main(args):
    ### Set random seed ###
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ### Setup directories ###
    train_data_node_str = f"initProp={args.train_init_inf_prop}_infThresh={args.train_inf_thresh}" \
        f"_maxDays={args.train_max_inf_days}" \
        f"_infParam={args.train_inf_param}_susParam={args.train_sus_param}" \
        f"_recParam={args.train_rec_param}_intParam={args.train_int_param}"
    train_data_str = os.path.join(f"#nodes={args.num_nodes}", 
        f"latDim={args.lat_dim}_edgeThresh={args.edge_thresh}", train_data_node_str, 
        f"#train={args.num_train}_#val={args.num_val}_#test={args.num_test}", f"seed={args.seed}")
    model_str = os.path.join(f"model={args.model_name}", 
        f"#epochs={args.epochs}_batch={args.batch_size}_lr={args.lr}_l2={args.l2}" \
        f"_patience={args.patience}_delta={args.delta}")
    model_dir = os.path.join(args.cp_dir, train_data_str, "Model", model_str)
    node_model_path = os.path.join(model_dir, "node_model.pt")
    assert os.path.isfile(node_model_path)
    edge_model_path = os.path.join(model_dir, "edge_model.pt")
    assert os.path.isfile(edge_model_path)
    eval_data_node_str = f"initProp={args.eval_init_inf_prop}_infThresh={args.eval_inf_thresh}" \
        f"_maxDays={args.eval_max_inf_days}" \
        f"_infParam={args.eval_inf_param}_susParam={args.eval_sus_param}" \
        f"_recParam={args.eval_rec_param}_intParam={args.eval_int_param}"
    policy_str = os.path.join(train_data_str, "Policy", model_str, eval_data_node_str,
                              f'#time={args.num_times}_#int={args.num_int}')
    policy_cp_dir = os.path.join(args.cp_dir, policy_str)
    os.makedirs(policy_cp_dir, exist_ok=True)
    ploicy_figure_dir = os.path.join(args.figure_dir, policy_str)
    os.makedirs(ploicy_figure_dir, exist_ok=True)

    ### Load models ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_name == "GCN":
        node_model = NodeGCN(num_node_features=5).to(device)
        edge_model = EdgeGCN(num_node_features=args.num_nodes).to(device)
    elif args.model_name == "SAGE":
        node_model = NodeSAGE(num_node_features=5).to(device)
        edge_model = EdgeSAGE(num_node_features=args.num_nodes).to(device)
    node_cp = torch.load(node_model_path, weights_only=False, map_location=device)
    node_model.load_state_dict(node_cp["state_dict"]) 
    edge_cp = torch.load(edge_model_path, weights_only=False, map_location=device)
    edge_model.load_state_dict(edge_cp["state_dict"]) 

    ### Evaluate different heuristic policies ###
    def repeat_polciy_eval(policy, policy_cp_dir, num_reps=100, overwrite=False):
        rewards_list = []
        for i in tqdm(range(num_reps)):
            SIS = DeterministicSIS(seed=args.seed+i, num_nodes=args.num_nodes, lat_dim=args.lat_dim, 
                edge_thresh=args.edge_thresh, int_param=args.eval_int_param, init_inf_prop=args.eval_init_inf_prop, 
                inf_thresh=args.eval_inf_thresh, max_inf_days=args.eval_max_inf_days, inf_param=args.eval_inf_param,
                sus_param=args.eval_sus_param, rec_param=args.eval_rec_param)
            reward_path = os.path.join(policy_cp_dir, f"rep={i}.npy")
            if not os.path.isfile(reward_path) or overwrite:
                rewards = eval_utils.evaluate_policy(policy, environment=deepcopy(SIS), 
                    num_times=args.num_times, num_int=args.num_int, verbose=False)
                np.save(reward_path, rewards)
            else:
                rewards = np.load(reward_path)
            rewards_list.append(rewards)
        return np.stack(rewards_list)

    ## Evaluate random policy ###
    num_reps = 10
    
    def no_intervention_policy(G, num_int): return []
    def random_policy(G, num_int):
        unint_idx = np.array(get_nodes_in_state(G, "Q", invert=True))
        int_idx = np.random.choice(unint_idx, size=num_int)
        return int_idx
    def risk_policy(G, num_int):
        unint_idx = np.array(get_nodes_in_state(G, "Q", invert=True))
        G_node_data = infer_utils.nx2pyg(G, device, pred_edge=False)
        node_risk = node_model(G_node_data).detach().cpu().numpy().reshape(-1)
        int_idx = unint_idx[np.argsort(node_risk[unint_idx])[::-1][:num_int]]
        return int_idx

    def naive_forecast_policy(G, num_int, num_times=1, num_mc=None):
        node_scores, adj_mats = infer_utils.forecast(G, node_model, edge_model, device=device, 
            num_times=num_times+1)
        cum_node_scores = node_scores.prod(axis=0)
        unint_idx = np.array(get_nodes_in_state(G, "Q", invert=True))
        int_idx = unint_idx[np.argsort(cum_node_scores[unint_idx])[::-1][:num_int]]
        return int_idx
    
    def attribute_forecast_policy(G, num_int, num_times=1):
        node_scores, adj_mats = infer_utils.forecast(G, node_model, edge_model, device=device, 
            num_times=num_times+1)
        # for node in get_nodes_in_state(G, "Q"):

        attribution = np.zeros(len(G.nodes))
        for i in range(num_times):
            delta = node_scores

        breakpoint()
        

    policy_dict = {
        "No Intervention": no_intervention_policy, 
        "Random": random_policy, "Risk": risk_policy, 
        "Forecast(t=1)": lambda G, num_int: naive_forecast_policy(G, num_int, num_times=1)
    }

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    for idx, policy in enumerate(policy_dict):
        cp_dir = os.path.join(policy_cp_dir, policy.replace(" ", ""))
        os.makedirs(cp_dir, exist_ok=True)
        rewards = repeat_polciy_eval(policy_dict[policy], cp_dir, num_reps=num_reps)
        ax.plot(rewards.mean(axis=0), label=f"{policy} Policy", color=colors[idx])
        lower, higher = eval_utils.confidence_interval(rewards)
        ax.fill_between(x=range(rewards.shape[1]), y1=lower, y2=higher, color=colors[idx], alpha=0.15)

    
    
    # def monte_carlo_policy(G, num_int, num_times=1, num_sims=10):
    #     for node in G.nodes:
    #         G_curr = deepcopy(G)
    #         for sim in range(num_sims):
    #             # predict next node labels
    #             G_node_data = infer_utils.nx2pyg(G_curr, device, pred_edge=False)
    #             node_scores = node_model(G_node_data).detach().cpu().numpy().reshape(-1)
    #             node_pred = 


    # for t in range(1, 4):
    #     print(f"Evaluate Naive Forecasting Policy with {t} timestep")
    #     curr_forecast_policy_cp_dir = os.path.join(policy_cp_dir, f"Forecast(t={t})")
    #     os.makedirs(curr_forecast_policy_cp_dir, exist_ok=True)
    #     curr_forecast_rewards = repeat_polciy_eval(lambda G, num_int: forecast_policy(G, num_int, num_times=t), 
    #         curr_forecast_policy_cp_dir, num_reps=num_reps, overwrite=args.overwrite)
    #     ax.plot(curr_forecast_rewards.mean(axis=0), label=f"{args.model_name} naive forecast policy "+r"($\theta$="f"{t})", color=colors[t+1])
    #     node_lower, node_higher = eval_utils.confidence_interval(curr_forecast_rewards)
    #     ax.fill_between(x=range(curr_forecast_rewards.shape[1]), y1=node_lower, y2=node_higher, color=colors[t+1], alpha=0.15)
    
    
    diff_str = f"InfThreshDiff={abs(args.train_inf_thresh-args.eval_inf_thresh):.3f}_" \
        f"MaxDaysDiff={abs(args.train_max_inf_days-args.eval_max_inf_days):.3f}"
    ax.legend()
    ax.set_xlabel("timestep")
    ax.set_ylabel("Percent of healthy nodes")
    ax.set_ylim(0, 1)
    ax.set_xlim(-1, args.num_times)
    ax.set_title(diff_str)
    fig.tight_layout()
    fig.savefig(os.path.join(ploicy_figure_dir, f"policy_eval.png"))
    fig.savefig(os.path.join(ploicy_figure_dir, f"policy_eval.pdf"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp_dir", type=str, default="../checkpoints")
    parser.add_argument("--figure_dir", type=str, default="../figures")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    # edge generation args
    parser.add_argument("--num_nodes", type=int, default=30, help="number of nodes")
    parser.add_argument("--lat_dim", type=int, default=30, help="dynamic node latent feature dimension")
    parser.add_argument("--edge_thresh", type=float, default=0.55, help="edge generation threshold")
    # node generation args
    parser.add_argument("--train_init_inf_prop", type=float, default=0.1, help="initial infection proportion")
    parser.add_argument("--train_inf_thresh", type=float, default=0.3, help="infection pressure threshold")
    parser.add_argument("--train_max_inf_days", type=int, default=10, help="maximum possible number of infected days")
    parser.add_argument("--train_int_param", type=float, nargs=2, default=None, 
                        help="Beta distribution parameter for intervenablness")
    parser.add_argument("--train_inf_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for infectiousnes")
    parser.add_argument("--train_sus_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for susceptibilty")
    parser.add_argument("--train_rec_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for recoverability")
    # sample size args
    parser.add_argument("--num_train", type=int, default=100, help="number of trainig graphs")
    parser.add_argument("--num_val", type=int, default=100, help="number of testing graphs")
    parser.add_argument("--num_test", type=int, default=100, help="number of testing graphs")
    # node modeling args
    parser.add_argument("--model_name", type=str, choices=["GCN", "SAGE", "GIN", "GAT"], default="GCN")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--delta", type=float, default=1e-4)
    # forecast eval node generation args  
    parser.add_argument("--eval_int_param", type=float, nargs=2, default=None)
    parser.add_argument("--eval_init_inf_prop", type=float, default=0.1, help="initial infection proportion")
    parser.add_argument("--eval_inf_thresh", type=float, default=0.3, help="infection pressure threshold")
    parser.add_argument("--eval_max_inf_days", type=int, default=10, help="maximum possible number of infected days")
    parser.add_argument("--eval_inf_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for infectiousnes")
    parser.add_argument("--eval_sus_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for susceptibilty")
    parser.add_argument("--eval_rec_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for recoverability")  
    parser.add_argument("--num_times", type=int, default=100, help="Number of timesteps to evaluate policy")
    parser.add_argument("--num_int", type=int, default=1, help="Number of interevntions applied at each timestep")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    main(args)