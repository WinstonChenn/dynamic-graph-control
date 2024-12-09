import argparse, random, os, itertools
from copy import deepcopy
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from models.DynamicGNN import SAGELSTM
from generation.sis import DeterministicSIS, get_edge_index, get_X_matrix, \
    get_nodes_in_state
from utils import eval_utils, infer_utils

def main(args):
    ### Set random seed ###
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ### Setup directories ###
    train_data_str = os.path.join(f"#nodes={args.num_nodes}", f"latDim={args.lat_dim}_edgeThresh={args.edge_thresh}", 
        f"initProp={args.train_init_inf_prop}_infThresh={args.train_inf_thresh}_maxDays={args.train_max_inf_days}" \
        f"_infParam={args.train_inf_param}_susParam={args.train_sus_param}" \
        f"_recParam={args.train_rec_param}_intParam={args.train_int_param}", 
        f"InterveneRate={args.train_intervene_rate}", f"#train={args.num_train}_#val={args.num_val}_#test={args.num_test}", 
        f"seed={args.seed}")
    model_str = os.path.join(f"model={args.model_name}", 
        f"#epochs={args.epochs}_batch={args.batch_size}_lr={args.lr}_l2={args.l2}" \
        f"_patience={args.patience}_delta={args.delta}")
    model_dir = os.path.join(args.cp_dir, train_data_str, "Model", model_str)
    model_path = os.path.join(model_dir, "model.pt")
    assert os.path.isfile(model_path)
    eval_data_node_str = f"initProp={args.eval_init_inf_prop}_infThresh={args.eval_inf_thresh}" \
        f"_maxDays={args.eval_max_inf_days}_infParam={args.eval_inf_param}_susParam={args.eval_sus_param}" \
        f"_recParam={args.eval_rec_param}_intParam={args.eval_int_param}"
    policy_str = os.path.join(train_data_str, "Policy", model_str, eval_data_node_str,
                              f'#time={args.num_times}_#int={args.num_int}')
    policy_cp_dir = os.path.join(args.cp_dir, policy_str)
    os.makedirs(policy_cp_dir, exist_ok=True)
    ploicy_figure_dir = os.path.join(args.figure_dir, policy_str)
    os.makedirs(ploicy_figure_dir, exist_ok=True)

    ### Load models ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_name == "SAGELSTM":
        model = SAGELSTM(num_nodes=args.num_nodes, num_X_features=5, hidden_dim=128).to(device)
    model_cp = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(model_cp["state_dict"])
    model.eval()

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
    num_reps = 100
    
    def no_intervention_policy(G, num_int): return []
    def random_policy(G, num_int):
        unint_idx = np.array(get_nodes_in_state(G, "Q", invert=True))
        int_idx = np.random.choice(unint_idx, size=num_int)
        return int_idx
    
    def joint_risk_policy(G, num_int, num_times=1):
        node_state_scores, _ = infer_utils.dynamic_forecast(G, model=model, intervention=[], device=device, num_times=num_times, 
            input_label_dict={"x_label":"X", "a_label":"A_curr", "t_label": "T", "y_label":"Y_curr", "edge_index_label":"edge_index"})
        node_risk = node_state_scores[:, :, 1].prod(axis=0)
        unint_idx = np.array(get_nodes_in_state(G, "Q", invert=True))
        int_idx = unint_idx[np.argsort(node_risk[unint_idx])[::-1][:num_int]]
        return int_idx

    def monte_carlo_policy(G, num_int, num_times=1):

        # int_idx = []
        # unint_idx = np.array(get_nodes_in_state(G, "Q", invert=True))
        # for i in range(num_int):
        #     np.random.shuffle(unint_idx)
        #     best_reward, best_node = -1, None
        #     for j in unint_idx:
        #         node_scores, _ = infer_utils.dynamic_forecast(G, model=model, intervention=int_idx+[j], device=device, num_times=num_times, 
        #             input_label_dict={"x_label":"X", "a_label":"A_curr", "t_label": "T", "y_label":"Y_curr", "edge_index_label":"edge_index"})
        #         reward = (np.argmax(node_scores, axis=-1)==0).sum()
        #         if reward > best_reward:
        #             best_reward, best_node = reward, j
        #     int_idx.append(best_node)
        #     unint_idx = unint_idx[unint_idx!=best_node]
        # assert len(int_idx) == num_int
        # return int_idx

        unint_idx = np.array(get_nodes_in_state(G, "Q", invert=True))
        rewards = []
        for i in unint_idx:
            node_scores, _ = infer_utils.dynamic_forecast(G, model=model, intervention=[i], device=device, num_times=num_times, 
                input_label_dict={"x_label":"X", "a_label":"A_curr", "t_label": "T", "y_label":"Y_curr", "edge_index_label":"edge_index"})
            reward = (np.argmax(node_scores, axis=-1)==0).sum()
            rewards.append(reward)
        
        return unint_idx[np.argsort(rewards)[::-1][:num_int]]
    
    def risk_attribution_policy(G, num_int, num_times=1):
        node_state_scores, adj_mats = infer_utils.dynamic_forecast(G, model=model, intervention=[], device=device, num_times=num_times, 
            input_label_dict={"x_label":"X", "a_label":"A_curr", "t_label": "T", "y_label":"Y_curr", "edge_index_label":"edge_index"})
        attribution = np.zeros((len(G.nodes), num_times))
        for i in np.arange(num_times-2, -1, -1):
            delta_plus = np.maximum(0, node_state_scores[i+1][:, 1] - node_state_scores[i][:, 1]) + attribution[:, i+1]
            delta_plus = delta_plus.reshape(-1, 1)
            curr_adj_scores = adj_mats[i]
            curr_risk_scores = node_state_scores[i][:, 1].reshape(-1, 1)
            curr_risk_mat = curr_risk_scores.dot(delta_plus.T)
            np.fill_diagonal(curr_risk_mat, 0)
            attribution[:, i] = (curr_adj_scores * curr_risk_mat).sum(axis=1)
        unint_idx = np.array(get_nodes_in_state(G, "Q", invert=True))
        int_idx = unint_idx[np.argsort(attribution[unint_idx, 0])[::-1][:num_int]]
        return int_idx

        

    policy_dict = {
        "No Intervention": no_intervention_policy, "Random": random_policy, 
        r"Joint Risk($\theta$=1)": lambda G, num_int: joint_risk_policy(G, num_int, num_times=1),
        r"Joint Risk($\theta$=5)": lambda G, num_int: joint_risk_policy(G, num_int, num_times=5),
        r"Joint Risk($\theta$=10)": lambda G, num_int: joint_risk_policy(G, num_int, num_times=10),
        # r"Monte Carlo($\theta$=5)": lambda G, num_int: monte_carlo_policy(G, num_int, num_times=5),
        # r"Risk Attribution($\theta$=5)": lambda G, num_int: risk_attribution_policy(G, num_int, num_times=5),
    }

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for idx, policy in enumerate(policy_dict):
        cp_dir = os.path.join(policy_cp_dir, policy.replace(" ", ""))
        os.makedirs(cp_dir, exist_ok=True)
        rewards = repeat_polciy_eval(policy_dict[policy], cp_dir, num_reps=num_reps, overwrite=policy.startswith("Risk Attribution"))
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
    
    
    # diff_str = f"InfThreshDiff={abs(args.train_inf_thresh-args.eval_inf_thresh):.3f}_" \
    #     f"MaxDaysDiff={abs(args.train_max_inf_days-args.eval_max_inf_days):.3f}"
    if args.train_max_inf_days-args.eval_max_inf_days == 0:
        diff_str = "Policy Performance under Historical Infectious Disease"
    elif abs(args.train_max_inf_days-args.eval_max_inf_days) >= 5:
        diff_str = "Polcy Performance under Novel Infectious Disease with Significant Change in Recovery Length"
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
    parser.add_argument("--train_intervene_rate", type=float, default=0.1, 
                        help="intervention assignment rate during training")
    # sample size args
    parser.add_argument("--num_train", type=int, default=100, help="number of trainig graphs")
    parser.add_argument("--num_val", type=int, default=100, help="number of testing graphs")
    parser.add_argument("--num_test", type=int, default=100, help="number of testing graphs")
    # node modeling args
    parser.add_argument("--model_name", type=str, choices=["SAGELSTM"], default="GCN")
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