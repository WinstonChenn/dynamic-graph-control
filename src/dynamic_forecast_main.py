import argparse, random, os
from copy import deepcopy
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from models.StaticGNN import NodeGCN, NodeSAGE, EdgeGCN, EdgeSAGE
from generation.sis import DeterministicSIS, get_edge_index, get_X_matrix, \
    get_all_node_attribute
from utils import eval_utils, infer_utils
from models.DynamicGNN import SAGELSTM

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
    forecast_str = os.path.join(train_data_str, "Forecast", model_str, eval_data_node_str, 
                                f"#eval={args.num_eval_times}_#forecast={args.num_forecast_times}")
    forecast_cp_dir = os.path.join(args.cp_dir, forecast_str)
    os.makedirs(forecast_cp_dir, exist_ok=True)
    forecast_figure_dir = os.path.join(args.figure_dir, forecast_str)
    os.makedirs(forecast_figure_dir, exist_ok=True)

    ### Load forecast models ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_name == "SAGELSTM":
        model = SAGELSTM(num_nodes=args.num_nodes, num_X_features=5, hidden_dim=128).to(device)
    model_cp = torch.load(model_path, weights_only=False, map_location=device)
    model.load_state_dict(model_cp["state_dict"])
    model.eval()

    ### Evaluate forecasting performance ###
    SIS = DeterministicSIS(seed=args.seed, num_nodes=args.num_nodes, lat_dim=args.lat_dim, 
        edge_thresh=args.edge_thresh, int_param=args.eval_int_param, init_inf_prop=args.eval_init_inf_prop, 
        inf_thresh=args.eval_inf_thresh, max_inf_days=args.eval_max_inf_days, inf_param=args.eval_inf_param,
        sus_param=args.eval_sus_param, rec_param=args.eval_rec_param)
    all_aurocs = []
    for t in tqdm(range(args.num_eval_times)):
        # forecast
        curr_forecast_path = os.path.join(forecast_cp_dir, f"time={t}.npy")
        if not os.path.isfile(curr_forecast_path) or args.overwrite:
            curr_forecast, _ = infer_utils.dynamic_forecast(SIS.G, model=model, intervention=[],
                device=device, num_times=args.num_forecast_times, input_label_dict={
                    "x_label":"X", "a_label":"A_curr", "t_label": "T", "y_label":"Y_curr", "edge_index_label":"edge_index"})
            curr_aurocs = []
            # evaluate forecast
            curr_SIS = deepcopy(SIS)
            for i in range(args.num_forecast_times):
                curr_SIS.update()
                curr_states = get_all_node_attribute(curr_SIS.G, "state")
                curr_labels = np.array([not s.startswith("S") for s in curr_states]).astype(int)
                curr_aurocs.append(roc_auc_score(curr_labels, curr_forecast[i], labels=[0,1,2], multi_class="ovr", average="micro"))
            curr_aurocs = np.array(curr_aurocs)
            np.save(curr_forecast_path, curr_aurocs)
        else:
            curr_aurocs = np.load(curr_forecast_path)
        all_aurocs.append(curr_aurocs)
        SIS.update()
    all_aurocs = np.stack(all_aurocs)

    diff_str = f"InfThreshDiff={abs(args.train_inf_thresh-args.eval_inf_thresh):.3f}_" \
        f"MaxDaysDiff={abs(args.train_max_inf_days-args.eval_max_inf_days):.3f}"
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(all_aurocs.mean(axis=0), color="tab:blue")
    auroc_lower, auroc_higher = eval_utils.confidence_interval(all_aurocs)
    ax.fill_between(x=range(all_aurocs.shape[1]), y1=auroc_lower, y2=auroc_higher, color="tab:blue", alpha=0.15)
    ax.set_xlabel("#Forecast Time Steps")
    ax.set_ylabel("Node Label AUROC")
    ax.set_title(diff_str)
    fig.tight_layout()
    fig.savefig(os.path.join(forecast_figure_dir, f"forecast_eval.png"))

    

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
    parser.add_argument("--num_eval_times", type=int, default=100, help="Number of timesteps to evaluate forecast")
    parser.add_argument("--num_forecast_times", type=int, default=100, help="Number of timesteps to forecast the graph")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    main(args)