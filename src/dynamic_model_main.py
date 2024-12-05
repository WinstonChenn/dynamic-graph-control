import argparse, os, pickle, random
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from utils import train_utils, viz_utils, eval_utils
from generation.sis import DeterministicSIS, get_Y_vector, \
    get_edge_index, get_all_decode_index_label, get_X_matrix, get_A_matrix, get_Y_vector
from models.StaticGNN import NodeGCN, NodeSAGE, EdgeGCN, EdgeSAGE
from models.DynamicGNN import NodeSAGERNN, EdgeSAGERNN, SAGELSTM

def main(args):
    ### Set random seed ###
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ### Setup directories ###
    data_str = os.path.join(f"#nodes={args.num_nodes}", 
        f"latDim={args.lat_dim}_edgeThresh={args.edge_thresh}", 
        f"initProp={args.init_inf_prop}_infThresh={args.inf_thresh}" \
        f"_maxDays={args.max_inf_days}" \
        f"_infParam={args.inf_param}_susParam={args.sus_param}" \
        f"_recParam={args.rec_param}_intParam={args.int_param}", f"InterveneRate={args.intervene_rate}",
        f"#train={args.num_train}_#val={args.num_val}_#test={args.num_test}", 
        f"seed={args.seed}")
    data_dir = os.path.join(args.data_dir, data_str)
    os.makedirs(data_dir, exist_ok=True)
    model_str = os.path.join(f"model={args.model_name}", 
        f"#epochs={args.epochs}_batch={args.batch_size}_lr={args.lr}_l2={args.l2}" \
        f"_patience={args.patience}_delta={args.delta}")
    cp_dir = os.path.join(args.cp_dir, data_str, "Model", model_str)
    os.makedirs(cp_dir, exist_ok=True)
    figure_dir = os.path.join(args.figure_dir, data_str)
    os.makedirs(figure_dir, exist_ok=True)
    data_figure_dir = os.path.join(figure_dir, "Data")
    os.makedirs(data_figure_dir, exist_ok=True)
    model_figure_dir = os.path.join(figure_dir, "Model", model_str)
    os.makedirs(model_figure_dir, exist_ok=True)

    ### Generate data ###
    graph_data_path = os.path.join(data_dir, "graphs.pkl")
    intervention_data_path = os.path.join(data_dir, "interventions.npy")
    if not os.path.isfile(graph_data_path) or not os.path.isfile(intervention_data_path)\
        or args.overwrite_data:
            # generate graphs
            SIS = DeterministicSIS(seed=args.seed, num_nodes=args.num_nodes, lat_dim=args.lat_dim, 
                edge_thresh=args.edge_thresh, int_param=args.int_param, init_inf_prop=args.init_inf_prop, 
                inf_thresh=args.inf_thresh, max_inf_days=args.max_inf_days, inf_param=args.inf_param,
                sus_param=args.sus_param, rec_param=args.rec_param)
            pos = nx.random_layout(SIS.G)
            graphs, interventions = [], []
            for i in tqdm(range(args.num_train+args.num_val+args.num_test+1)):
                assert len(np.unique(get_Y_vector(SIS.G))) > 1
                graphs.append(SIS.G.copy())
                viz_utils.plot_SIS_graph(SIS.G, path=os.path.join(data_figure_dir, f"t={i}"), pos=pos)
                SIS.update()
                intervention_nodes = np.random.choice(SIS.G.nodes, size=round(args.num_nodes*args.intervene_rate))
                SIS.intervene(intervention_nodes)
                interventions.append(intervention_nodes)
            interventions = np.stack(interventions)
            with open(graph_data_path, 'wb') as f: pickle.dump(graphs, f)
            np.save(intervention_data_path, interventions)
    else:
        with open(graph_data_path, 'rb') as f: graphs = pickle.load(f)
        interventions = np.load(intervention_data_path)

    ### Convert Data ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_list = []
    for i in tqdm(range(len(graphs)-1)):
        G_curr, G_next = graphs[i], graphs[i+1]
        int_curr = interventions[i]
        
        # convert input data
        X_curr = torch.from_numpy(get_X_matrix(G_curr, time=True).astype(np.float32)).to(device)
        A_curr = torch.from_numpy(get_A_matrix(G_curr).astype(np.float32)).to(device)
        edge_index = torch.from_numpy(np.array(get_edge_index(G_curr)).astype(int).T).to(device)
        Y_curr = torch.from_numpy(get_Y_vector(G_curr).reshape(-1, 1).astype(int)).to(device)
        T_curr = torch.zeros((args.num_nodes, 1)).to(device)
        T_curr[int_curr, 0] = 1

        # convert output data
        Y_next = torch.from_numpy(get_Y_vector(G_next).astype(int)).to(device)
        A_next = torch.from_numpy(get_A_matrix(G_next).astype(np.float32)).to(device)

        data_list.append(Data(X=X_curr, edge_index=edge_index, A_curr=A_curr, A_next=A_next, 
            Y_curr=Y_curr, Y_next=Y_next, T=T_curr))

    # node data train/val/test split
    train_data = data_list[:args.num_train]
    val_data = data_list[args.num_train:args.num_train+args.num_val]
    test_data = data_list[args.num_train+args.num_val:]
    
    ### Training GNN ###
    if args.model_name == "SAGELSTM":
        model = SAGELSTM(num_nodes=args.num_nodes, num_X_features=train_data[0]["X"].shape[1], hidden_dim=128).to(device)
    model_path = os.path.join(cp_dir, "model.pt")
    if not os.path.isfile(model_path) or args.overwrite_model:
        model, loss_dict, eval_dict = train_utils.train_dynamic_torch_model(model, lr=args.lr, l2=args.l2, 
            epochs=args.epochs, patience=args.patience, delta=args.delta, train_data_list=train_data, 
            val_data_list=val_data, test_data_list=test_data, verbose=True, edge_pred=False, 
            input_label_dict={"x_label":"X", "a_label":"A_curr", "t_label": "T", "y_label":"Y_curr", "edge_index_label":"edge_index"})
        torch.save({"state_dict": model.state_dict(), "loss_dict": loss_dict, "eval_dict": eval_dict}, model_path)
    else:
        node_cp = torch.load(model_path, weights_only=False, map_location=device)
        model.load_state_dict(node_cp["state_dict"]) 
        loss_dict, eval_dict = node_cp["loss_dict"], node_cp["eval_dict"]
    model.flatten_parameters()
    model.eval()

    ### Evaluate Model ###
    # plot learning curve
    fig = eval_utils.plot_dynamic_learning_curves(loss_dict, eval_dict, log=False)
    fig.suptitle(f"{args.model_name} Loss & AUROC")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "loss_eval_curve.png"))
    
    fig = eval_utils.plot_dynamic_temporal_eval(model=model, data_list=data_list[args.num_train:], val_time=args.num_val, 
        input_label_dict={"x_label":"X", "a_label":"A_curr", "t_label": "T", "y_label":"Y_curr", "edge_index_label":"edge_index"}, metric="auroc")
    fig.suptitle(f"{args.model_name} Performance AUROC")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "auroc_by_time.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--cp_dir", type=str, default="../checkpoints")
    parser.add_argument("--figure_dir", type=str, default="../figures")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    # edge generation args
    parser.add_argument("--num_nodes", type=int, default=30, help="number of nodes")
    parser.add_argument("--lat_dim", type=int, default=30, help="dynamic node latent feature dimension")
    parser.add_argument("--edge_thresh", type=float, default=0.55, help="edge generation threshold")
    # node generation args
    parser.add_argument("--init_inf_prop", type=float, default=0.1, help="initial infection proportion")
    parser.add_argument("--inf_thresh", type=float, default=0.3, help="infection pressure threshold")
    parser.add_argument("--max_inf_days", type=int, default=10, help="maximum possible number of infected days")
    parser.add_argument("--int_param", type=float, nargs=2, default=None, 
                        help="Beta distribution parameter for intervenablness")
    parser.add_argument("--inf_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for infectiousnes")
    parser.add_argument("--sus_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for susceptibilty")
    parser.add_argument("--rec_param", type=float, nargs=2, default=[1.0, 1.0], 
                        help="Beta distribution parameter for recoverability")
    # intervention policy args
    parser.add_argument("--intervene_rate", type=float, default=0.0)
    parser.add_argument("--overwrite_data", action="store_true")
    # modeling args
    parser.add_argument("--num_train", type=int, default=100, help="number of trainig graphs")
    parser.add_argument("--num_val", type=int, default=100, help="number of testing graphs")
    parser.add_argument("--num_test", type=int, default=100, help="number of testing graphs")
    parser.add_argument("--model_name", type=str, choices=["GCNRNN", "SAGERNN", "SAGELSTM"], default="GCN")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_model", action="store_true")

    args = parser.parse_args()
    main(args)