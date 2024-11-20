import argparse, os, pickle, random
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from utils import train_utils, viz_utils, eval_utils
from generation.sis import DeterministicSIS, get_node_pred_feature_matrix, \
    get_node_pred_label_vector, get_edge_index, get_all_decode_index_label
from models.StaticGNN import NodeGCN, NodeSAGE, EdgeGCN, EdgeSAGE

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
        f"_recParam={args.rec_param}_intParam={args.int_param}", 
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
            assert len(np.unique(get_node_pred_label_vector(SIS.G))) > 1
            graphs.append(SIS.G.copy())
            viz_utils.plot_SIS_graph(SIS.G, path=os.path.join(data_figure_dir, f"t={i}"), pos=pos)
            SIS.update()
        with open(data_path, 'wb') as f: pickle.dump(graphs, f)
    else:
        with open(data_path, 'rb') as f: graphs = pickle.load(f)

    ### Convert Data ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_data_list, edge_data_list = [], []
    num_neg_samples = 5
    for i in tqdm(range(len(graphs)-1)):
        G_curr, G_next = graphs[i], graphs[i+1]
        # convert node model data
        edge_index = torch.from_numpy(np.array(get_edge_index(G_curr)).astype(int).T).to(device)
        x_node = torch.from_numpy(get_node_pred_feature_matrix(G_curr).astype(np.float32)).to(device)
        y_node = torch.from_numpy(np.array(get_node_pred_label_vector(G_next)).reshape(-1, 1).astype(np.float32)).to(device)
        node_data_list.append(Data(x=x_node, edge_index=edge_index, y=y_node))
        # convert edge model data
        x_edge = torch.from_numpy(np.identity(args.num_nodes).astype(np.float32)).to(device)
        if i < args.num_train: # during training decode index generated with negative sample
            for _ in range(num_neg_samples):
                positive_index = torch.from_numpy(np.array(get_edge_index(G_next)).astype(int).T).to(device)
                negative_index = negative_sampling(positive_index, num_nodes=args.num_nodes, 
                    num_neg_samples=round(positive_index.shape[1])).to(device)
                decode_index = torch.hstack((positive_index, negative_index))
                y_edge = torch.concat((torch.ones(positive_index.shape[1]), torch.zeros(negative_index.shape[1]))).to(device)
                edge_data_list.append(train_utils.PairData(x=x_edge, edge_index=edge_index, y=y_edge, decode_index=decode_index))
        else: # during val/test decode index generated with all permutation of index pairs
            decode_index, y = get_all_decode_index_label(G_next)
            decode_index = torch.from_numpy(np.array(decode_index).astype(int).T).to(device)
            y_edge = torch.from_numpy(np.array(y).astype(np.float32)).to(device)
            edge_data_list.append(train_utils.PairData(x=x_edge, edge_index=edge_index, y=y_edge, decode_index=decode_index))
    # node data train/val/test split
    node_train_data = node_data_list[:args.num_train]
    node_val_data = node_data_list[args.num_train:args.num_train+args.num_val]
    node_test_data = node_data_list[args.num_train+args.num_val:]
    node_train_dataloader = DataLoader(node_train_data, batch_size=args.batch_size)
    node_val_dataloader = DataLoader(node_val_data, batch_size=args.batch_size)
    node_test_dataloader = DataLoader(node_test_data, batch_size=args.batch_size)
    # edge data train/val/test split
    num_edge_train = args.num_train*num_neg_samples
    edge_train_data = edge_data_list[:num_edge_train]
    edge_val_data = edge_data_list[num_edge_train:num_edge_train+args.num_val]
    edge_test_data = edge_data_list[num_edge_train+args.num_val:]
    edge_train_dataloader = DataLoader(edge_train_data, batch_size=args.batch_size)
    edge_val_dataloader = DataLoader(edge_val_data, batch_size=args.batch_size)
    edge_test_dataloader = DataLoader(edge_test_data, batch_size=args.batch_size)
    
    ### Training GNN ###
    if args.model_name == "GCN":
        node_model = NodeGCN(num_node_features=node_data_list[0].x.shape[1]).to(device)
        edge_model = EdgeGCN(num_node_features=edge_data_list[0].x.shape[1]).to(device)
    elif args.model_name == "SAGE":
        node_model = NodeSAGE(num_node_features=node_data_list[0].x.shape[1]).to(device)
        edge_model = EdgeSAGE(num_node_features=edge_data_list[0].x.shape[1]).to(device)
    node_model_path = os.path.join(cp_dir, "node_model.pt")
    if not os.path.isfile(node_model_path) or args.overwrite_node_model:
        node_model, node_loss_dict, node_eval_dict = train_utils.train_torch_model(node_model, lr=args.lr, l2=args.l2, 
            epochs=args.epochs, patience=args.patience, delta=args.delta, train_dataloader=node_train_dataloader, 
            val_dataloader=node_val_dataloader, test_dataloader=node_test_dataloader, verbose=True, edge_pred=False, 
            x_label="x", y_label="y", edge_index_label="edge_index")
        torch.save({"state_dict": node_model.state_dict(), "loss_dict": node_loss_dict, 
                    "eval_dict": node_eval_dict}, node_model_path)
    else:
        node_cp = torch.load(node_model_path, weights_only=False, map_location=device)
        node_model.load_state_dict(node_cp["state_dict"]) 
        node_loss_dict, node_eval_dict = node_cp["loss_dict"], node_cp["eval_dict"]
    edge_model_path = os.path.join(cp_dir, "edge_model.pt")
    if not os.path.isfile(edge_model_path) or args.overwrite_edge_model:
        edge_model, edge_loss_dict, edge_eval_dict = train_utils.train_torch_model(edge_model, lr=args.lr, l2=args.l2, 
            epochs=args.epochs, patience=args.patience, delta=args.delta, train_dataloader=edge_train_dataloader, 
            val_dataloader=edge_val_dataloader, test_dataloader=edge_test_dataloader, verbose=True, edge_pred=True, 
            x_label="x", y_label="y", edge_index_label="edge_index", decode_index_label="decode_index")
        torch.save({"state_dict": edge_model.state_dict(), "loss_dict": edge_loss_dict, 
                    "eval_dict": edge_eval_dict}, edge_model_path)
    else:
        edge_cp = torch.load(edge_model_path, weights_only=False, map_location=device)
        edge_model.load_state_dict(edge_cp["state_dict"]) 
        edge_loss_dict, edge_eval_dict = edge_cp["loss_dict"], edge_cp["eval_dict"]

    ### Evaluate Model ###
    # plot learning curve
    fig = eval_utils.plot_learning_curves(node_loss_dict, node_eval_dict, log=True)
    fig.suptitle(f"{args.model_name} Node Prediction")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "node_loss_eval_curve.png"))
    fig = eval_utils.plot_learning_curves(edge_loss_dict, edge_eval_dict, log=True)
    fig.suptitle(f"{args.model_name} Edge Prediction")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "edge_loss_eval_curve.png"))

    # plot auroc over time
    fig = eval_utils.plot_temporal_eval(model=node_model, data_list=node_data_list[args.num_train:], eval_edge=False,
        val_time=args.num_val, y_label="y", x_label="x", metric="auroc",
        edge_index_label="edge_index", y_axis_label="Next Graph Node Prediction AUROC")
    fig.suptitle(f"{args.model_name} Node Prediction AUROC")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "node_auroc_by_time.png"))
    fig = eval_utils.plot_temporal_eval(model=edge_model, data_list=edge_data_list[num_edge_train:], eval_edge=True,
        val_time=args.num_val, y_label="y", x_label="x", metric="auroc", decode_index_label="decode_index",
        edge_index_label="edge_index", y_axis_label="Next Graph Edge Prediction AUROC")
    fig.suptitle(f"{args.model_name} Edge Prediction AUROC")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "edge_auroc_by_time.png"))

    # plot precision by time
    fig = eval_utils.plot_temporal_eval(model=node_model, data_list=node_data_list[args.num_train:], eval_edge=False,
        val_time=args.num_val, y_label="y", x_label="x", metric="precision",
        edge_index_label="edge_index", y_axis_label="Next Graph Node Prediction Precision")
    fig.suptitle(f"{args.model_name} Node Prediction Precision")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "node_precision_by_time.png"))
    fig = eval_utils.plot_temporal_eval(model=edge_model, data_list=edge_data_list[num_edge_train:], eval_edge=True,
        val_time=args.num_val, y_label="y", x_label="x", metric="precision", decode_index_label="decode_index",
        edge_index_label="edge_index", y_axis_label="Next Graph Edge Prediction Precision")
    fig.suptitle(f"{args.model_name} Edge Prediction Precision")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "edge_precision_by_time.png"))

    # plot recall by time
    fig = eval_utils.plot_temporal_eval(model=node_model, data_list=node_data_list[args.num_train:], eval_edge=False,
        val_time=args.num_val, y_label="y", x_label="x", metric="recall",
        edge_index_label="edge_index", y_axis_label="Next Graph Node Prediction Recall")
    fig.suptitle(f"{args.model_name} Node Prediction Recall")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "node_recall_by_time.png"))
    fig = eval_utils.plot_temporal_eval(model=edge_model, data_list=edge_data_list[num_edge_train:], eval_edge=True,
        val_time=args.num_val, y_label="y", x_label="x", metric="recall", decode_index_label="decode_index",
        edge_index_label="edge_index", y_axis_label="Next Graph Edge Prediction Recall")
    fig.suptitle(f"{args.model_name} Edge Prediction Recall")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "edge_recall_by_time.png"))

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
    parser.add_argument("--overwrite_data", action="store_true")
    # modeling args
    parser.add_argument("--num_train", type=int, default=100, help="number of trainig graphs")
    parser.add_argument("--num_val", type=int, default=100, help="number of testing graphs")
    parser.add_argument("--num_test", type=int, default=100, help="number of testing graphs")
    parser.add_argument("--model_name", type=str, choices=["GCN", "SAGE", "GIN", "GAT"], default="GCN")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_node_model", action="store_true")
    parser.add_argument("--overwrite_edge_model", action="store_true")

    args = parser.parse_args()
    main(args)