import argparse, os, pickle, random
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from utils import train_utils, viz_utils, eval_utils
from generation.sis import DeterministicSIS, get_node_label_vector, get_edge_index, \
    get_decode_index_label
from models.StaticGNN import EdgeGCN, EdgeSAGE

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
    cp_dir = os.path.join(args.cp_dir, data_str, "EdgePrediction", model_str)
    os.makedirs(cp_dir, exist_ok=True)
    figure_dir = os.path.join(args.figure_dir, data_str)
    os.makedirs(figure_dir, exist_ok=True)
    data_figure_dir = os.path.join(figure_dir, "data")
    os.makedirs(data_figure_dir, exist_ok=True)
    model_figure_dir = os.path.join(figure_dir, "model", "EdgePrediction", model_str)
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
            viz_utils.plot_SIS_graph(SIS.G, path=os.path.join(data_figure_dir, f"t={i}.png"), pos=pos)
            SIS.update()
        with open(data_path, 'wb') as f: pickle.dump(graphs, f)
    else:
        with open(data_path, 'rb') as f: graphs = pickle.load(f)

    # convert networkx to pyG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_list = []
    num_samples = 5
    for i in tqdm(range(len(graphs)-1)):
        G_curr, G_next = graphs[i], graphs[i+1]
        edge_index = torch.from_numpy(np.array(get_edge_index(G_curr)).astype(int).T).to(device)
        x = torch.from_numpy(np.identity(args.num_nodes).astype(np.float32)).to(device)
        if i < args.num_train: 
            for _ in range(num_samples):
                positive_index = torch.from_numpy(np.array(get_edge_index(G_next)).astype(int).T).to(device)
                negative_index = negative_sampling(positive_index, num_nodes=args.num_nodes, 
                    num_neg_samples=round(positive_index.shape[1])).to(device)
                decode_index = torch.hstack((positive_index, negative_index))
                y = torch.concat((torch.ones(positive_index.shape[1]), torch.zeros(negative_index.shape[1]))).to(device)
                data_list.append(train_utils.PairData(x=x, edge_index=edge_index, y=y, decode_index=decode_index))
        else:
            decode_index, y = get_decode_index_label(G_next)
            decode_index = torch.from_numpy(np.array(decode_index).astype(int).T).to(device)
            y = torch.from_numpy(np.array(y).astype(np.float32)).to(device)
            data_list.append(train_utils.PairData(x=x, edge_index=edge_index, y=y, decode_index=decode_index))
        
    train_data, val_data = data_list[:args.num_train*num_samples], data_list[args.num_train*num_samples:args.num_train*num_samples+args.num_val]
    test_data = data_list[args.num_train*num_samples+args.num_val:]
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    ### Training GNN ###
    if args.model_name == "GCN":
        model = EdgeGCN(num_node_features=data_list[0].x.shape[1]).to(device)
    elif args.model_name == "SAGE":
        model = EdgeSAGE(num_node_features=data_list[0].x.shape[1]).to(device)
   
    # train edge prediction GNN
    model_path = os.path.join(cp_dir, "model.pt")
    if not os.path.isfile(model_path) or args.overwrite_model:
        model, loss_dict, eval_dict = train_utils.train_torch_model(model, lr=args.lr, 
            l2=args.l2, epochs=args.epochs, train_dataloader=train_dataloader, val_dataloader=val_dataloader, 
            test_dataloader=test_dataloader, edge_pred=True, verbose=True, x_label="x", edge_index_label="edge_index", 
            y_label="y", decode_index_label="decode_index")
        torch.save({"state_dict": model.state_dict(), "loss_dict": loss_dict, 
                    "eval_dict": eval_dict}, model_path)
    else:
        cp = torch.load(model_path, weights_only=False, map_location=device)
        model.load_state_dict(cp["state_dict"]) 
        loss_dict, eval_dict = cp["loss_dict"], cp["eval_dict"]
    
    model.eval()
    # plot loss and eval
    fig = eval_utils.plot_learning_curves(loss_dict, eval_dict)
    fig.suptitle(f"{args.model_name} Edge Prediction")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "loss_eval_curve.png"))
    
    # plot auroc by time
    fig = eval_utils.plot_temporal_eval(model=model, data_list=data_list[args.num_train*num_samples:], eval_edge=True,
        val_time=args.num_val, y_label="y", x_label="x", metric="auroc",
        edge_index_label="edge_index", decode_index_label="decode_index", y_axis_label="Next Graph Edge Prediction AUROC")
    fig.suptitle(f"{args.model_name} Edge Prediction AUROC")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "auroc_by_time.png"))

    # plot precision by time
    fig = eval_utils.plot_temporal_eval(model=model, data_list=data_list[args.num_train*num_samples:], eval_edge=True,
        val_time=args.num_val, y_label="y", x_label="x", metric="precision",
        edge_index_label="edge_index", decode_index_label="decode_index", y_axis_label="Next Graph Edge Prediction Precision")
    fig.suptitle(f"{args.model_name} Edge Prediction Precision")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "precision_by_time.png"))

    # plot recall by time
    fig = eval_utils.plot_temporal_eval(model=model, data_list=data_list[args.num_train*num_samples:], eval_edge=True,
        val_time=args.num_val, y_label="y", x_label="x", metric="recall",
        edge_index_label="edge_index", decode_index_label="decode_index", y_axis_label="Next Graph Edge Prediction Recall")
    fig.suptitle(f"{args.model_name} Edge Prediction Recall Recall")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "recall_by_time.png"))

    # plot f1 by time
    fig = eval_utils.plot_temporal_eval(model=model, data_list=data_list[args.num_train*num_samples:], eval_edge=True,
        val_time=args.num_val, y_label="y", x_label="x", metric="f1",
        edge_index_label="edge_index", decode_index_label="decode_index", y_axis_label="Next Graph Edge Prediction F1")
    fig.suptitle(f"{args.model_name} Edge Prediction F1")
    fig.tight_layout()
    fig.savefig(os.path.join(model_figure_dir, "f1_by_time.png"))


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
    parser.add_argument("--model_name", type=str, choices=["GCN", "SAGE"], default="GCN")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=5e-4)
    parser.add_argument("--overwrite_model", action="store_true")
    args = parser.parse_args()

    main(args)