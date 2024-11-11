import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import roc_auc_score

def get_upper_tri_values(mat):
    m = mat.shape[0]
    r,c = np.triu_indices(m,1)
    return mat[r,c]

def eval_model(model, dataloader, y_label, x_label="x", eval_edge=False, decode_index_label=None):
    if eval_edge: assert decode_index_label is not None
    loss_list, auroc_list = [], []
    for data in dataloader:
        if eval_edge:
            pred = model(data, x_label=x_label, decode_index=data[decode_index_label])
        else:
            pred = model(data, x_label)
        target = data[y_label]
        loss = F.binary_cross_entropy(pred, target)
        loss_list.append(loss.item())
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        auroc_list.append(roc_auc_score(target_np, pred_np))
    return loss_list, auroc_list

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['edge_index_curr', 'edge_index_next', 'decode_index_curr', 'decode_index_next']:
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def train_torch_model(model, lr, l2, epochs, train_dataloader, edge_pred=False,
        val_dataloader=None, test_dataloader=None, verbose=True, decode_index_label=None,
        x_label="x", y_label="y", edge_index_label="edge_index"):
    if edge_pred: assert decode_index_label is not None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    mean_train_loss_list, mean_val_loss_list, mean_test_loss_list = [], [], []
    mean_train_auroc_list, mean_val_auroc_list, mean_test_auroc_list = [], [], []
    for epoch in range(epochs):
        model.train()
        # train loop
        for data in train_dataloader:
            optimizer.zero_grad()
            if edge_pred:
                pred = model(data, x_label=x_label, edge_index_label=edge_index_label, 
                             decode_index=data[decode_index_label])
                loss = F.binary_cross_entropy(pred, data[y_label])
            else:
                pred = model(data, x_label=x_label, edge_index_label=edge_index_label)
                loss = F.binary_cross_entropy(pred, data[y_label])
            loss.backward()
            optimizer.step()

        # eval loop
        model.eval()
        train_loss_list, train_auroc_list = eval_model(model, train_dataloader, 
            y_label=y_label, x_label=x_label, eval_edge=edge_pred, 
            decode_index_label=decode_index_label)
        train_loss, train_auroc = np.mean(train_loss_list), np.mean(train_auroc_list)
        mean_train_loss_list.append(train_loss)
        mean_train_auroc_list.append(train_auroc)
        train_str = f"train: loss={train_loss:.3f} auroc={train_auroc:.3f}"
        if val_dataloader is not None:
            val_loss_list, val_auroc_list = eval_model(model, val_dataloader, 
                y_label=y_label, x_label=x_label, eval_edge=edge_pred, 
                decode_index_label=decode_index_label)
            val_loss, val_auroc = np.mean(val_loss_list), np.mean(val_auroc_list)
            mean_val_loss_list.append(val_loss)
            mean_val_auroc_list.append(val_auroc)
            val_str = f" val: loss={val_loss:.3f}, auroc={val_auroc:.3f}"
        else: val_str = ""
        if test_dataloader is not None:
            test_loss_list, test_auroc_list = eval_model(model, test_dataloader, 
                y_label=y_label, x_label=x_label, eval_edge=edge_pred, 
                decode_index_label=decode_index_label)
            test_loss, test_auroc = np.mean(test_loss_list), np.mean(test_auroc_list)
            mean_test_loss_list.append(test_loss)
            mean_test_auroc_list.append(test_auroc)
            test_str = f" test: loss={test_loss:.3f}, auroc={test_auroc:.3f}"
        else: test_str = ""
        if verbose and (epoch+1)%5==0:
            print(f"Epoch=[{epoch+1}/{epochs}]: ")
            print(f"\t" + train_str + val_str + test_str)
        
        
    loss_dict = {"train": mean_train_loss_list, "val": mean_val_loss_list, "test": mean_test_loss_list}
    eval_dict = {"train": mean_train_auroc_list, "val": mean_val_auroc_list, "test": mean_test_auroc_list}
    return model, loss_dict, eval_dict