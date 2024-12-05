import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
from utils.eval_utils import eval_static_model, eval_dynamic_model

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['edge_index', 'decode_index']: return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def train_static_torch_model(model, lr, l2, epochs, train_dataloader, edge_pred=False,
        val_dataloader=None, test_dataloader=None, verbose=True, decode_index_label=None,
        x_label="x", y_label="y", edge_index_label="edge_index", patience=10, delta=1e-4):
    """
    Trains a PyTorch model with optional early stopping.

    Parameters:
        model: PyTorch model to train.
        lr: Learning rate.
        l2: Weight decay (L2 regularization).
        epochs: Number of epochs to train.
        train_dataloader: DataLoader for training data.
        edge_pred: Boolean, whether edge prediction is enabled.
        val_dataloader: DataLoader for validation data.
        test_dataloader: DataLoader for test data.
        verbose: Boolean, whether to print progress.
        decode_index_label: Name of the decode index key in the data (required if edge_pred=True).
        x_label: Key for input features in the data.
        y_label: Key for target labels in the data.
        edge_index_label: Key for edge indices in the data.
        patience: Number of epochs to wait for improvement in validation loss before stopping.
        delta: Minimum improvement in validation loss to reset patience.

    Returns:
        model: Trained model.
        loss_dict: Dictionary of loss histories.
        eval_dict: Dictionary of evaluation metric histories.
    """
    if edge_pred: 
        assert decode_index_label is not None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    mean_train_loss_list, mean_val_loss_list, mean_test_loss_list = [], [], []
    mean_train_auroc_list, mean_val_auroc_list, mean_test_auroc_list = [], [], []

    best_val_loss = float('inf')  # Initialize best validation loss
    best_model = copy.deepcopy(model)
    patience_counter = 0          # Counter for early stopping

    for epoch in range(epochs):
        model.train()
        # train loop
        for data in train_dataloader:
            optimizer.zero_grad()
            if edge_pred:
                pred = model(data, x_label=x_label, edge_index_label=edge_index_label, 
                             decode_index=data[decode_index_label])
            else:
                pred = model(data, x_label=x_label, edge_index_label=edge_index_label)
            loss = F.binary_cross_entropy(pred, data[y_label])
            loss.backward()
            optimizer.step()

        # eval loop
        model.eval()
        train_loss_list, train_auroc_list = eval_static_model(model, train_dataloader, 
            y_label=y_label, x_label=x_label, eval_edge=edge_pred, metric="auroc",
            decode_index_label=decode_index_label, edge_index_label=edge_index_label)
        train_loss, train_auroc = np.mean(train_loss_list), np.mean(train_auroc_list)
        mean_train_loss_list.append(train_loss)
        mean_train_auroc_list.append(train_auroc)
        train_str = f"train: loss={train_loss:.3f} auroc={train_auroc:.3f}"

        if val_dataloader is not None:
            val_loss_list, val_auroc_list = eval_static_model(model, val_dataloader, 
                y_label=y_label, x_label=x_label, eval_edge=edge_pred, metric="auroc",
                decode_index_label=decode_index_label, edge_index_label=edge_index_label)
            val_loss, val_auroc = np.mean(val_loss_list), np.mean(val_auroc_list)
            mean_val_loss_list.append(val_loss)
            mean_val_auroc_list.append(val_auroc)
            val_str = f" val: loss={val_loss:.3f}, auroc={val_auroc:.3f}"
        else: 
            val_loss, val_str = None, ""

        if test_dataloader is not None:
            test_loss_list, test_auroc_list = eval_static_model(model, test_dataloader, 
                y_label=y_label, x_label=x_label, eval_edge=edge_pred, metric="auroc",
                decode_index_label=decode_index_label, edge_index_label=edge_index_label)
            test_loss, test_auroc = np.mean(test_loss_list), np.mean(test_auroc_list)
            mean_test_loss_list.append(test_loss)
            mean_test_auroc_list.append(test_auroc)
            test_str = f" test: loss={test_loss:.3f}, auroc={test_auroc:.3f}"
        else: 
            test_str = ""

        if verbose and (epoch+1)%5==0:
            print(f"Epoch=[{epoch+1}/{epochs}]: ")
            print(f"\t" + train_str + val_str + test_str)

        # Early stopping logic
        if val_loss is not None:
            if val_loss < best_val_loss - delta:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                break
        else:
            best_model = copy.deepcopy(model)
        
    loss_dict = {"train": mean_train_loss_list, "val": mean_val_loss_list, "test": mean_test_loss_list}
    eval_dict = {"train": mean_train_auroc_list, "val": mean_val_auroc_list, "test": mean_test_auroc_list}
    return best_model, loss_dict, eval_dict


def train_dynamic_torch_model(model, lr, l2, epochs, train_data_list, edge_pred=False,
        val_data_list=None, test_data_list=None, verbose=True,
        input_label_dict={"x_label":"X", "y_label":"Y_curr", "edge_index_label":"edge_index"},
        patience=10, delta=1e-4):
    """
    Trains a PyTorch model with optional early stopping.

    Parameters:
        model: PyTorch model to train.
        lr: Learning rate.
        l2: Weight decay (L2 regularization).
        epochs: Number of epochs to train.
        train_data_list: List of training data.
        edge_pred: Boolean, whether edge prediction is enabled.
        val_data_list: List of validation data.
        test_data_list: List of test data.
        verbose: Boolean, whether to print progress.
        decode_index_label: Name of the decode index key in the data (required if edge_pred=True).
        x_label: Key for input features in the data.
        y_label: Key for target labels in the data.
        edge_index_label: Key for edge indices in the data.
        patience: Number of epochs to wait for improvement in validation loss before stopping.
        delta: Minimum improvement in validation loss to reset patience.

    Returns:
        model: Trained model.
        loss_dict: Dictionary of loss histories.
        eval_dict: Dictionary of evaluation metric histories.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    mean_y_train_loss_list, mean_y_val_loss_list, mean_y_test_loss_list = [], [], []
    mean_y_train_eval_list, mean_y_val_eval_list, mean_y_test_eval_list = [], [], []
    mean_a_train_loss_list, mean_a_val_loss_list, mean_a_test_loss_list = [], [], []
    mean_a_train_eval_list, mean_a_val_eval_list, mean_a_test_eval_list = [], [], []

    best_val_loss = float('inf')  # Initialize best validation loss
    best_model = copy.deepcopy(model)
    patience_counter = 0          # Counter for early stopping

    for epoch in range(epochs):
        model.train()
        # train loop
        y_hidden, a_hidden = None, None
        for data in train_data_list:
            optimizer.zero_grad()
            y_score, a_score, y_hidden, a_hidden = model(
                data, y_hidden=y_hidden, a_hidden=a_hidden, **input_label_dict)
            if y_hidden is not None:
                y_hidden = tuple(y.detach() for y in y_hidden)
            if a_hidden is not None:
                a_hidden = tuple(a.detach() for a in a_hidden)
            mask = torch.triu(torch.ones_like(a_score, dtype=torch.bool), diagonal=1)
            a_score, a_target = a_score[mask], data["A_next"][mask]
            loss = nn.CrossEntropyLoss()(y_score, data["Y_next"]) + \
                   nn.BCEWithLogitsLoss(pos_weight=a_target.size(0)/a_target.sum())(a_score, a_target)
            loss.backward()
            optimizer.step()

        # eval loop
        model.eval()
        eval_metric = "auroc" 
        y_train_loss_list, a_train_loss_list, y_train_eval_list, a_train_eval_list = eval_dynamic_model(
            model, train_data_list, input_label_dict, metric=eval_metric)
        y_train_loss, y_train_eval = np.mean(y_train_loss_list), np.mean(y_train_eval_list)
        a_train_loss, a_train_eval = np.mean(a_train_loss_list), np.mean(a_train_eval_list)
        mean_y_train_loss_list.append(y_train_loss)
        mean_y_train_eval_list.append(y_train_eval)
        mean_a_train_loss_list.append(a_train_loss)
        mean_a_train_eval_list.append(a_train_eval)
        train_str = f"train: Y:loss={y_train_loss:.3f} {eval_metric}={y_train_eval:.3f}" \
            f" A:loss={a_train_loss:.3f} {eval_metric}={a_train_eval:.3f}"

        if val_data_list is not None:
            y_val_loss_list, a_val_loss_list, y_val_eval_list, a_val_eval_list = eval_dynamic_model(
               model, val_data_list, input_label_dict, metric=eval_metric)
            y_val_loss, y_val_eval = np.mean(y_val_loss_list), np.mean(y_val_eval_list)
            a_val_loss, a_val_eval = np.mean(a_val_loss_list), np.mean(a_val_eval_list)
            mean_y_val_loss_list.append(y_val_loss)
            mean_y_val_eval_list.append(y_val_eval)
            mean_a_val_loss_list.append(a_val_loss)
            mean_a_val_eval_list.append(a_val_eval)
            val_str = f"val: Y:loss={y_val_loss:.3f}, {eval_metric}={y_val_eval:.3f}" \
                f" A:loss={a_val_loss:.3f}, {eval_metric}={a_val_eval:.3f}"
        else: 
            y_val_loss, a_val_loss, y_val_eval, a_val_eval, val_str = None, None, None, None, ""

        if test_data_list is not None:
            y_test_loss_list, a_test_loss_list, y_test_eval_list, a_test_eval_list = eval_dynamic_model(
               model, test_data_list, input_label_dict, metric=eval_metric)
            y_test_loss, y_test_eval = np.mean(y_test_loss_list), np.mean(y_test_eval_list)
            a_test_loss, a_test_eval = np.mean(a_test_loss_list), np.mean(a_test_eval_list)
            mean_y_test_loss_list.append(y_test_loss)
            mean_y_test_eval_list.append(y_test_eval)
            mean_a_test_loss_list.append(a_test_loss)
            mean_a_test_eval_list.append(a_test_eval)
            test_str = f"test: Y:loss={y_test_loss:.3f}, {eval_metric}={y_test_eval:.3f}" \
                f" A:loss={a_test_loss:.3f}, {eval_metric}={a_test_eval:.3f}"
        else: 
            y_test_loss, a_test_loss, y_test_eval, a_test_eval, test_str = None, None, None, None,""

        if verbose and (epoch+1)%5==0:
            print(f"Epoch=[{epoch+1}/{epochs}]: ")
            print(f"\t{train_str}\n\t{val_str}\n\t{test_str}")

        # Early stopping logic
        if (y_val_loss+a_val_loss) is not None:
            if y_val_loss+a_val_loss < best_val_loss - delta:
                best_val_loss = y_val_loss+a_val_loss
                best_model = copy.deepcopy(model)
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
                break
        else:
            best_model = copy.deepcopy(model)
    loss_dict = {"train": {"Y": mean_y_train_loss_list, "A": mean_a_train_loss_list}, 
                 "val": {"Y": mean_y_val_loss_list, "A": mean_a_val_loss_list}, 
                 "test": {"Y": mean_y_test_loss_list, "A": mean_a_test_loss_list}}
    eval_dict = {"train": {"Y": mean_y_train_eval_list, "A": mean_a_train_eval_list}, 
                 "val": {"Y": mean_y_val_eval_list, "A": mean_a_val_eval_list}, 
                 "test": {"Y": mean_y_test_eval_list, "A": mean_a_test_eval_list}}
    return best_model, loss_dict, eval_dict