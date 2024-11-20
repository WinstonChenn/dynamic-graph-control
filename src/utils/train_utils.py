import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
from utils.eval_utils import eval_model

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['edge_index', 'decode_index']: return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def train_torch_model(model, lr, l2, epochs, train_dataloader, edge_pred=False,
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
        train_loss_list, train_auroc_list = eval_model(model, train_dataloader, 
            y_label=y_label, x_label=x_label, eval_edge=edge_pred, metric="auroc",
            decode_index_label=decode_index_label, edge_index_label=edge_index_label)
        train_loss, train_auroc = np.mean(train_loss_list), np.mean(train_auroc_list)
        mean_train_loss_list.append(train_loss)
        mean_train_auroc_list.append(train_auroc)
        train_str = f"train: loss={train_loss:.3f} auroc={train_auroc:.3f}"

        if val_dataloader is not None:
            val_loss_list, val_auroc_list = eval_model(model, val_dataloader, 
                y_label=y_label, x_label=x_label, eval_edge=edge_pred, metric="auroc",
                decode_index_label=decode_index_label, edge_index_label=edge_index_label)
            val_loss, val_auroc = np.mean(val_loss_list), np.mean(val_auroc_list)
            mean_val_loss_list.append(val_loss)
            mean_val_auroc_list.append(val_auroc)
            val_str = f" val: loss={val_loss:.3f}, auroc={val_auroc:.3f}"
        else: 
            val_loss, val_str = None, ""

        if test_dataloader is not None:
            test_loss_list, test_auroc_list = eval_model(model, test_dataloader, 
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