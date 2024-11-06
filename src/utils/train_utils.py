import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

def eval_model(model, dataloader, label):
    loss_list, auroc_list = [], []
    for data in dataloader:
        pred = model(data)
        loss = F.binary_cross_entropy(pred, data[label])
        loss_list.append(loss.item())
        auroc_list.append(roc_auc_score(data[label].detach().cpu().numpy(), 
            pred.detach().cpu().numpy()))
    return loss_list, auroc_list


def train_torch_model(model, lr, l2, epochs, train_dataloader, label, 
                      val_dataloader=None, test_dataloader=None, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    mean_train_loss_list, mean_val_loss_list, mean_test_loss_list = [], [], []
    mean_train_auroc_list, mean_val_auroc_list, mean_test_auroc_list = [], [], []
    for epoch in range(epochs):
        model.train()
        # train loop
        for data in train_dataloader:
            optimizer.zero_grad()
            pred = model(data)
            loss = F.binary_cross_entropy(pred, data[label])
            loss.backward()
            optimizer.step()
        # eval loop
        model.eval()
        train_loss_list, train_auroc_list = eval_model(model, train_dataloader, label)
        train_loss, train_auroc = np.mean(train_loss_list), np.mean(train_auroc_list)
        mean_train_loss_list.append(train_loss)
        mean_train_auroc_list.append(train_auroc)
        train_str = f"train: loss={train_loss:.3f} auroc={train_auroc:.3f}"
        if val_dataloader is not None:
            val_loss_list, val_auroc_list = eval_model(model, val_dataloader, label)
            val_loss, val_auroc = np.mean(val_loss_list), np.mean(val_auroc_list)
            mean_val_loss_list.append(val_loss)
            mean_val_auroc_list.append(val_auroc)
            val_str = f" val: loss={val_loss:.3f}, auroc={val_auroc:.3f}"
        else: val_str = ""
        if test_dataloader is not None:
            test_loss_list, test_auroc_list = eval_model(model, test_dataloader, label)
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