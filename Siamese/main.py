# generic imports
import pandas as pd
import numpy as np
from datetime import datetime

# torch imports
import torch

# pyg imports
import torch_geometric

# deep graph library imports
import dgl

# pytorch lightning imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

# framework imports
from utils.data_utils import HiggsPyGDataset
from utils.models import GCN, EdgeConv, GINModel

class GraphLevelGNN(pl.LightningModule):
    def __init__(self, module, batch_size, num_classes=2):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.model = module()
        # initialize metric
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_metric = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_auc = torchmetrics.AUROC(task="multiclass", num_classes=2)

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-3, weight_decay=0.0)
        return optimizer

    def training_step(self, data, batch_idx):
        graphs, labels = data["graphs"], data["labels"]
        linear_out = self.model(graphs)
        pred = torch.nn.functional.softmax(linear_out, dim=1)
        loss = torch.nn.functional.cross_entropy(pred, labels)

        self.log("train_loss", loss, batch_size=self.batch_size, prog_bar=True)
        self.log("train_acc", self.train_metric(pred, labels),
                 batch_size=self.batch_size, prog_bar=True)
        return loss

    def validation_step(self, data, batch_idx):
        graphs, labels = data["graphs"], data["labels"]
        linear_out = self.model(graphs)
        val_pred = torch.nn.functional.softmax(linear_out, dim=1)
        val_loss = torch.nn.functional.cross_entropy(val_pred, labels)

        self.log("val_loss", val_loss, sync_dist=True,
                 batch_size=self.batch_size, prog_bar=True)
        self.log("val_acc", self.val_metric(val_pred, labels), sync_dist=True,
                 batch_size=self.batch_size, prog_bar=True)

    def test_step(self, data, batch_idx):
        graphs, labels = data["graphs"], data["labels"]
        linear_out = self.model(graphs)
        test_pred = torch.nn.functional.softmax(linear_out, dim=1)
        test_loss = torch.nn.functional.cross_entropy(test_pred, labels)

        self.log("test_acc", self.test_metric(test_pred, labels),
                 batch_size=self.batch_size, prog_bar=True)
        self.log("test_auc", self.test_auc(test_pred, labels),
                 batch_size=self.batch_size, prog_bar=True)


def main():
    torch.set_float32_matmul_precision('medium')
    # some variables
    BATCH_SIZE = 256
    N_WORKERS = 4
    NUM_EPOCHS = 5
    DEVICE_ID = [0]
    MAX_EVENTS = 500000

    PATH = "/mnt/d/waleed/higgs/ASCII/"

    background = pd.read_csv(PATH+"ascii_bkg1.csv")
    higgs_even   = pd.read_csv(PATH+"ascii_sig1.csv")

    higgs_even['event_id'] = higgs_even.event_id + MAX_EVENTS

    # assign labels
    background['label'] = np.zeros(background.shape[0])
    higgs_even['label'] = np.ones(higgs_even.shape[0])

    # combine both datasets
    df = pd.concat([background, higgs_even], ignore_index=True)
    dataset = HiggsPyGDataset(df)

    # randomly split the data into (train/val/test) (60/10/30)
    train_dataset, val_dataset, test_dataset = dgl.data.utils.split_dataset(
        dataset, frac_list=[0.6, 0.1, 0.3], shuffle=True, random_state=124)

    # create dataloaders
    graph_train_loader = torch_geometric.loader.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS, drop_last=True, pin_memory=True)   
    graph_val_loader = torch_geometric.loader.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, drop_last=True, pin_memory=True)
    graph_test_loader = torch_geometric.loader.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, drop_last=True)    

    # seed for everything
    pl.seed_everything(42)

    trainer = pl.Trainer(callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         devices=DEVICE_ID, accelerator="gpu", max_epochs=NUM_EPOCHS)

    model = GraphLevelGNN(module=EdgeConv, batch_size=BATCH_SIZE)
    strat = datetime.now()
    trainer.fit(model=model, train_dataloaders=graph_train_loader,
                val_dataloaders=graph_val_loader)
    end = datetime.now()
    test_result = trainer.test(
        model, dataloaders=graph_test_loader, verbose=False)
    result = {"acc": test_result[0]["test_acc"],
              "auc": test_result[0]["test_auc"]}

    start = datetime.now()
    print("Test performance:  %4.2f%%" % (100.0 * result["acc"]))
    print("Test ROC AUC: {}".format((100.0 * result["auc"])))
    print("Train completed in: {}".format(datetime.now()-start))

if __name__ == "__main__":
    main()
