# generic imports
from cProfile import label
import pandas as pd
import numpy as np
from datetime import datetime

# torch imports
import torch

# deep graph library imports
import dgl

# pytorch lightning imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

# framework imports
from data_handing import HiggsGnnDataset
from models import GCN


class GraphLevelGNN(pl.LightningModule):
    def __init__(self, module, batch_size, num_classes=2):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.model = module()
        # initialize metric
        self.train_metric = torchmetrics.Accuracy()
        self.val_metric = torchmetrics.Accuracy()
        self.test_metric = torchmetrics.Accuracy()
        self.test_auc = torchmetrics.AUROC(num_classes=num_classes)

    def configure_optimizers(self):
        # High lr because of small dataset and small model
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=5e-3, weight_decay=0.0)
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
        pred = torch.nn.functional.softmax(linear_out, dim=1)
        val_loss = torch.nn.functional.cross_entropy(pred, labels)
        self.log("val_loss", val_loss, sync_dist=True,
                 batch_size=self.batch_size, prog_bar=True)
        self.log("val_acc", self.val_metric(pred, labels), sync_dist=True,
                 batch_size=self.batch_size, prog_bar=True)

    def test_step(self, data, batch_idx):
        graphs, labels = data["graphs"], data["labels"]
        linear_out = self.model(graphs)
        pred = torch.nn.functional.softmax(linear_out, dim=1)
        self.log("test_acc", self.test_metric(pred, labels),
                 batch_size=self.batch_size, prog_bar=True)
        self.log("test_auc", self.test_auc(pred, labels),
                 batch_size=self.batch_size, prog_bar=True)


def main():
    # some variables
    batch_size = 8
    num_workers = 4
    num_epochs = 5
    device_id = [0]

    background = pd.read_csv(
        "Delphes-3.5.0/data_ascii_background.csv", index_col=0)
    higgs_even = pd.read_csv("Delphes-3.5.0/data_ascii_higgs.csv", index_col=0)
    # assign labels
    background['label'] = np.zeros(background.shape[0])
    higgs_even['label'] = np.ones(higgs_even.shape[0])

    # combine both datasets
    df = pd.concat([background, higgs_even], ignore_index=True)
    dataset = HiggsGnnDataset(df)

    # randomly split the data into (train/val/test) (60/10/30)
    train_dataset, val_dataset, test_dataset = dgl.data.utils.split_dataset(
        dataset, frac_list=[0.6, 0.1, 0.3], shuffle=True, random_state=124)

    # create dataloaders
    graph_train_loader = dgl.dataloading.GraphDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    graph_val_loader = dgl.dataloading.GraphDataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    graph_test_loader = dgl.dataloading.GraphDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # seed for everything
    pl.seed_everything(42)

    trainer = pl.Trainer(callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         devices=device_id, accelerator="gpu", max_epochs=num_epochs)

    model = GraphLevelGNN(module=GCN, batch_size=batch_size)
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
