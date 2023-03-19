# generic imports
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, confusion_matrix

# torchmetrics
from torchmetrics.classification import ConfusionMatrix, BinaryConfusionMatrix

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
    def __init__(self, module, batch_size):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        self.batch_size = batch_size

        self.model = module()

    def test_step(self, data, batch_idx):
        graphs, labels = data["graphs"], data["labels"]
        linear_out = self.model(graphs)
        
        return linear_out
    
def main():
    # some variables
    BATCH_SIZE = 512
    N_WORKERS = 8
    NUM_EPOCHS = 5
    DEVICE_ID = [0]
    MAX_EVENTS = 500000
    CHECKPOINT = 'lightning_logs/version_1/checkpoints/epoch=2-step=5046.ckpt'

    PATH = "/mnt/d/waleed/higgs/ASCII/"

    background = pd.read_csv(PATH+"ascii_bkg1.csv")
    higgs_even = pd.read_csv(PATH+"ascii_sig1.csv")
    # assign labels
    background['label'] = np.zeros(background.shape[0])
    higgs_even['label'] = np.ones(higgs_even.shape[0])

    background['event_id'] = background.event_id + MAX_EVENTS
    # combine both datasets
    df = pd.concat([background, higgs_even], ignore_index=True)
    dataset = HiggsPyGDataset(df)

    # randomly split the data into (train/val/test) (60/10/30)
    train_dataset, val_dataset, test_dataset = dgl.data.utils.split_dataset(
        dataset, frac_list=[0.6, 0.2, 0.2], shuffle=True, random_state=124)

    # create dataloaders
    graph_train_loader = torch_geometric.loader.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS, drop_last=True)   
    graph_val_loader = torch_geometric.loader.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, drop_last=True)
    graph_test_loader = torch_geometric.loader.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, drop_last=True)    

    # seed for everything
    pl.seed_everything(42)

    model = GraphLevelGNN(module=EdgeConv, batch_size=BATCH_SIZE)
    model = model.load_from_checkpoint(CHECKPOINT)

    start = datetime.now()
    
    predictions, labels = [], []

    for data in tqdm(graph_test_loader):
        linear_out = model.test_step(data, None)
        yhat       = torch.nn.functional.softmax(linear_out, dim=1).detach().numpy()
        labels.extend(data["labels"].flatten().numpy())
        predictions.extend(yhat)   

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)


    fig_1 = plt.figure(figsize=(12,10))
    _ = plt.hist(predictions[labels==1,0], bins=100, alpha=0.3, color="blue", label="signal", density=True) 
    _ = plt.hist(predictions[labels==0,0], bins=100, alpha=0.3, color="red", label="background", density=True)
    plt.semilogy()
    plt.xlabel("GNN Score", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    predictions = np.argmax(predictions, axis=1)
    fig_2 = plt.figure(figsize=(12,10))
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    # plot ROC curve
    plt.plot(fpr, tpr, lw=2.5, label="AUC = {:.1f}%".format(auc(fpr,tpr)*100))
    plt.xlabel(r'False positive rate', fontsize=16)
    plt.ylabel(r'True positive rate', fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0.,1)
    plt.xlim(0.,1)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=16)
    plt.tight_layout() 
    plt.show()

    # --------------------------------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------------------------------
    cm = confusion_matrix(labels, np.round(predictions))
    cm = np.round((cm.astype('float') / cm.sum(axis=1)), decimals=2)
    figcm, ax = plt.subplots(figsize=(12,10))

    sns.set(font_scale=2.0)
    sns.heatmap(cm, square=True, annot=True, annot_kws={"size": 22}, cmap='Blues')
    #classes=['$\pi$','   $K$','   $p$']
    #tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45, fontsize=18)
    #plt.yticks(tick_marks, classes, fontsize=18)
    #ax.set_xticks(np.arange(len(classes)), minor=True)
    #ax.set_yticks(np.arange(len(classes)), minor=True)
    plt.xlabel('Predication', horizontalalignment = 'center',  fontsize=22)
    plt.ylabel('True Species',  fontsize=22)

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------------------------------     

    end = datetime.now()
    #print("Test performance:  %4.2f%%" % (100.0 * result["acc"]))
    #print("Test ROC AUC: {}".format((100.0 * result["auc"])))
    print("Train completed in: {}".format(end-start))

if __name__ == "__main__":
    main()    