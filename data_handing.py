# generic imports
from re import S
import numpy as np
import pandas as pd

# torch imports
import torch

# dgl imports
import dgl
from torch_cluster import knn_graph


class HiggsGnnDataset(dgl.data.DGLDataset):
    """
    PyTorch class to generate graph data
    """

    def __init__(self, indata, shuffle=False, start=0, end=-1):
        super().__init__(name='HiggsDataset')
        # read in the ascii file
        self.data = indata
        if shuffle:
            self.data = self.data.sample(frac=1.0)
        # split into events
        gb = self.data.groupby('Id')
        dfs = [gb.get_group(x) for x in gb.groups]
        self.graphs = dfs[start:end]

    def load_edges(self, nodes):
        # complete graph (fully connected without self loop)
        edge_index = torch.ones(
            [nodes, nodes], dtype=torch.int32) - torch.eye(nodes, dtype=torch.int32)
        self.edge_index = edge_index.to_sparse()._indices()

    def __getitem__(self, item):
        graph = self.graphs[item]
        n_nodes = graph.shape[0]
        # edge index
        self.load_edges(n_nodes)
        # node features
        x = graph[['I1', 'I2', 'I3', 'I4', 'Pt', 'E', 'M']].to_numpy()
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(np.unique(graph['label']).item(), dtype=torch.int64)

        src, dst = self.edge_index[0], self.edge_index[1]
        g = dgl.graph((src, dst), num_nodes=n_nodes)
        g.ndata['x'] = x

        return {"graphs": g, "labels": y}

    def __len__(self):
        return len(self.graphs)
