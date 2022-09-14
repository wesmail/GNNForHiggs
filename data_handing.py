# generic imports
from re import S
import numpy as np
import pandas as pd
import scipy
import sklearn.preprocessing as sklp

# torch imports
import torch

# dgl imports
import dgl
from torch_cluster import knn_graph

# pyg
import torch_geometric
import torch_geometric.data as PyGData

class HiggsDGLDataset(dgl.data.DGLDataset):
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
        # edge index (use complete graph without edge weighting)
        self.load_edges(n_nodes)
        # node features
        x = graph[['I1', 'I2', 'I3', 'I4', 'Pt', 'E', 'M']].to_numpy()

        # edge index and edge attributes
        # compute pair-wise distance matrix from (eta, phi)
        hitPosMatrix = graph[['Eta', 'Phi']].to_numpy()
        hitDistMatrix = scipy.spatial.distance_matrix(
            hitPosMatrix, hitPosMatrix)
        norm_hitDistMatrix = sklp.normalize(hitDistMatrix)
        norm_sparse_hitDistMatrix = scipy.sparse.csr_matrix(norm_hitDistMatrix)
        hitEdges = torch_geometric.utils.convert.from_scipy_sparse_matrix(
            norm_sparse_hitDistMatrix)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(np.unique(graph['label']).item(), dtype=torch.int64)

        src, dst = hitEdges[0][0],  hitEdges[0][1]
        g = dgl.graph((src, dst), num_nodes=n_nodes)
        g.ndata['x'] = x
        g.edata['e'] = hitEdges[1].float()

        return {"graphs": g, "labels": y}

    def __len__(self):
        return len(self.graphs)

    
class HiggsPyGDataset(torch_geometric.data.InMemoryDataset):
    """
    PyTorch class to generate graph data
    """
    def __init__(self, indata, shuffle=False, start=0, end=-1):
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
        # edge index (use complete graph without edge weighting)
        self.load_edges(n_nodes)
        # node features
        x = graph[['I1', 'I2', 'I3', 'I4', 'Pt', 'E', 'M']].to_numpy()

        # edge index and edge attributes
        # compute pair-wise distance matrix from (eta, phi)
        hitPosMatrix = graph[['Eta', 'Phi']].to_numpy()
        hitDistMatrix = scipy.spatial.distance_matrix(
            hitPosMatrix, hitPosMatrix)
        norm_hitDistMatrix = sklp.normalize(hitDistMatrix)
        norm_sparse_hitDistMatrix = scipy.sparse.csr_matrix(norm_hitDistMatrix)
        hitEdges = torch_geometric.utils.convert.from_scipy_sparse_matrix(
            norm_sparse_hitDistMatrix)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(np.unique(graph['label']).item(), dtype=torch.int64)
        g = PyGData.Data(x=x, y=y, edge_index=hitEdges[0], edge_attr=hitEdges[1])
        
        return {"graphs": g, "labels": y}

    def __len__(self):
        return len(self.graphs)    
