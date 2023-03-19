# torch imports
import torch

# torch_geometric imports
import torch_geometric

class GINModel(torch.nn.Module):
    def __init__(self, in_feat=7, h_feat=128, num_classes=2, dropout=0.1):
        super().__init__()
        self.gnn = torch_geometric.nn.GIN(in_feat, h_feat, 3, dropout=dropout, jk='cat')
        self.classifier = torch_geometric.nn.MLP([h_feat, h_feat, num_classes], norm="batch_norm", dropout=dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gnn(x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = torch_geometric.nn.global_add_pool(x, batch)
        h = self.classifier(x)
        return h


class EdgeConv(torch.nn.Module):
    def __init__(self, in_feat=7, h_feat=128, num_classes=2, dropout=0.1):
        super().__init__()

        self.gnn = torch_geometric.nn.EdgeCNN(in_feat, h_feat, 4, dropout=dropout, jk='cat', norm="BatchNorm")
        self.classifier = torch_geometric.nn.MLP([h_feat, h_feat, num_classes], norm="batch_norm", dropout=dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gnn(x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = torch_geometric.nn.global_add_pool(x, batch)
        h = self.classifier(x)
        return h
      
class GCN(torch.nn.Module):
    def __init__(self, in_feat=7, h_feat=32, num_classes=2):
        super().__init__()

        self.gcn = torch_geometric.nn.GCN(in_channels=in_feat, hidden_channels=h_feat, num_layers=4, norm="batch_norm")
        self.classifier = torch_geometric.nn.MLP([h_feat, h_feat, num_classes], norm="batch_norm")

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.gcn(x=x, edge_index=edge_index, edge_weight=edge_attr)
        h = torch_geometric.nn.global_add_pool(h, batch)
        h = self.classifier(h)
        return h     

    
class ResGCN(torch.nn.Module):
    def __init__(self,  in_feat=7, h_feat=128, num_classes=2, num_layers=5, dropout=0.1):
        super().__init__()

        self.node_encoder = torch.nn.Linear(in_feat, h_feat)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = torch_geometric.nn.GENConv(
                h_feat, h_feat, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = torch.nn.LayerNorm(h_feat, elementwise_affine=True)
            act = torch.nn.ReLU(inplace=True)

            layer = torch_geometric.nn.DeepGCNLayer(
                conv, norm, act, block='res+', dropout=dropout, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.classifier = torch.nn.Linear(h_feat, num_classes)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x=x, edge_index=edge_index, edge_weight=edge_attr)
        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index, edge_weight=edge_attr)

        x = torch_geometric.nn.global_add_pool(x, batch)
        return self.classifier(x)
