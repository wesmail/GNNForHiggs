# torch imports
import torch

# torch_geometric imports
import torch_geometric

class GINModel(torch.nn.Module):
    def __init__(self, in_feat=7, h_feat=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.gnn = torch_geometric.nn.GIN(in_feat, h_feat, 3, dropout=0.5, jk='cat')
        self.classifier = torch_geometric.nn.MLP([h_feat, h_feat*2, num_classes], norm="batch_norm", dropout=dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gnn(x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = torch_geometric.nn.global_add_pool(x, batch)
        x = self.classifier(x)
        return x


class EdgeConv(torch.nn.Module):
    def __init__(self, in_feat=7, h_feat=64, num_classes=2, dropout=0.5):
        super().__init__()

        self.gnn = torch_geometric.nn.EdgeCNN(in_feat, h_feat, 3, dropout=dropout, jk='cat', norm="BatchNorm")
        self.classifier = torch_geometric.nn.MLP([h_feat, h_feat*2, num_classes], norm="batch_norm", dropout=dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gnn(x=x, edge_index=edge_index, edge_weight=edge_attr)
        x = torch_geometric.nn.global_add_pool(x, batch)
        x = self.classifier(x)
        return x
      
class GCN(torch.nn.Module):
    def __init__(self, in_feat=7, h_feat=8, num_classes=2):
        super().__init__()

        self.conv1 = torch_geometric.nn.GraphConv(in_feat, h_feat)
        self.conv2 = torch_geometric.nn.GraphConv(h_feat, h_feat*2)
        self.conv3 = torch_geometric.nn.GraphConv(h_feat*2, h_feat*4)
        self.conv4 = torch_geometric.nn.GraphConv(h_feat*4, h_feat*8)
        self.conv5 = torch_geometric.nn.GraphConv(h_feat*8, h_feat*16)
        self.lin = torch.nn.Linear(h_feat*16, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_attr)
        h = self.conv2(x=h, edge_index=edge_index, edge_weight=edge_attr)
        h = self.conv3(x=h, edge_index=edge_index, edge_weight=edge_attr)
        h = self.conv4(x=h, edge_index=edge_index, edge_weight=edge_attr)
        h = self.conv5(x=h, edge_index=edge_index, edge_weight=edge_attr)

        out = torch_geometric.nn.global_add_pool(h, batch)
        out = self.lin(out)
        return out      

    
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
