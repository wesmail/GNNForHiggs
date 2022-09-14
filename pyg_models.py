# torch imports
import torch

# torch_geometric imports
import torch_geometric

class GINModel(torch.nn.Module):
    def __init__(self, in_feat=7, h_feat=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.gnn = torch_geometric.nn.GIN(in_feat, h_feat, 3, dropout=0.5, jk='cat')
        self.classifier = torch_geometric.nn.MLP([h_feat, h_feat*2, num_classes], norm="batch_norm", dropout=dropout)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = torch_geometric.nn.global_add_pool(x, batch)
        x = self.classifier(x)
        return x


class EdgeConv(torch.nn.Module):
    def __init__(self, in_feat=7, h_feat=128, num_classes=2, dropout=0.2):
        super().__init__()

        self.gnn = torch_geometric.nn.EdgeCNN(in_feat, h_feat, 3, dropout=dropout, jk='cat', norm="BatchNorm")
        self.classifier = torch_geometric.nn.MLP([h_feat, h_feat*2, num_classes], norm="batch_norm", dropout=dropout)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
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

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
        h = self.conv3(h, edge_index)
        h = self.conv4(h, edge_index)
        h = self.conv5(h, edge_index)

        out = torch_geometric.nn.global_add_pool(h, batch)
        out = self.lin(out)
        return out      
