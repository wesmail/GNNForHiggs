# torch imports
import torch

# deep graph library
import dgl


class GCN(torch.nn.Module):
    def __init__(self, in_feats=7, h_feats=32, num_classes=2):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats)
        self.conv2 = dgl.nn.GraphConv(h_feats, num_classes)

    def forward(self, g):
        h = self.conv1(g, g.ndata['x'])
        h = torch.nn.functional.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')
