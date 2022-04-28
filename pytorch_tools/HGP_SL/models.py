import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv

from layers import GCN, HGPSLPool

"""
Source: https://github.com/cszhangzhen/HGP-SL/blob/master/main.py
"""
class Model(torch.nn.Module):
    def __init__(
        self,
        dataset_num_node_features,
        dataset_num_classes,
        n_hidden_channels = 128,
        pooling_ratio = 0.5,
        dropout_ratio = 0.0,
        sample = True,
        sparse = True,
        sl = True,
        lamb = 1.0,
        ):
        super(Model, self).__init__()
        self.num_features = dataset_num_node_features
        self.nhid = n_hidden_channels
        self.num_classes = dataset_num_classes
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        self.sample = sample
        self.sparse = sparse
        self.sl = sl
        self.lamb = lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

        
    def encode(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        batch = getattr(data,"batch",None)
        
        if batch is None:
            batch = torch.zeros(x.shape[0],dtype=torch.int64)
            
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)
        return x

        
        
        
    def forward(self, data):
        x = self.encode(data)
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.softmax(self.lin3(x), dim=-1)

        return x
