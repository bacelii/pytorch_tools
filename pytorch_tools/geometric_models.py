import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool,global_add_pool,global_mean_pool,global_sort_pool

"""
Notes: 
Usually only have to use the batch variable when doing global pooling



"""
# ---------------- basic graph neural network models -----------
# Define our GCN class as a pytorch Module
class GCN(torch.nn.Module):
    def __init__(
        self, 
        n_hidden_channels,
        dataset_num_node_features,
        dataset_num_classes,
        n_layers = 3,
        activation_function = "relu",
        global_pool_type="mean",
                ):
        
        super(GCN, self).__init__()
        # We inherit from pytorch geometric's GCN class, and we initialize three layers
        self.conv0 = GCNConv(dataset_num_node_features, n_hidden_channels)
        for i in range(1,n_layers):
            setattr(self,f"conv{i}",GCNConv(n_hidden_channels, n_hidden_channels))
        self.n_conv = n_layers
        
        # Our final linear layer will define our output
        self.lin = Linear(n_hidden_channels, dataset_num_classes)
        self.act_func = getattr(F,activation_function)
        self.global_pool_func = eval(f"global_{global_pool_type}_pool")
                
        
    def encode(self,data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data,"batch",None)
        
        if batch is None:
            batch = torch.zeros(x.shape[0],dtype=torch.int64)
        # 1. Obtain node embeddings 
        for i in range(self.n_conv):
            x = getattr(self,f"conv{i}")(x, edge_index)
            if i < self.n_conv-1:
                x = self.act_func(x)
                    
        # 2. Readout layer
        
        x = self.global_pool_func(x, batch)  # [batch_size, hidden_channels]
        return x
    
    def forward(self, data):
        x = self.embedding(data)
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return F.softmax(x,dim=1)
    
    
# ---------- Graph Attention Network -------------
from torch_geometric.nn import GATConv
import torch.nn as nn
class GAT(nn.Module):
    """
    Source: https://github.com/marblet/GNN_models_pytorch_geometric/blob/master/models/gat.py
    """
    def __init__(
        self, 
        dataset_num_node_features, 
        dataset_num_classes,
        n_hidden_channels=8, 
        first_heads=8, 
        output_heads=1, 
        dropout=0.6,
        global_pool_type="mean"):
        super(GAT, self).__init__()
        self.gc1 = GATConv(dataset_num_node_features, n_hidden_channels,
                           heads=first_heads, dropout=dropout)
        self.gc2 = GATConv(n_hidden_channels*first_heads, dataset_num_classes,
                           heads=output_heads, dropout=dropout)
        self.dropout = dropout
        if global_pool_type is not None:
            self.global_pool_func = eval(f"global_{global_pool_type}_pool")
        else:
            self.global_pool_func = None

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def encode(self,data):    
        x, edge_index = data.x, data.edge_index
        batch = getattr(data,"batch",None)
        
        if batch is None:
            batch = torch.zeros(x.shape[0],dtype=torch.int64)
            
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        if self.global_pool_func is not None:
            x = self.global_pool_func(x, batch)
        return x
    def forward(self, data):
        x = self.encode(data)
        return F.softmax(x,dim=1)