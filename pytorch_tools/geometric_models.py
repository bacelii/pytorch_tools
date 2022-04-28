import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool,global_add_pool,global_mean_pool,global_sort_pool
import numpy as np
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
    

# ---- simple diff pool -------------
"""
Source: https://github.com/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial16/Tutorial16.ipynb

Note: This requires the data to be a dense matrix 
1. so need to T.ToDense() transform
2. and the DenseDataLoader

-- Still had bugs and couldn't get to work


"""


from torch_geometric.nn import DenseGCNConv as DenseGCNConv, dense_diff_pool
class DiffPoolSimpleGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        normalize=False,
        lin=True,
        activation_function = "relu",
        use_bn = True,):
        super(DiffPoolSimpleGNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(DenseGCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(DenseGCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(DenseGCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))
        
        self.act_func = getattr(F,activation_function)
        
        self.use_bn = use_bn


    def forward(self, x=None,adj = None,mask=None,data=None):
        if x is None:
            x,adj,mask = data.x,data.adj,data.mask
            
        
        
        for step in range(len(self.convs)):
            x = F.relu(self.convs[step](x, adj, mask))
            if self.use_bn:
                self.bns[step] = nn.BatchNorm1d(x.size(1))#.to(self.device)
                x = self.bns[step](x)
        
        return x
from math import ceil
class DiffPoolSimple(torch.nn.Module):
    def __init__(
        self,
        dataset_num_node_features, 
        dataset_num_classes,
        n_hidden_channels=64, 
        max_nodes=150,
        ):
        super(DiffPoolSimple, self).__init__()

#         if max_nodes > dataset_num_node_features:
#             max_nodes = dataset_num_node_features
        
        num_nodes = ceil(0.25 * max_nodes)
        
        self.gnn1_pool = DiffPoolSimpleGNN(dataset_num_node_features, n_hidden_channels, num_nodes)
        self.gnn1_embed = DiffPoolSimpleGNN(dataset_num_node_features, n_hidden_channels, n_hidden_channels)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = DiffPoolSimpleGNN(n_hidden_channels, n_hidden_channels, num_nodes)
        self.gnn2_embed = DiffPoolSimpleGNN(n_hidden_channels, n_hidden_channels, n_hidden_channels, lin=False)

        self.gnn3_embed = DiffPoolSimpleGNN(n_hidden_channels, n_hidden_channels, n_hidden_channels, lin=False)

        self.lin1 = torch.nn.Linear(n_hidden_channels, n_hidden_channels)
        self.lin2 = torch.nn.Linear(n_hidden_channels, dataset_num_classes)

        
    def encode(self,data,return_loss = True):
        x,adj,mask = data.x, data.adj, data.mask
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        #x_1 = s_0.t() @ z_0
        #adj_1 = s_0.t() @ adj_0 @ s_0
        
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        if return_loss:
            return x,l1+l2,e1+e2
        else:
            return x
    def forward(self,data):
        x,gnn_loss,cluster_loss = self.encode(data,return_loss = True)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.softmax(x, dim=-1), gnn_loss, cluster_loss
    
    
# ----------- Graph Sage and Diff Pool ----------
"""
Official source: https://github.com/RexYing/diffpool

pyg implementation paper source = https://github.com/VoVAllen/diffpool
"""

class BatchedGraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat, device='cpu', use_bn=True, mean=False, add_self=False):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.device = device
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj, mask=None):
        if self.add_self:
            adj = adj + torch.eye(adj.size(0)).to(self.device)

        if self.mean:
            adj = adj / adj.sum(1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(h_k.size(1)).to(self.device)
            h_k = self.bn(h_k)
        if mask is not None:
            h_k = h_k * mask.unsqueeze(2).expand_as(h_k)
        return h_k


class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, is_final=False, device='cpu', link_pred=False):
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        self.device = device
        self.is_final = is_final
        self.embed = BatchedGraphSAGE(nfeat, nhid, device=self.device, use_bn=True)
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, device=self.device, use_bn=True)
        self.log = {}
        self.link_pred_loss = 0
        self.entropy_loss = 0

    def forward(self, x, adj, mask=None, log=False):
        z_l = self.embed(x, adj)
        s_l = F.softmax(self.assign_mat(x, adj), dim=-1)
        if log:
            self.log['s'] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)
        if self.link_pred:
            # TODO: Masking padded s_l
            self.link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
            self.entropy_loss = torch.distributions.Categorical(probs=s_l).entropy()
            if mask is not None:
                self.entropy_loss = self.entropy_loss * mask.expand_as(self.entropy_loss)
            self.entropy_loss = self.entropy_loss.sum(-1)
        return xnext, anext
    
# ------------- Models that did not work ------------
"""
import geometric_models as gm
sys.path.append("/pytorch_tools/pytorch_tools/HGP_SL")
import models

model_name = "HGP_SL"
n_epochs = 500


architecture_kwargs = dict(
    n_hidden_channels = 8, 
    #first_heads=8, 
    #output_heads=1, 
    #dropout=0.6,
    #global_pool_type="mean"
)

model = models.Model(
    dataset_num_node_features=dataset.num_node_features,
    dataset_num_classes=dataset.num_classes,
    **architecture_kwargs
    )

"""