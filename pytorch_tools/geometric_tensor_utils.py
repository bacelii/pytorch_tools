import torch

import torch_geometric.nn as nn 
#global_mean_pool,global_add_pool,global_mean_pool,global_sort_pool

import torch_geometric.nn as nn_geo
def global_mean_weighted_pool(x,batch,weights):
    """
    Purpose: To do a weighted mean pooling for 
    a batch
    
    Ex: 
    import tensor_utils as tenu
    x = torch.Tensor([70,80,90,100,110,120])
    w = torch.Tensor([10,5,15,10,5,15])
    batch = torch.tensor(np.array([0,0,0,1,1,1]),dtype=torch.int64)
    tenu.global_mean_weighted_pool(x,batch,w)
    """
    weights = (weights.unsqueeze(1))
    weight_sum = nn_geo.global_add_pool(weights,batch)
    weighted_value_sum = nn_geo.global_add_pool(x*weights,batch)
    return weighted_value_sum/weight_sum

def global_mean_pool(x,batch,**kwargs):
    return nn_geo.global_mean_pool(x,batch)

def n_in_pool_from_pool_tensor(tensor):
    y = torch.ones(tensor.shape,dtype=torch.int64)
    return nn.global_add_pool(y,tensor.to(dtype=torch.int64))

def pool_idx_from_pool_tensor(tensor):
    return nn.global_mean_pool(tensor,tensor.to(dtype=torch.int64)).to(dtype=torch.int64)

def normalize_in_pool_from_pool_tensor(tensor):
    y = torch.ones(tensor.shape,dtype=torch.int64)
    numb_in_pool = nn.global_add_pool(y,tensor.to(dtype=torch.int64))
    normalize_over_pool = torch.ones(numb_in_pool.shape)/numb_in_pool
    return normalize_over_pool[tensor.to(dtype=torch.int64)]

import geometric_tensor_utils as gtu
