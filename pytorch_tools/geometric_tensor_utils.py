
import torch
import torch_geometric.nn as nn 
import torch_geometric.nn as nn_geo

#global_mean_pool,global_add_pool,global_mean_pool,global_sort_pool

def global_mean_weighted_pool(x,batch,weights,debug_nan = False):
    """
    Purpose: To do a weighted mean pooling for 
    a batch
    
    Ex: 
    from pytorch_tools import tensor_utils as tenu
    x = torch.Tensor([70,80,90,100,110,120])
    w = torch.Tensor([10,5,15,10,5,15])
    batch = torch.tensor(np.array([0,0,0,1,1,1]),dtype=torch.int64)
    tenu.global_mean_weighted_pool(x,batch,w)
    """
    weights = (weights.unsqueeze(1))
    if debug_nan:
        if tenu.isnan_any(batch):
            raise Exception(f"Nan batch")
    weight_sum = nn_geo.global_add_pool(weights,batch)
    
    
    if debug_nan:
        if tenu.isnan_any(weight_sum):
            raise Exception(f"Nan weight_sum")
    weighted_value_sum = nn_geo.global_add_pool(x*weights,batch)
    if debug_nan:
        if tenu.isnan_any(weighted_value_sum):
            raise Exception(f"Nan weighted_value_sum")
    
    weight_result = weighted_value_sum/weight_sum
    weight_result[weight_result != weight_result] = 0
    
    if debug_nan:
        if tenu.isnan_any(weight_result):
            raise Exception(f"Nan weight_result")
    return weight_result

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

def ptr_from_pool_tensor(tensor):
    """
    Purpose: to compute a new ptr tensor from
    a pool tensor
    """
    ptr = torch.cumsum(gtu.n_in_pool_from_pool_tensor(tensor),0)
    new_ptr =  torch.hstack([torch.zeros(1),ptr]).to(dtype=torch.int64)
    return new_ptr


#--- from pytorch_tools ---
from . import tensor_utils as tenu

from . import geometric_tensor_utils as gtu