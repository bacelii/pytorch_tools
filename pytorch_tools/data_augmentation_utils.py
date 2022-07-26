"""
Deepmind paper that talks about training with high confidence samples: 

https://www.nature.com/articles/s41586-021-03819-2.pdf
- noisy student self-distillation

https://github.com/rish-16/grafog
- general procedure
1) Write a class that inherits from module and overloads the forward function

2) create composition of node and edge augmentations
3) At every epoch fun the data through the augmentation

Ex: 

node_aug = T.Compose([
    T.NodeDrop(p=0.45),
    T.NodeMixUp(lamb=0.5, classes=7),
    ...
])

edge_aug = T.Compose([
    T.EdgeDrop(0=0.15),
    T.EdgeFeatureMasking()
])

data = CoraFull()
model = ...

for epoch in range(10): # begin training loop
    new_data = node_aug(data) # apply the node augmentation(s)
    new_data = edge_aug(new_data) # apply the edge augmentation(s)
    
    x, y = new_data.x, new_data.y
    ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import tensor_utils as tenu
import numpy_utils as nu

# ----------- utility functions for augmentation classes ----- 
pool_attributes_affected_by_nodes = [
    "node_weight_pool0",
    "pool1",
]

from copy import deepcopy
import tensor_utils as tenu

def set_ptr(data,return_ptr = False):
    """
    Purpose: to compute the new ptr
    if a batch is readjusted
    """
    ptr = torch.hstack([torch.tensor([0]),torch.where(data.batch[1:] - data.batch[:-1]>0)[0],torch.tensor(len(data.batch))])
    if return_ptr:
        return ptr
    else:
        data.ptr = ptr
        
        
def mask_addition_if_totally_eliminated(data,mask):
    """
    Purpose: To come up with a mask indicating
    what positions need to be fixed in order for the
    mask not to totally eliminated a batch
    """
    batch = data.batch
    
    batch_after_elim = batch[torch.logical_not(mask)]
    remaining_batches = torch.unique(batch_after_elim)
    batch_idx = np.arange(torch.max(batch)+1)
    batch_mask = torch.ones(batch_idx.shape,dtype=torch.bool)
    batch_mask[remaining_batches] = False
    mask_fix = tenu.intersect1d(batch,batch_idx[batch_mask],return_mask=True)
    
    return mask_fix

import time
import numpy as np
def drop_nodes(
    data,
    mask=None,
    p=None,
    seed=None,
    clone=True,
    node_attributes_to_adjust = None,
    verbose = False,
    debug_time = False,):
    """
    Purpose: to delete certain nodes from a dataset
    based on a mask
    
    Pseudocode: 
    1) Generate the mask if not defined
    2) Get the node_idx of the mask
    3) Generate vector of the new node ids
    
    4) Find the new edges by dropping the edges with the nodes
    and then finding the new edge
    
    Ex: 
    from torch_geometric.data import Data
    import numpy as np
    import geometric_dataset_utils as gdu

    d = dau.example_data_obj()
    new_d = dau.drop_nodes(
        d,
        p=0.5,
        verbose = True,
        #node_attributes_to_adjust=pool_attributes_affected_by_nodes
    )
    
    Ex: With a prescribed mask
    
    d = dau.example_data_obj()
    new_d = dau.drop_nodes(
        d,
        mask = torch.tensor([0,0,1,0,0,0],dtype=torch.bool),
        p=0.5,
        verbose = True,
        #node_attributes_to_adjust=pool_attributes_affected_by_nodes
    )
    """
    st = time.time()
    debug_batch = False
    debug_time = False
    if clone:
        data = deepcopy(data)
        
    #data.batch.shape
    if debug_batch:
        print(f"Beginning")
        print(f"data.x.shape = {data.x.shape}")
        print(f"data.batch.shape = {data.batch.shape}")
        print(f"data.ptr = {data.ptr}")
        
    
    if node_attributes_to_adjust is None:
        node_attributes_to_adjust = pool_attributes_affected_by_nodes.copy()
        
    
        
    node_attributes_to_adjust = nu.convert_to_array_like(node_attributes_to_adjust)
    node_attributes_to_adjust.append("batch")
        
    n_nodes = data.x.shape[0]
    if mask is None:
        mask = tenu.random_mask(n_nodes,p=p,seed=seed)
        if verbose:
            print(f"filter_away_mask = {mask}")
    # -- want to check that no neuron is completely filtered away
    # -- and if so then adds back the nodes ----
    """
    Psuedocode: 
    1) Use the mask to index to the batch to get the leftover batches
    2) Get the batches that are totally eliminated
    3) Then find the positions in the map where the eliminated masks are and turn back to true
    """
    mask[dau.mask_addition_if_totally_eliminated(data,mask)] = False
    
    
    
    # new node ids to index into
    previous_nodes = torch.arange(n_nodes)
    new_node_ids = previous_nodes - tenu.cumsum(mask)
    
    if debug_time:
        print(f"New nodes id time: {time.time() - st}")
    
    if verbose:
        print(f"new_node_ids = {new_node_ids}")
        st = time.time()
    
    # find the edges to keep:
    nodes_dropped = previous_nodes[mask]
    if verbose:
        print(f"nodes_dropped ({len(nodes_dropped)}) = {nodes_dropped}")
        
    if debug_time:
        print(f"Node dropped time: {time.time() - st}")
        st = time.time()
    
    edge_idx_keep = torch.logical_not((tenu.intersect1d(data.edge_index[0],nodes_dropped,return_mask=True)
                     | tenu.intersect1d(data.edge_index[1],nodes_dropped,return_mask=True)))
    
    if debug_time:
        print(f"Edge_idx_keep time: {time.time() - st}")
        st = time.time()
    
    if verbose:
        edges_dropped = data.edge_index[:,~edge_idx_keep].T
        print(f"edges dropped ({len(edges_dropped)})= {edges_dropped}")
        
    edges = data.edge_index[:,edge_idx_keep]
    data.edge_index = new_node_ids[edges]
    
    if debug_batch:
        print(f"Before x adjustment")
        print(f"data.x.shape = {data.x.shape}")
        print(f"data.batch.shape = {data.batch.shape}")
        print(f"data.ptr = {data.ptr}")
        
    #--fixing all the node attributes
    keep_mask = torch.logical_not(mask)
    data.x = data.x[keep_mask,:]
    #data.y = data.y[keep_mask]
    
    if debug_time:
        print(f"Setting x data: {time.time() - st}")
        st = time.time()
    
#     if verbose:
#         print(f"keep_mask = {keep_mask}")
    if debug_batch:
        print(f"After x adjustment")
        print(f"data.x.shape = {data.x.shape}")
        print(f"data.batch.shape = {data.batch.shape}")
        print(f"data.ptr = {data.ptr}")
        
    
    if node_attributes_to_adjust is not None:
        for n in node_attributes_to_adjust:
            curr_val = getattr(data,n,None)
            if curr_val is None:
                continue
            try:
                setattr(data,n,curr_val[keep_mask])
            except:
                if n == "batch":
                    pass
                else:
                    raise Exception("")
                    
            if debug_time:
                print(f"Setting attribute {n}: {time.time() - st}")
                st = time.time()
            
    # -- resolving the ptr
    dau.set_ptr(data)
    if debug_time:
        print(f"Setting pointer: {time.time() - st}")
        st = time.time()
    
    if debug_batch:
        print(f"AFter node attributes adjustment")
        print(f"data.x.shape = {data.x.shape}")
        print(f"data.batch.shape = {data.batch.shape}")
        print(f"data.ptr = {data.ptr}")
        
        
    
     
    return data

def pool_idx_stacked(
    data,
    pool_name = "pool1",
    return_n_limbs_for_neuron = False,
    return_adjustment = False):
    """
    Purpose: To generate a vector that represents the 
    mapping of nodes to unique limbs
    """
    pool1 = getattr(data,pool_name)
    n_limbs_for_neuron = pool1[data.ptr[1:]-1]
    
    if return_n_limbs_for_neuron:
        return n_limbs_for_neuron
    adjust = (tenu.cumsum(n_limbs_for_neuron)[data.batch] - n_limbs_for_neuron[0])
    if return_adjustment:
        return adjust
    else:
        return  adjust + pool1

def drop_limbs(
    data,
    mask=None,
    p = None,
    seed=None,
    clone=True,
    limb_map_attribute = "pool1",
    verbose = False,
    **kwargs
    ):
    """
    Purpose: To drop limbs randomly from a dataset

    Pseudocode: 
    1) Get a mask for the limbs
    2) Turn the mask of the limbs into a mask for the nodes
    3) Use drop_nodes function
    
    Ex: 
    new_d = dau.drop_limbs(
        data = dau.example_data_obj(),
        verbose = True,
        p = 0.3
    )
    """
    debug_verbose = False
    
    if debug_verbose:
        print(f"Beginning of drop limbs")
        print(f"data.x.shape = {data.x.shape}")
        print(f"data.batch.shape = {data.batch.shape}")
        print(f"data.ptr = {data.ptr}")
    pool1 = dau.pool_idx_stacked(data,limb_map_attribute)
    
    if debug_verbose:
        print(f"pool1.shape = {pool1.shape}")
    n_limbs = int(torch.max(pool1)+1)
    if verbose:
        print(f"n_limbs = {n_limbs}")
    
    if mask is None:
        mask = tenu.random_mask(n_limbs,p=p)
        
    limb_idx_to_drop = np.arange(n_limbs)[mask]
    node_mask = tenu.intersect1d(pool1,limb_idx_to_drop,return_mask=True)

    if debug_verbose:
        print(f"limb_idx_to_drop ({len(limb_idx_to_drop)}) = {limb_idx_to_drop}")
        print(f"node_mask = {node_mask.shape}")
        print("")
    
    return dau.drop_nodes(
        data,
        mask=node_mask,
        seed=seed,
        clone=clone,
        verbose = verbose,
    )




clone_default = False

class Compose(nn.Module):
    def __init__(self, transforms,clone=clone_default):
        super().__init__()
        self.transforms = transforms
        self.clone = clone

    def forward(self, data):
        for aug in self.transforms:
            data = aug(data)
        return data
    
class NodeFeatureNoise(nn.Module):
    def __init__(self,amplitude = 0.1,clone=clone_default,**kwargs):
        super().__init__()
        self.amplitude = amplitude
        self.clone = clone
        
    def forward(self,data):
        if self.clone:
            data = data.clone()
        data.x = tenu.add_noise(data.x,
                                amplitude = self.amplitude,
                                seed = None)
        return data
    
    
class NodeFeatureMask(nn.Module):
    def __init__(self,p = 0.2,clone=clone_default,**kwargs):
        super().__init__()
        self.p = p
        self.clone = clone
    def forward(self,data):
        if self.clone:
            data = data.clone()
        data.x = data.x[:,tenu.random_mask(data.x.shape[1],p=self.p)]
        return data
    
import geometric_dataset_utils as gdu
class NodeDrop(nn.Module):
    def __init__(self,p = 0.1,clone=clone_default,**kwargs):
        super().__init__()
        self.p = p
        self.clone = clone
    def forward(self,data,verbose = False):
        """
        Ex: 
        d = dau.example_data_obj()
        dau.NodeDrop(p = 0.5)(d,verbose =True)
        """
        if self.clone:
            data = data.clone()
        return dau.drop_nodes(
                    data,
                    p=self.p,
                    clone=False,
                    verbose = verbose,)
    
class LimbDrop(nn.Module):
    def __init__(self,p = 0.2,clone=True,**kwargs):
        super().__init__()
        self.p = p
        self.clone = clone
    def forward(self,data,verbose = False):
        """
        Ex: 
        d = dau.example_data_obj()
        dau.NodeDrop(p = 0.5)(d,verbose =True)
        """
        if self.clone:
            data = data.clone()
        return dau.drop_limbs(
                    data,
                    p=self.p,
                    clone=False,
                    verbose = verbose,)
    
def compose_augmentation(
    augmentations,
    **kwargs):
    if augmentations is None:
        augmentations = []
    augmentations = nu.convert_to_array_like(augmentations)
    aug_func = []
    for k in augmentations:
        if "str" in str(type(k)):
            aug_func.append(getattr(dau,k)(**kwargs))
        else:
            aug_func.append(k(**kwargs))
            
    return dau.Compose(aug_func)
    
    
    
import data_augmentation_utils as dau
'''
class NodeDrop(nn.Module):
    def __init__(self, p=0.05):
        super().__init__()
        self.p = p

    def forward(self, data):
        x = data.x
        y = data.y
        train_mask = data.train_mask
        test_mask = data.test_mask
        edge_idx = data.edge_index

        idx = torch.empty(x.size(0)).uniform_(0, 1)
        train_mask[torch.where(idx < self.p)] = 0
        test_mask[torch.where(idx < self.p)] = 0
        new_data = tg.data.Data(x=x, edge_index=edge_idx, y=y, train_mask=train_mask, test_mask=test_mask)

        return new_data
'''