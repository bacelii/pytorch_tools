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
        d = gdu.example_data_obj()
        dau.NodeDrop(p = 0.5)(d,verbose =True)
        """
        if self.clone:
            data = data.clone()
        return gdu.drop_nodes(
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
        d = gdu.example_data_obj()
        dau.NodeDrop(p = 0.5)(d,verbose =True)
        """
        if self.clone:
            data = data.clone()
        return gdu.drop_limbs(
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