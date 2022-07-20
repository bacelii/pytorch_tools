import torch

def map_tensor_with_tensor(tensor,tensor_map):
    """
    Purpose: Map numbers of one tensor
    to another number where the mapping is defined in the second tensort where

    index --> array_map[index]
    """
    return tensor_map[tensor]
    
def cat(tensors,dim = None,**kwargs):
    return torch.cat(tensors,dim=dim,**kwargs)

def numpy_array(tensor):
    return tensor.cpu().detach().numpy()

def vstack(tensors,**kwargs):
    """
    Purpose: To vertically stack tensors on top
    of each other
    
    Ex: 
    x = torch.Tensor([[1,2,3,4,5],[7,8,9,10,11]])
    new_values = torch.Tensor([10,20,30,40,50])
    torch.vstack([x,new_values])
    """
    return torch.vstack(tensors,**kwargs)

def isnan(tensor):
    return torch.isnan(tensor)
def isnan_any(tensor):
    return torch.any(torch.isnan(tensor))


import tensor_utils as tenu
    