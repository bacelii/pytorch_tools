
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


def generator_from_seed(seed):
    if seed is None:
        return None
    return torch.random.manual_seed(seed)

def random_gaussian(shape,amplitude = 0.1,seed = None,):
    generator  = tenu.generator_from_seed(seed)
    return (torch.randn(shape,generator=generator) * amplitude)

def add_noise(tensor,amplitude = 0.1,seed = None,):
    return tensor + tenu.random_gaussian(tensor.size(),amplitude = amplitude,seed = seed,)


def random_sample_between_interval(shape,min=0,max=1,seed=None):
    if type(shape) == int:
        shape = (shape,)
    return torch.empty(shape, dtype=torch.float32).uniform_(min,max ,generator = tenu.generator_from_seed(seed))

def random_mask(shape,p=0.5,seed=None):
    if type(shape) == int:
        shape = (shape,)
    #print(f"shape ={shape}")
    return tenu.random_sample_between_interval(shape,min=0,max=1,seed=seed) < p

def cumsum(tensor,dim=0):
    return torch.cumsum(tensor,dim=dim)

# idx = torch.empty((d,), dtype=torch.float32).uniform_(0, 1) < self.p
#         x = x.clone()
#         x[:, idx] = 0

def intersect1d(tensor1,tensor2,return_mask=False,):
    """
    Purpose: To find the intersection between 2 tensors
    
    Ex: 
    a = torch.tensor([1,2,3,4,],dtype=torch.int)
    b = torch.tensor([4,5,],dtype=torch.int)

    tenu.intersect1d(a,b,return_mask=True)
    """
    indices = torch.zeros_like(tensor1, dtype = torch.bool,)
    for elem in tensor2:
        indices = indices | (tensor1 == elem)  
    
    if return_mask:
        return indices
    else:
        return tensor1[indices]  

    

from . import tensor_utils as tenu