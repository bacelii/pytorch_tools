def map_tensor_with_tensor(tensor,tensor_map):
    """
    Purpose: Map numbers of one tensor
    to another number where the mapping is defined in the second tensort where

    index --> array_map[index]
    """
    return tensor_map[tensor]
    
def cat(tensors,dim = None,**kwargs):
    return torch.cat(tensors,dim=dim,**kwargs)