
import torch

def set_printoptions(
    precision=3,
    threshold=None,
    edgeitems=None, 
    linewidth=None, 
    profile=None, 
    sci_mode=False,
    **kwargs):
    
    torch.set_printoptions(
    precision=precision,
    threshold=threshold,
    edgeitems=edgeitems, 
    linewidth=linewidth, 
    profile=profile, 
    sci_mode=sci_mode)
    
    
