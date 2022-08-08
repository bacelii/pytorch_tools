from torch_geometric.utils import add_remaining_self_loops
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor

add_remaining_self_loops
from torch_geometric.nn import GCNConv as gc
from typing import Union
#from types import NoneType

import copy

import torch_sparse

class GCNConv(gc):
    def __init__(self,in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):
        
        super().__init__(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    improved=improved,
                    cached=cached,
                    add_self_loops=add_self_loops,
                    normalize=normalize,
                    bias=bias,
                    **kwargs)
        #print(f"self.add_self_loops = {self.add_self_loops}")
        #print(f"hi")
    
    def forward(
            self,
            x:torch.Tensor,
            edge_index:Union[torch.Tensor, torch_sparse.tensor.SparseTensor],
            edge_weight:Union[torch.Tensor, type(None)]=None,**kwargs
        ) -> torch.Tensor:
        
        if self.add_self_loops:
            old_n_edges = copy.deepcopy(len(edge_index))
            edge_index,edge_weight = add_remaining_self_loops(
                edge_index, 
                edge_attr = None,
                num_nodes = len(x))
            
            #print(f"Edge change = {len(edge_index) - old_n_edges}")
            #print(f"edge_index[-10:] = {edge_index.T[-10:] }")
        
        return super().forward(x,edge_index,edge_weight,**kwargs)
    