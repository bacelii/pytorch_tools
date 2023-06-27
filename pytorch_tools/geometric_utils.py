from typing import Optional, Union, List
import torch_geometric.utils as geou

def from_networkx(G, group_node_attrs: Optional[Union[List[str], all]] = None,
                  group_edge_attrs: Optional[Union[List[str], all]] = None):
    return geou.convert.from_networkx(G,group_node_attrs,group_edge_attrs)
