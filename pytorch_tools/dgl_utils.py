
import dgl
import networkx as nx
import numpy as np

def g_from_data(data,binary_tree = True):
    """
    To create a dgl graph with the correct number of nodes
    and edges (assume DiGraph)
    """
    G = nx.DiGraph()
    G.add_edges_from(data.edge_index.numpy().T,)
    G.add_nodes_from(np.arange(data.x.shape[0]))
    g = dglu.from_networkx(
        G,
        binary_tree=binary_tree)
    return g

def from_networkx(
    G,
    node_attrs=None, 
    edge_attrs=None, 
    edge_id_attr_name=None,
    idtype=None, 
    device=None,
    binary_tree = True):
    
    """
    Documenation: https://docs.dgl.ai/en/0.6.x/generated/dgl.from_networkx.html
    
    Ex: 
    
    segment_id = 31495760671
    split_index = 0
    G = hdju.graph_obj_from_proof_stage(segment_id=segment_id,split_index = split_index)
    
    import dgl
    g = dgl.from_networkx(
        G, 
        node_attrs=['n_synapses', 'n_synapses_post',"n_synapses_pre"], 
        #device = 
    )
    """
    if binary_tree:
        G = xu.binary_tree_from_di_tree(G)
        
    new_G =  dgl.from_networkx(
        G,
        node_attrs=node_attrs, 
        edge_attrs=edge_attrs, 
        edge_id_attr_name=edge_id_attr_name,
        idtype=idtype, 
        device=device)

        
    return new_G

    
#--- from datasci_tools ---
from datasci_tools import networkx_utils as xu

from . import dgl_utils as dglu