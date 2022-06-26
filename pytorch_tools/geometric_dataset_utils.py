import numpy as np
import pandas_utils as pu
import pandas as pd

from pathlib import Path
import time

#python_tools modules
import system_utils as su
import pandas_utils as pu
import pandas as pd
import numpy as np
import numpy_utils as nu
import networkx_utils as xu
from tqdm_utils import tqdm

#neuron_morphology_tools modules
import neuron_nx_io as nxio

#pytorch and pytorch geometric modeuls
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
from torch_geometric import transforms

# for the dataset object
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
from torch_geometric.data import DenseDataLoader

#pytorch_tools modules
import preprocessing_utils as pret


def normalization_df(
    data_df,
    data_extraction_func,
    column = "data",
    verbose = False,
    **kwargs
    ):
    st = time.time()
    if verbose:
        print(f"Started calculating normalization")
    all_batch_df_flat = [data_extraction_func(
        k,
        **kwargs) if len(k) > 0
        else pd.DataFrame() for k in data_df[column].to_list()]
    
    all_batch_df = pd.concat(all_batch_df_flat)

#     if label_name is not None:
#         all_batch_df = all_batch_df[[k for k in 
#                 all_batch_df.columns if k not in nu.convert_to_array_like(label_name)]]
#     else:
#         all_batch_df = all_batch_df

    # will use these to normalize the data
    col_means = all_batch_df.mean(axis=0).to_numpy()
    col_stds = all_batch_df.std(axis=0).to_numpy()
    df_standardization = pd.DataFrame(np.array([col_means,col_stds]),
         index=["norm_mean","norm_std"],
        columns=all_batch_df.columns)

    if verbose:
        print(f"Finished calculating normalization: {time.time() - st}")

    max_nodes = np.max(all_batch_df.index.to_numpy()) + 1
    if verbose:
        print(f"max_nodes = {max_nodes}")
    
    return df_standardization

def load_data(
    data_column,
    graph_label,
    data_extraction_func,
    gnn_task_name = "gnn_task",
    output_folder = "./",
    
    # -- dataset arguments ---
    data_df = None,
    data_filepath = None,

    
    #for the standardization
    df_standardization = None,
    
    
    # for the mapping of the labels to integers
    label_int_map = None,
    
    # For prepping dataset
    dense_adj = False,
    directed = False,
    features_to_delete = None,
    features_to_keep = None,
    processed_data_folder_name = None,
    max_nodes = 300,
    
    #--------- processing the dataset ----
    clean_prior_dataset = False,
    
    # for appending 
    data_source = None,
    
    
    verbose = True,
    
    only_process_labeled = False,
    return_label_int_map = False,
    ):
    
    """
    Purpose: Will load the data for processing using the GNN models
    
    """
    output_folder = Path(output_folder)
    try:
        output_folder.mkdir(exist_ok=True) 
    except:
        pass
    
    if processed_data_folder_name is None:
        if dense_adj:
            processed_data_folder = output_folder / Path(f"{gnn_task_name}")#_processed_dense")
        elif directed:
            processed_data_folder = output_folder / Path(f"{gnn_task_name}_directed")#_processed_dense")
        else:
            processed_data_folder = output_folder / Path(f"{gnn_task_name}_no_dense")#_processed_dense")
    else:
        processed_data_folder = output_folder / Path(f"{processed_data_folder_name}")

        
    #1) Load the data
    if verbose:
        print(f"Starting to load data")
        st = time.time()
        
    if data_df is None:
        data_filepath = Path(data_filepath)
        data_df = su.decompress_pickle(data_filepath)
    
    if verbose:
        print(f"Finished loading data: {time.time() - st}")
    
    #2) Getting the means and standard deviations if not already computed
    if df_standardization is None:
        df_standardization = gdu.normalization_df(
            data_df=data_df,
            column = data_column,
            verbose = verbose,
            data_extraction_func=data_extraction_func,
            **kwargs
            )
    try:
        col_means = df_standardization.loc["norm_mean",:].to_numpy()
    except:
        col_means = df_standardization.iloc[0,:].to_numpy()
    
    try:
        col_stds = df_standardization.loc["norm_std",:].to_numpy()
    except:
        col_stds = df_standardization.iloc[1,:].to_numpy()
        
    print(f"max_nodes = {max_nodes}")

    
    #3) Creating the Dataclass
    if label_int_map is None:
        total_labels,label_counts = np.unique((data_df.query(f"{graph_label}=={graph_label}")[
        graph_label]).to_numpy(),return_counts = True)
        label_int_map = {k:i for i,k in enumerate(total_labels)}
    else:
        print(f"Using precomputed cell map")
    
    
    # ---------- Creating the dataset --------------------
    
    # --------- Functions for loading custom dataset -----
    def pytorch_data_from_gnn_info(
        gnn_info,
        y = None,
        verbose = False,
        normalize = True,
        features_to_delete=None,
        features_to_keep = None,
        data_name = None,
        data_source = None,
        default_y = -1
        ): 
        """
        Purpose: To convert our data format into pytorch Data object

        Pseudocode: 
        1) Create the edgelist (turn into tensor)
        2) Get the 
        """
        edgelist = torch.tensor(xu.edgelist_from_adjacency_matrix(
            array = gnn_info["adjacency"],
            verbose = False,
        ).T,dtype=torch.long)

        x,y_raw = data_extraction_func(
            gnn_info,
            #return_data_labels_split = True
            )
        if y is None:
            y = y_raw

        if not type(y) == str:
            y = None

        if y is None:
            y_int = np.array(default_y ).reshape(1,-1)
        else:
            y_int = np.array(label_int_map[y] ).reshape(1,-1)

        if normalize:
            x = (x-col_means)/col_stds

        # --- keeping or not keeping sertain features
        gnn_features = gnn_info["features"]

        keep_idx = np.arange(len(gnn_features))
        if features_to_delete is not None:
            curr_idx = np.array([i for i,k in enumerate(gnn_features)
                           if k not in features_to_delete])
            keep_idx = np.intersect1d(keep_idx,curr_idx)
            if verbose:
                print(f"keep_idx AFTER DELETE= {keep_idx}")
        if features_to_keep is not None:
            curr_idx = np.array([i for i,k in enumerate(gnn_features)
                           if k in features_to_keep])
            keep_idx = np.intersect1d(keep_idx,curr_idx)
            if verbose:
                print(f"keep_idx AFTER KEEP = {keep_idx}")

        x = x[:,keep_idx]

        x = torch.tensor(x,dtype=torch.float)
        y = torch.tensor(y_int,dtype=torch.long)

        if len(y) > 1:
            raise Exception(f"y = {y}")

        if y.shape[0] != 1 or y.shape[1] != 1:
            raise Exception(f"y = {y}")


        if verbose:
            print(f"x.shape = {x.shape},y.shape ={y.shape}")
        
        data_dict = dict(x=x,y=y,edge_index=edgelist)
        if data_name is not None:
            data_dict["data_name"] = data_name
            
        if data_source is not None:
            data_dict["data_source"] = data_source
            
        
        data = Data(**data_dict)
        
        return data

    class CellTypeDataset(InMemoryDataset):
        def __init__(self, root,
                     transform=None,
                     pre_transform=None, 
                     pre_filter=None,
                    only_process_labeled = False,):
            self.only_process_labeled = only_process_labeled
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
            

#         @property
#         def raw_file_names(self):
#             #return ['some_file_1', 'some_file_2', ...]
#             return [str(data_filepath.absolute())]

        @property
        def processed_file_names(self):
            return ['data.pt']

        # def download(self):
        #     # Download to `self.raw_dir`.
        #     download_url(url, self.raw_dir)
        #     ...

        def process(self,):
            # Read data into huge `Data` list.

            data_list = []
            for k,y,segment_id,split_index in tqdm(zip(
                data_df[data_column].to_list(),
                data_df[graph_label].to_list(),
                data_df["segment_id"],
                data_df["split_index"])):
                
                if y is None and self.only_process_labeled:
                    #print(f"Skipping becuase unlabeled")
                    continue
                
                if len(k) > 0:
                    data_list.append(pytorch_data_from_gnn_info(
                        k,
                        y=y,
                        features_to_delete=features_to_delete,
                        features_to_keep = features_to_keep,
                        data_name = f"{segment_id}_{split_index}",
                        data_source = data_source,
                        verbose = False))

            if self.pre_filter is not None:
                data_list_final = []
                for data in data_list:
                    try:
                        if self.pre_filter(data):
                            data_list_final.append(data)
                    except:
                        continue

                data_list = data_list_final

            for j,d in enumerate(data_list):
                if d.y.shape[0] != 1 or d.y.shape[1] != 1:
                    raise Exception(f"{j}")

            if self.pre_transform is not None:
                data_list_final = []
                for j,data in enumerate(data_list):
                    try:
                        curr_t = self.pre_transform(data)
                        if curr_t.y.shape[0] != 1 or curr_t.y.shape[1] != 1:
                            raise Exception(f"{j}, data = {curr_t}")
                        data_list_final.append(curr_t)
                    except:
                        continue
                data_list = data_list_final

            for j,d in enumerate(data_list):
                if d.y.shape[0] != 1 or d.y.shape[1] != 1:
                    raise Exception(f"{j}, data = {d}")

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            
    # --- creating the folder for the dataset --
    if clean_prior_dataset:
        try:
            su.rm_dir(processed_data_folder)
        except:
            pass
        
    processed_data_folder.mkdir(exist_ok = True)
    
    
    # a) Processing Filteres
    class MyFilter(object):
        def __call__(self, data):
            return data.num_nodes <= max_nodes

    if dense_adj:
        #gets the maximum number of nodes in any of the graphs
        transform_list = [
            transforms.ToUndirected(),
            T.ToDense(max_nodes),
            #transforms.NormalizeFeatures(),
            ]
        pre_filter = MyFilter()
    elif directed:
        transform_list = []
        pre_filter = None
    else:
        transform_list = [
            transforms.ToUndirected(),]

        pre_filter = None


    transform_norm = transforms.Compose(transform_list)
    
    
    # b) Creating the Dataset
    dataset = CellTypeDataset(
            processed_data_folder.absolute(),
            pre_transform = transform_norm,
            pre_filter = pre_filter,
            only_process_labeled = only_process_labeled,
            )
    
    if return_label_int_map:
        return dataset,label_int_map
    else:
        return dataset

def closest_neighbors_in_embedding_df(
    df,
    data_name,
    n_neighbors = 5,
    embedding_columns = None,
    verbose = False,
    return_data_names=False,
    
    plot = False,
    data_fetcher = None,
    plotting_func_name = "plot_proofread_neuron",
    plotting_func = None,
    **kwargs
    ):
    """
    Purpose: To find the n_neighbors
    closest to a certain embedding

    Pseudocode:

    1) Extract the matrix
    
    
    Ex: 
    from dataInterfaceMinnie65 import data_interface as hdju_m65
    closest_neighbors_in_embedding_df(
        df = data_source_df,
        data_name = "864691134884769786_0",
        n_neighbors = 5,
        verbose = True,
        data_fetcher = hdju_m65,
        plot = True
        )
    """
    df = df.reset_index(drop=True)
    
    if embedding_columns is None:
        embedding_columns = [k for k in df.columns if "int" in str(type(k))]
        if verbose:
            print(f"embedding_columns determined as {embedding_columns}")
    


    n_neighbors_search = n_neighbors + 1

    node_df = df.query(f"data_name=='{data_name}'")
    if verbose:
        print(f"node_df data = {pu.df_to_dicts(node_df[['cell_type','cell_type_predicted']])[0]}")

    node_data = node_df[embedding_columns].to_numpy()
    X_extract = df[embedding_columns].to_numpy()
    dist = np.linalg.norm(X_extract - node_data,axis=1)
    closest_neighbors_idx = np.argsort(dist)[:n_neighbors_search]
    if verbose:
        print(f"closest_neighbors_idx = {closest_neighbors_idx}")

    neighbors_df = df.loc[list(closest_neighbors_idx),:].query(f"data_name != '{data_name}'")
    neighbors_dicts = pu.df_to_dicts(neighbors_df[['data_name','cell_type','cell_type_predicted']])

    if verbose:
        print(f"Closest Neighbors: ")
        for k in neighbors_dicts:
            print(f"{k}")
            
    if plot:
        
        if plotting_func is None:
            if data_fetcher is None:
                raise Exception("")
            plotting_func = getattr(data_fetcher,plotting_func_name)
        
        
            
        
        print(f"\n\n--- Plotting -----")
        print(f"Original data: {data_name}")
        plotting_func(data_name,**kwargs)
        
        
        for i,k in enumerate(neighbors_dicts):
            print(f" \n\n   -->  Neighbor {i}:{k}")
            plotting_func(k["data_name"],**kwargs)

    if return_data_names:
        return_value = np.array([k["data_name"] for k in neighbors_dicts])
    else:
        return_value = neighbors_dicts
        
    return return_value


import geometric_dataset_utils as gdu