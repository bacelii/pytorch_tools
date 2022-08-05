#torch modules
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool,global_add_pool,global_mean_pool,global_sort_pool
import numpy as np
import torch as th
import torch.nn as nn
import geometric_tensor_utils as gtu

#custom modules
import numpy_utils as nu

class GCNHierarchicalClassifier(torch.nn.Module):
    """
    Purpose: To run a GCN network that
    has a middle step that hierarchically pools
    nodes together and runs a classification on them
    and returns both the overall classification and the node classification
    """
    
    def __init__(
        self,
        dataset_num_node_features,
        dataset_num_classes,
        
        # parameters for model architecture
        n_hidden_channels=None,
        n_layers = None,
        n_hidden_channels_pool0 = None,
        n_layers_pool0 = None,
        n_hidden_channels_pool1 = None,
        n_layers_pool1 = None,
        
        #-- classier of the architecture --
        
        
        #-- number of features added on for the intermediate pooling step --
        num_node_features_pool1 = 1,
        
        #hyper-parameters for the training
        activation_function = "relu",
        use_bn = True,
        track_running_stats=True,
        
        #pooling parameters
        global_pool_type="mean",
        global_pool_weight = "node_weight",
        
        
        
        # parameters to control the flow of data through architecture
        aggregate_layer_outputs = False,
        aggregate_layer_outputs_func = "concatenate",
        residual_connections = False,
        
        # for applyin any edge weights
        edge_weight = False,
        edge_weight_name = "edge_weight",
        add_self_loops = None,
        
        # linear layer training
        dropout_p = 0.5,

        
        verbose = True,
        **kwargs
        ):
        
        super(GCNHierarchicalClassifier, self).__init__()
        self.n_pool = 2
        self.use_bn = use_bn
        self.act_func = getattr(F,activation_function)
        
        # -- for the pooling --
        self.global_pool_type = global_pool_type
        self.global_pool_func = getattr(gtu,f"global_{global_pool_type}_pool")
        self.global_pool_weight = global_pool_weight
        
        # --- control of flow parameters
        self.residual_connections = residual_connections
        self.aggregate_layer_outputs = aggregate_layer_outputs
        self.aggregate_layer_outputs_func = aggregate_layer_outputs_func
        
        
        # --- for the edge weights ---
        self.edge_weight = edge_weight
        if add_self_loops is None:
            if self.edge_weight is not None:
                self.add_self_loops = False
            else:
                self.add_self_loops = True
        else:
            self.add_self_loops = add_self_loops
            
        self.edge_weight_name = edge_weight_name
        
        self.dropout_p = dropout_p
        
        
        # --- architecture ----
        n_input_layer = dataset_num_node_features
        
        n_hidden_channels_for_aggregator = []
        for pool_idx in range(self.n_pool):
            
            suffix = f"_pool{pool_idx}"
            
            # sets up the number of hidden channels
            n_hidden_channels_pool = eval(f"n_hidden_channels{suffix}")
            if n_hidden_channels_pool is None:
                n_hidden_channels_pool = n_hidden_channels
                
            #sets up the number of pooling layers requested
            if not nu.is_array_like(n_hidden_channels_pool):
                n_layers_pool = eval(f"n_layers{suffix}")
                if n_layers_pool is None:
                    n_layers_pool = n_layers
                n_hidden_channels_pool = [n_hidden_channels_pool]*(n_layers_pool)
            else:
                n_layers_pool = len(n_hidden_channels_pool)
                
            
            n_hidden_channels_pool = np.hstack([n_input_layer,n_hidden_channels_pool])
            
            if verbose:
                print(f"Pool {pool_idx} n_hidden_channels_pool = {n_hidden_channels_pool}")
                
            # Creating the convolutional and batch layers
            for i in range(len(n_hidden_channels_pool)-1):
                n_hidden_channels_for_aggregator.append(n_hidden_channels_pool[i+1])
                
                n_input = n_hidden_channels_pool[i]
                n_output = n_hidden_channels_pool[i+1]
                
                setattr(self,f"conv{i}{suffix}",GCNConv(n_input, n_output,add_self_loops=self.add_self_loops))
                
                if use_bn: 
                    setattr(self,
                            f"bn{i}{suffix}",
                            torch.nn.BatchNorm1d(
                                n_output,
                                track_running_stats=track_running_stats
                    ))
                    
            # Register the number of layers for this pooling stage
            setattr(self,f"n_conv{suffix}",n_layers_pool)
            
            if pool_idx != self.n_pool - 1:
                n_extra_features = eval(f"num_node_features_pool{pool_idx+1}")
            else:
                n_extra_features= 0
                
                
            # ---- ********** this is changed where now feed in the number of classes instead of previous layer size ---
            
            
            # now have to do the linear layers
            if self.aggregate_layer_outputs:
                lin_n_layers = sum(n_hidden_channels_for_aggregator)
            else:
                lin_n_layers = n_hidden_channels_pool[-1]
            
            # Creating the linear classification layers
            setattr(self,f"lin{suffix}",Linear(lin_n_layers, dataset_num_classes))
            
            n_input_layer = (
                dataset_num_classes + n_extra_features
                )
    
    def encode(
        self,
        data,
        debug_encode = False,
        ):
        
        """
        Purpose: To pass dta through all of the gcn layers
        and classifier layers and return the following:
        
        Pseudocode:
        1) pool0 GCN layer
        2) Pool the nodes together into limbs
        2) Feed into classifier (save the outputs and y)
        3) concatenate x_pool1 to features
        4) pool1 GCN layer
        5) Pool the nodes together
        6) Pass through classifier
        """
        
        x, edge_index = data.x, data.edge_index
        batch = getattr(data,"batch",None)
        
        if batch is None:
            batch = torch.zeros(len(x),dtype=torch.int64)
        
        all_layer_x = []
        for pool_idx in range(self.n_pool):
            if debug_encode:
                print(f"Working on Pool {pool_idx}")
                
            suffix = f"_pool{pool_idx}"
            n_conv = getattr(self,f"n_conv{suffix}")
            
            #1) GCN layers
            for i in range(n_conv):
                if debug_encode:
                    print(f"Working on Layer {i}")
                
                if self.edge_weight:
                    edge_weight = getattr(data,f"{self.edge_weight_name}{suffix}")
                else:
                    edge_weight = None
                    
                if debug_encode:
                    print(f"edge_weight iter {i} = {edge_weight}")
                    
                if self.residual_connections:
                    x_old = x.clone()
                    
                x = getattr(self,f"conv{i}{suffix}")(x, edge_index,
                                                     edge_weight=edge_weight)
                
                if self.use_bn:
                    if debug_encode:
                        print(f"Using bn iter {i}")
                    x = getattr(self,f"bn{i}{suffix}")(x)
                if (i < n_conv-1): # and (pool_idx == self.n_pool - 1):
                    if debug_encode:
                        print(f"Using act_fun {self.act_func} {i}")
                    x = self.act_func(x)
                    
                if self.aggregate_layer_outputs:
                    all_layer_x.append(x.clone())
                    
                if self.residual_connections:
                    x = torch.mean([x,x_old])
            
                
            
            # pooling the nodes together
            weight_values = getattr(data,f"{self.global_pool_weight}_pool{pool_idx}",None)
            if weight_values is None:
                weight_values = getattr(data,f"{self.global_pool_weight}",None)
                
                
            if debug_encode:
                print(f"Right before pooling weight_values = {weight_values}")
                
            if debug_encode:
                print(f'weight_values = {weight_values}')
            
            #doing the aggregation of x's if aggregating all outputs
            if self.n_pool == pool_idx + 1:
                if debug_encode:
                    print(f"About to aggregate layers if requested")
                if self.aggregate_layer_outputs:
                    if self.aggregate_layer_outputs_func == "concatenate":
                        x = torch.hstack(all_layer_x)
                    elif self.aggregate_layer_outputs_func == "mean":
                        x = torch.sum(all_layer_x)/len(all_layer_x)
                    else:
                        raise Exception("")
                           
            # Pooling the nodes together
            next_pool = f"pool{pool_idx+1}"
            pool_vec = getattr(data,next_pool,None)
            if pool_vec is None:
                pool_vec = batch
                need_batch = False
            else:
                need_batch = True
                
            if debug_encode:
                print(f"Right before pool func: {self.global_pool_func.__name__}")
                
            x = self.global_pool_func(x, pool_vec,weights=weight_values)
            
            
            # Feed into classifier
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            
            x = getattr(self,f"lin{suffix}")(x)
            x = F.softmax(x,dim=1)
            
            if pool_idx == self.n_pool - 1:
                return x_0,x
            
            x_0 = x.clone()
            
            x_pool = getattr(data,f"x_{next_pool}",torch.Tensor([]))
            #print(f"x_pool = {x_pool}")
            x = torch.hstack([x,x_pool])
            
            #getting new edge index
            edge_index = getattr(data,f"edge_index_{next_pool}",None)
            
            batch = global_mean_pool(batch,pool_vec)
            
    def forward(self,data,**kwargs):
        return self.encode(data,**kwargs)
    
    
# ----------- For doing the forward pass ----

import torch.nn.functional as F
import torch
import evaluation_utils as evu
import numpy as np
import parameters_utils as paru

import pandas as pd
eps=1e-13

import tensor_utils as tenu
import data_augmentation_utils as dau

# -- for helping compute the weighted loss --
import geometric_tensor_utils as gtu

def hierarchical_loss(
    loss_function,
    x1,x2,
    y1,y2,
    data,
    class_weights,
    loss_pool1_weight_by_pool2_group = True,
    loss_pool1_weight = 1,
    loss_pool2_weight = 1,
    return_separate_loss = False,
    debug = False,
    ):
    
    print(f"loss_pool1_weight = {loss_pool1_weight}")
    
    loss_pool1 = loss_function(
        torch.log(x1 + eps), 
        y1,
        weight = class_weights,
        reduction = "none"
        )

    if debug:
        print(f"x1= {x1.shape}")
        print(f"y1= {y1.shape}")
    
    
    if loss_pool1_weight_by_pool2_group:
        loss_pool1 = loss_pool1 * gtu.normalize_in_pool_from_pool_tensor(
            global_mean_pool(data.batch,data.pool1)
        )
    else:
        loss_pool1 = loss_pool1 / len(y2)
        
#     if debug:
#         print(f"data.x_pool1 = {data.x_pool1}")

    loss_pool1 = loss_pool1 * loss_pool1_weight

    # need to adjust the  loss_pool1

    loss_pool2 = loss_function(
        torch.log(x2 + eps), 
        y2,
        weight = class_weights,
        reduction = "none",
    ) * loss_pool2_weight
    
    loss_pool2 = loss_pool2.mean()
    loss_pool1 = loss_pool1.mean()

    if return_separate_loss:
        return loss_pool1, loss_pool2
    else:
        return loss_pool1 + loss_pool2


def forward_pass(
    model,
    data_loader,
    loss_pool1_weight_by_pool2_group = True,
    loss_pool1_weight = 1,
    loss_pool2_weight = 1,
    n_batches_per_update = 1,
    optimizer=None,
    model_name = None,
    mode = "train",
    loss_function = "nll_loss",
    device = "cpu",
    tensor_map = None,
    class_weights = None,
    augmentations = None,
    return_predicted_labels = False,
    return_data_names = False,
    return_data_sources = False,
    features_to_return = None,
    return_dict_for_embed = False,
    return_df = False,
    debug_nan = False,
    verbose = False,
    debug = False,
    **kwargs):
    
    """
    Purpose: Want to generate y_pred/y_true or dataframes for 
    """
    
    # -- gets the actual loss function from str name---
    if type(loss_function) == str:
        loss_function = getattr(F,loss_function)
        
    # -- creates composition of augemntations
    if augmentations is not None:
        aug_compose = dau.compose_augmentation(augmentations)
        
    # Puts model into modes to compute gradients or not
    if mode == "train":
        model.train()
    else:
        model.eval()
        
    y_pred_list_1 = []
    y_true_list_1 = []
    
    embeddings_1 = []
    labels_1 = []
    
    y_pred_list_2 = []
    y_true_list_2 = []
    
    embeddings_2 = []
    labels_2 = []
    
    
    # for storing all of the attributes requested to be returned
    if features_to_return is not None:
        if "str" in str(type(features_to_return)):
            features_to_return = [features_to_return]
    else:
        features_to_return = []
        
    if return_data_names:
        features_to_return += ["data_name","name"]
        
    if return_data_sources:
        features_to_return.append("data_source")
        
        
    features_dict = {k:[] for k in features_to_return}
    
    loss = 0
    loss_1 = 0
    loss_2 = 0
    loss_for_update = []
    
    for jj,data in enumerate(data_loader):
        batch = getattr(data,"batch",None)
        if batch is None:
            batch=torch.zeros(len(data.x),dtype=torch.int64)
        
        if debug_nan or verbose:
            print(f"\n\n------ iteration {jj}/{len(data_loader)}")
        data = data.to(device)
        
        # -- Running the augmentations on the model ---
        if augmentations is not None:
            #print(f"Data before augmentation = \n\t{data.x}")
            data = aug_compose(data)
            #print(f"Data after augmentation = \n\t{data.x}\n\n\n")
            
        # -- get the output of the model
        x1,x2 = model(data)
        
        # --- getting what the y values should have been
        y2 = data.y.squeeze_()
        y1 = y2[global_mean_pool(batch,data.pool1)]
        
#         if debug:
#             print(f"y2 = {y2}")
#             print(f"y1 = {y1}")
        
        if debug_nan:
            print(f"x1= {x1}")
            if tenu.isnan_any(x1):
                raise Exception("Nan output")
            print(f"x2= {x2}")
            if tenu.isnan_any(x2):
                raise Exception("Nan output")
                
        # --- computing hte loss if in training mode --
        if mode == "train":
            
            # --- determining the 2 loss functions ----
            curr_loss_1,curr_loss_2 = hierarchical_loss(
                loss_function,
                x1,x2,
                y1,y2,
                data,
                class_weights=class_weights,
                loss_pool1_weight_by_pool2_group = loss_pool1_weight_by_pool2_group,
                loss_pool1_weight = loss_pool1_weight,
                loss_pool2_weight = loss_pool2_weight,
                return_separate_loss = True,
                debug = False,
                )
            
            curr_loss = curr_loss_1 + curr_loss_2
            
            loss_for_update.append(curr_loss)
            loss += curr_loss
            
            loss_1+= curr_loss_1
            loss_2+= curr_loss_2
            
            # -- only performing a backpropogation step ever [n_batches_per_update] batches
            if (jj % n_batches_per_update == n_batches_per_update-1):
                if debug:
                    print(f"updating")
                t_loss = sum(loss_for_update)
                t_loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad(set_to_none=True)  # Clear gradients.
                loss_for_update=[]
            else:
                if debug:
                    print(f"Not updating")
                    
            if debug_nan:
                paru.print_parameters(model)
                print(f"loss = {loss/(jj+1)}")
                if paru.isnan_in_parameters(model):
                    raise Exception("Nan parameters")
                    
        elif mode == "test":
            with torch.no_grad():
                curr_loss_1,curr_loss_2 = hierarchical_loss(
                    loss_function,
                    x1,x2,
                    y1,y2,
                    data,
                    class_weights=class_weights,
                    loss_pool1_weight_by_pool2_group = loss_pool1_weight_by_pool2_group,
                    loss_pool1_weight = loss_pool1_weight,
                    loss_pool2_weight = loss_pool2_weight,
                    return_separate_loss = True
                    )
                
                curr_loss = curr_loss_1 + curr_loss_2
            
                loss += curr_loss

                loss_1+= curr_loss_1
                loss_2+= curr_loss_2
                
                
            # computing the 
            y_pred_1 = x1.argmax(dim=1)
            y_pred_list_1.append(y_pred_1)
            y_true_list_1.append(y1)

            y_pred_2 = x2.argmax(dim=1)
            y_pred_list_2.append(y_pred_2)
            y_true_list_2.append(y2)
            
        elif mode == "embed":
            embeddings_1.append(x1.detach().cpu().numpy())
            labels_1.append(y1.numpy().reshape(-1))
            
            embeddings_2.append(x2.detach().cpu().numpy())
            labels_2.append(y2.numpy().reshape(-1))
            
                
            if features_to_return is not None:
                for f in features_to_return:
                    try:
                        features_dict[f].append(getattr(data,f))
                    except:
                        pass
        else:
            raise Exception("Unknown mode")
            
            
    # -------- Preparing the Return Value ------ 
    if mode == "train":
        return loss_1/(jj+1),loss_2/(jj+1)
    
    elif mode == "test":
        y_pred_1 = torch.cat(y_pred_list_1)
        y_true_1 = torch.cat(y_true_list_1)
        
        y_pred_2 = torch.cat(y_pred_list_2)
        y_true_2 = torch.cat(y_true_list_2)
        
        print(f"----Pool 1 metric Dict ---")
        met_dict_1 = evu.metric_dict(
                    y_true_1,
                    y_pred_1,
                    tensor_map=tensor_map,
                    metrics=["accuracy"],
                    )
        print(f"----Pool 2 metric Dict ---")
        met_dict_2 = evu.metric_dict(
                    y_true_2,
                    y_pred_2,
                    tensor_map=tensor_map,
                    metrics=["accuracy"],
                    )
        return (met_dict_1,
                met_dict_2,
                loss_1/(jj+1),loss_2/(jj+1),
               )
    elif mode == "embed":
        embeddings_1 = np.vstack(embeddings_1)
        labels_1 = np.hstack(labels_1)
        
        embeddings_2 = np.vstack(embeddings_2)
        labels_2 = np.hstack(labels_2)
        
        return_value = [np.vstack(embeddings_1),
                        np.hstack(labels_1),
                        np.vstack(embeddings_2),
                       np.hstack(labels_2)]
        return_value_names = [
            "embeddings_1",
            "labels_1",
            "embeddings_2",
            "labels_2",]
        
        if features_to_return is not None:
            for f in features_to_return:
                if len(features_dict[f]) == 0:
                    continue
                return_value.append(np.hstack(features_dict[f]))
                return_value_names.append(f)
                
        return_dict = {k:v for k,v in zip(return_value_names,return_value)}
        
        if return_df:
            df = pd.DataFrame(embeddings)
            for k,v in return_dict.items():
                if k == "embeddings":
                    continue
                df[k] = v
            return df
        if return_dict_for_embed:
            return return_dict
        else:
            return return_value
        
    else:
        raise Exception("")
        
            
            
                
        
        
import hierarchical_models as hm
            
        
            
            
        
                
                
        
        
        
        
        
    
    
    