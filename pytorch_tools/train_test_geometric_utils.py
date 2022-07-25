"""
Purpose: To store functions that help with training and testing

"""

import torch.nn.functional as F
import torch
import evaluation_utils as evu
import numpy as np
import parameters_utils as paru

import pandas as pd


eps=1e-13


import tensor_utils as tsu
import data_augmentation_utils as dau

def forward_pass(
    model,
    data_loader,
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
    verbose = False,):
    
    debug = False
    
    if type(loss_function) == str:
        loss_function = getattr(F,loss_function)
        
    if augmentations is not None:
        aug_compose = dau.compose_augmentation(augmentations)
    
    if mode == "train":
        model.train()
    else:
        model.eval()
        
    # for testing
    y_pred_list = []
    y_true_list = []
    
    # for training
    embeddings = []
    labels = []
    data_names = []
    data_sources = []
    
    
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
    loss_for_update = []
    for jj,data in enumerate(data_loader):#train_loader:  # Iterate in batches over the training dataset.
        
        if debug_nan or verbose:
            print(f"\n\n------ iteration {jj}/{len(data_loader)}")
        data = data.to(device)
        
        
        if augmentations is not None:
            #print(f"Data before augmentation = \n\t{data.x}")
            data = aug_compose(data)
            #print(f"Data after augmentation = \n\t{data.x}\n\n\n")
            
        if model_name is not None:
            if "DiffPool" in model_name:
                out,gnn_loss, cluster_loss = model(data)  # Perform a single forward pass.
                #y_true = data.y.reshape(-1,3)
            elif model_name == "TreeLSTM":
                n = data.x.shape[0]
                h = torch.zeros((n, architecture_kwargs["n_hidden_channels"]))
                c = torch.zeros((n, architecture_kwargs["n_hidden_channels"]))
                out = model(
                    data,
                    h = h,
                    c = c,
                    embeddings = data.x
                    )
            else:
                out = model(data)
        else:
            out = model(data)
        y_true = data.y.squeeze_()
        #print(f"out.shape = {out.shape}, data.y.shape = {data.y.shape}")
        
        #print(f"out.min() = {out.min()}, out.max() = {out.max()}")
        if debug_nan:
            print(f"out= {out}")
            if tsu.isnan_any(out):
                raise Exception("Nan output")
                
             
        
        if mode == "train":
            curr_loss = loss_function(
                torch.log(out + eps), 
                y_true,
                weight = class_weights,
                )  # Compute the loss.
            
            loss_for_update.append(curr_loss)
            loss += curr_loss
            
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
                curr_loss = loss_function(
                    torch.log(out + eps), 
                    y_true,
                    weight = class_weights,
                    )  # Compute the loss.
                loss += curr_loss
            y_pred = out.argmax(dim=1)  # Use the class with highest probability.
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)
            
        elif mode == "embed":
            out_array = out.detach().cpu().numpy()
            out_labels = data.y.numpy().reshape(-1)
            
            embeddings.append(out_array)
            labels.append(out_labels)
#             if return_data_names:
#                 data_names.append(data.data_name)
#             if return_data_sources:
#                 data_sources.append(data.data_source)
                
            if features_to_return is not None:
                for f in features_to_return:
                    try:
                        features_dict[f].append(getattr(data,f))
                    except:
                        pass
        else:
            raise Exception("Unknown mode")
            
        
            
    
    if mode == "train":
        return loss/(jj+1)
    elif mode == "test":
        y_pred = torch.cat(y_pred_list)
        y_true = torch.cat(y_true_list)

        return evu.metric_dict(
            y_true,
            y_pred,
            tensor_map=tensor_map,
            metrics=["accuracy"],
        ),loss/(jj+1)
    elif mode == "embed":
        embeddings = np.vstack(embeddings)
        labels = np.hstack(labels)
        
        return_value = [embeddings,labels]
        return_value_names = ["embeddings","labels"]
        if return_predicted_labels:
            return_value.append(np.argmax(embeddings,axis=1))
            return_value_names.append("predicted_labels")
#         if return_data_names:
#             data_names = np.hstack(data_names)
#             return_value.append(data_names)
#             return_value_names.append("data_names")
        
#         if return_data_sources:
#             data_sources = np.hstack(data_sources)
#             return_value.append(data_sources)
#             return_value_names.append("data_sources")
            
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
        
        
        
import general_utils as gu
import pandas_utils as pu
def forward_pass_embed_df(
    #arguments for the forward pass
    model,
    data_loader,
    
    return_predicted_labels = True,
    return_data_names=True,
    return_data_sources = True,
    
    #for the decoding
    decoder_map = None,
    encoder_map = None,

    ):
    """
    Purpose: To return a dataframe with the embeddings
    and the names and soruce if specified
    """
    
    return_dict = ttu.forward_pass(
        model,
        data_loader=data_loader,
        mode = "embed",
        return_dict_for_embed = True,
        return_predicted_labels = return_predicted_labels,
        return_data_names=return_data_names,
        return_data_sources = return_data_sources,
    )
    
    embedding_df = pd.DataFrame(return_dict["embeddings"])
    
    for k,v in return_dict.items():
        if k == "embeddings":
            continue
        embedding_df[k] = v
        
    if decoder_map is None:
        decoder_map = dict([(v,k) if k is not None else (v,"Unknown") for k,v in encoder_map.items()])
    
    for l in ["labels","predicted_labels"]:
        if l in embedding_df.columns:
            embedding_df[l] = pu.new_column_from_dict_mapping(
                embedding_df,
                decoder_map,
                column_name = l
            )
            
    return embedding_df

import train_test_geometric_utils as ttu
    
