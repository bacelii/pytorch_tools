import numpy as np
import torch
def train_val_test_split(
    dataset,
    test_size = 0.2,
    val_size = 0.2,
    verbose = False,
    return_dict = False
    ):
    
    total_num = len(dataset)

    data_lengths = []
    data_names = []
    for d_name,d in zip(
        ["test","validation"],
        [test_size,val_size]):
        if d is not None:
            d_size = np.floor(total_num*d)
            if verbose:
                print(f"{d_name} size = {d_size} ({d} %)")
            data_lengths.append(d_size)
            data_names.append(d_name)
        else:
            if verbose:
                print(f"{d_name} was None")
                
    train_size = total_num - np.sum(data_lengths)
    
    if verbose:
        print(f"train_size = {train_size}")
    data_lengths_with_train = [train_size] + data_lengths
    data_lengths_with_train = np.array(data_lengths_with_train).astype('int')
    data_names = ["train"] + data_names
    if verbose:
        print(f"data_lengths_with_train = {data_lengths_with_train}")
    
    
    dataset_splits = torch.utils.data.random_split(dataset,data_lengths_with_train)
    
    if return_dict:
        return {f"{k}_dataset":v for k,v in zip(data_names,dataset_splits)}
    else:
        return dataset_splits