'''
# how to run tensorboard inside a docker container
%load_ext tensorboard
%tensorboard --logdir /neuron_mesh_tools/Auto_Proofreading/Minnie65_Analysis/GNN_Classification/GNN_Models/runs --bind_all


'''
from pathlib import Path
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import os
import pandas as pd

#from tensorflow.python.summary.summary_iterator import summary_iterator

def df_tensorboard(
    root_dir=None,
    filepaths = None,
    sort_by=None,
    verbose = False,):
    """Convert local TensorBoard data into Pandas DataFrame.
    
    Function takes the root directory path and recursively parses
    all events data.    
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.
    
    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.
    
    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.
    
    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.
    
    source: https://laszukdawid.com/blog/2021/01/26/parsing-tensorboard-data-locally/
    
    """
    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].tensor.float_val[0]),
        )

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in EventFileLoader(str(filepath)).Load() if len(e.summary.value)
        ])

    
    
    columns_order = ["run","file_name",'wall_time', 'name', 'step', 'value']
    
    if filepaths is None:
        filepaths = []
        for (root, _, filenames) in os.walk(root_dir):
            for filename in filenames:
                if "events.out.tfevents" not in filename:
                    continue
                file_full_path = os.path.join(root, filename)
                filepaths.append(file_full_path)
                
    out = []
    for f in filepaths:
        if verbose:
            print(f"---working on {f}---")
        f = Path(f)
        curr_df = convert_tfevent(f)
        if len(curr_df) > 0:
            curr_df["run"] = f.parents[0].name
            curr_df["file_name"] = f.name
            out.append(curr_df)

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)
        
    return all_df.reset_index(drop=True)



from . import tensorboard_utils as tbu