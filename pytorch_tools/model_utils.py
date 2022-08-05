import torch


model_save_name = "model_state_dict"
optimizer_save_name = "optimizer_state_dict"
def save_checkpoint(
    model,
    filepath,
    optimizer = None,
    epoch = None,
    loss = None,
    **kwargs):
    
    save_dict = dict()
    save_dict[model_save_name] = model.state_dict()
    
    
    if optimizer is not None:
        save_dict[optimizer_save_name] = optimizer.state_dict()
    if epoch is not None:
        save_dict["epoch"] = epoch
    if loss is not None:
        save_dict["loss"] = loss
        
    for k,v in kwargs.items():
        save_dict[k] = v
        
    torch.save(save_dict,filepath)
        
def load_checkpoint_to_model(filepath,model):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint[model_save_name])
    
    return model
    

import model_utils as mdlu