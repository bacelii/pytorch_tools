import torch
from collections import OrderedDict

def layer_bias(layer):
    return layer.bias

def layer_parameters(layer):
    return list(layer.lin.parameters())

def check_gradient_on_parameters_finite(model):
    for name, param in model.named_parameters():
        print(name, torch.isfinite(param.grad).all())
        
def state_dict(model):
    return model.state_dict()
def load_state_dict(model,state_dict):
    model.load_state_dict(state_dict)
    
def copy_model_parameters_to_model(model_source,model_target):
    model_target.load_state_dict(model_source.state_dict())
    
def set_geometric_parameters(
    model,
    layer_name=None,
    verbose = False,
    strict = False,
    **kwargs
    ):
    
    curr_dict = {}
    
    if layer_name is None:
        layer_name = ""
    else:
        layer_name = f"{layer_name}."
    
    for name,value in kwargs.items():
        name = name.replace("_",".")
        long_name = f"{layer_name}{name}"
        if "str" in str(type(value)):
            if value == "eye":
                value = torch.eye(len(eval(f"model.{long_name}")))
            else:
                value = getattr(torch,value)(eval(f"model.{long_name}.shape"))
        curr_dict[long_name] = value
    
    if verbose:
        print(f"curr_dict = {curr_dict}")
                                         
    new_state_dict = OrderedDict(curr_dict)
    model.load_state_dict(new_state_dict,strict = strict,)
    
    
def parameters(
    model,
    verbose = False,
    print_extrema = False,):
    """
    Purpose: To print the out the parameters and their values
    """
    model
    
    params = list(model.named_parameters())
    if verbose:
        for name,param in params:
            if print_extrema:
                print(f"{name}: min = {torch.min(torch.abs(param))}, max = {torch.max(torch.abs(param))}")
            else:
                print(f"{name}: {param}")
    return params

def print_parameters(
    model,
    print_extrema = False):
    
    parameters(
        model,
        verbose = True,
        print_extrema = print_extrema,)
    
import tensor_utils as tsu
def isnan_in_parameters(model,verbose = False):
    return_value = False
    for name,param in model.named_parameters():
        if tsu.isnan_any(param):
            return_value = True
            if verbose:
                print(f"{name} nad nan values")

    return return_value

def add_noise_to_parameters(model,noise_amplitude = 0.1 ):
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn(param.size()) * noise_amplitude)