def layer_bias(layer):
    return layer.bias

def layer_parameters(layer):
    return list(model.lin.parameters())

def check_gradient_on_parameters_finite(model):
    for name, param in model.named_parameters():
        print(name, torch.isfinite(param.grad).all())
        
def state_dict(model):
    return model.state_dict()
def load_state_dict(model,state_dict):
    model.load_state_dict(state_dict)
    
def copy_model_parameters_to_model(model_source,model_target):
    model_target.load_state_dict(model_source.state_dict())