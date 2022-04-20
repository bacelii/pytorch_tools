def layer_bias(layer):
    return layer.bias

def layer_parameters(layer):
    return list(model.lin.parameters())

def check_gradient_on_parameters_finite(model):
    for name, param in model.named_parameters():
        print(name, torch.isfinite(param.grad).all())