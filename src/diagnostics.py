import torch.nn as nn


# L2 norm of the weight gradient of each Linear layer, ordered from input to output
# call after loss.backward(), returns one float per layer
def layer_grad_norms(model):
    return [
        layer.weight.grad.norm().item()
        for layer in model.modules()
        if isinstance(layer, nn.Linear)
    ]
