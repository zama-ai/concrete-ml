import torch
from torch import nn


class ForwardModule(nn.Module):
    def __init__(self, weight, bias=None):
        super(ForwardModule, self).__init__()
        self.weight = weight  # Assume weight is passed as a pre-initialized tensor
        self.bias = bias

    def forward(self, input):
        output = input @ self.weight
        if self.bias is not None:
            return output + self.bias
            
class BackwardModule(nn.Module):
    def __init__(self, weight):
        super(BackwardModule, self).__init__()
        self.weight = weight  # This is the same weight used in ForwardModule

    def forward(self, grad_output):
        return grad_output @ self.weight.t()


class ForwardBackwardModule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, forward_module, backward_module):
        ctx.backward_module = backward_module
        output = forward_module.forward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        backward_module = ctx.backward_module
        grad_input = backward_module.forward(grad_output)
        
        # grad_weight and grad_bias are not needed when computing the backward for lora
        return grad_input, None, None 

class CustomConv1D(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.forward_module = ForwardModule(weight, bias=bias)
        self.backward_module = BackwardModule(weight)

    def forward(self, input):
        return ForwardBackwardModule.apply(input, self.forward_module, self.backward_module)
