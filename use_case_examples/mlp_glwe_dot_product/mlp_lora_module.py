import torch
from torch import nn
import torch.nn.functional as F
from utils_lora import compute_grad_output


class ForwardModule(nn.Module):
    def __init__(self, weight, bias=None):
        super(ForwardModule, self).__init__()
        self.weight = weight  # Assume weight is passed as a pre-initialized tensor
        self.bias = bias

    def forward(self, input):
        output = input @ self.weight.t()
        if self.bias is not None:
            return output + self.bias
            
class BackwardModule(nn.Module):
    def __init__(self, weight):
        super(BackwardModule, self).__init__()
        self.weight = weight  # This is the same weight used in ForwardModule

    def forward(self, grad_output):
        return grad_output @ self.weight

class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, forward_module, backward_module):
        ctx.backward_module = backward_module
        output = forward_module(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        backward_module = ctx.backward_module
        grad_input = backward_module.forward(grad_output)
        return grad_input, None, None  # No gradients for the modules

class CustomLinear(nn.Module):
    def __init__(self, weight, bias=None):
        super(CustomLinear, self).__init__()
        self.forward_module = ForwardModule(weight, bias=bias)
        self.backward_module = BackwardModule(weight)

    def forward(self, input):
        return CustomFunction.apply(input, self.forward_module, self.backward_module)



class LoRALayerOnly(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.1)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.1)

    def forward(self, x, fc_x):
        return fc_x + self.alpha * F.linear(F.linear(x, self.B), self.A)

class MLPWithLoRATrainingAuto(nn.Module):    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, lora_rank: int, alpha: float = 1.0, learning_rate=0.01, use_lora: bool = False, criterion=None, optimizer=None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc1_lora = LoRALayerOnly(input_size, hidden_size, lora_rank, alpha)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc2_lora = LoRALayerOnly(hidden_size, output_size, lora_rank, alpha)
        
        self.learning_rate = learning_rate
        self.optimizer_func = optimizer if optimizer is not None else torch.optim.Adam
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.calibrate = False
        
        self.toggle_lora(use_lora)


    def toggle_calibrate(self, enable: bool = True):
        self.calibrate = enable

    def inference(self, x):
        self.input = x
        self.fc1_output = self.fc1(self.input)  # server side
        
        if self.use_lora:
            self.fc1_output = self.fc1_lora(self.input, self.fc1_output)

        self.relu_output = self.relu(self.fc1_output)
        
        output = self.fc2(self.relu_output)  # server side

        if self.use_lora:
            output = self.fc2_lora(self.relu_output, output)
        
        return output
    
    def forward(self, inputs):
        # FIXME: handle multi-inputs in hybrid model
        x, y = inputs

        self.optimizer.zero_grad()
            
        # some parts on server side 
        output = self.inference(x)
        
        _, loss = compute_grad_output(output, y, criterion=self.criterion)

        if not self.calibrate:
            self.optimizer.step()
        
        return loss

    def toggle_lora(self, enable: bool = True):      
        self.use_lora = enable
        
        # Replace linear layer by custom linear layer the first time we enable lora
        if enable and not isinstance(self.fc2, CustomLinear):
            self.fc2 = CustomLinear(self.fc2.weight, bias=self.fc2.bias)
 
        for module in self.modules():
            if isinstance(module, LoRALayerOnly):
                module.A.requires_grad = enable
                module.B.requires_grad = enable

            elif isinstance(module, nn.Linear):
                module.weight.requires_grad = not enable  # Freeze original weights
                module.bias.requires_grad = not enable  # Freeze original weights

        self.optimizer = self.optimizer_func(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
