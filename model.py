import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_length, output_length, time_range):
        super(Model, self).__init__()

        self.input_length = input_length - time_range + 1
        self.output_length = output_length
        self.time_range = time_range
        self.Linear = nn.Linear(self.input_length, self.output_length, bias = True)
    
    def forward(self, x):
        x = self.Linear(x)
        return x

if __name__ == '__main__':
    model = Model(4, 3, 3)

    inputs = torch.rand(2)
    print(model(inputs).shape)