import torch
from torch import nn
from linear_layer import Linear
import torch.nn.functional as F

class output(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(output, self).__init__()
        self.fc= Linear(in_dim, out_dim)
    def forward(self, inputs):
        out = self.fc(inputs)
        out = F.relu(out)
        return out