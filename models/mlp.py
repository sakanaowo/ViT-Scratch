import torch.nn as nn
import torch.nn.functional as F

class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        x = F.gelu(self.fc1(x), approximate="tanh")
        return self.fc2(x)
