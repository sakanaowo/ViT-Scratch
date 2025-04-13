import torch
from torch import nn

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = nn.Conv2d(config.num_channels, config.hidden_size,
                                         kernel_size=config.patch_size, stride=config.patch_size)
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.Embedding(num_patches, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(num_patches).unsqueeze(0), persistent=False)

    def forward(self, x):
        x = self.patch_embedding(x)            # (B, C, H, W)
        x = x.flatten(2).transpose(1, 2)       # (B, N, C)
        x = x + self.position_embedding(self.position_ids)
        return x
