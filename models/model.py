import torch.nn as nn
from .embeddings import SiglipVisionEmbeddings
from .encoder import SiglipEncoder


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        return self.norm(x)


class SiglipVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, x):
        return self.vision_model(x)

