from dataclasses import dataclass

@dataclass
class SiglipVisionConfig:
    num_hidden_layers: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    num_attention_heads: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
