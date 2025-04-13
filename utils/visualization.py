import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_patch_embeddings(embeddings: torch.Tensor, title="Patch Embeddings"):
    """
    embeddings: Tensor of shape (1, num_patches, embed_dim)
    """
    data = embeddings[0].detach().cpu().numpy()  # shape: [196, 768]
    plt.figure(figsize=(15, 8))
    plt.imshow(data, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Patch Index")
    plt.show()

def visualize_attention_map(attn_weights: torch.Tensor, head: int = 0):
    """
    attn_weights: Tensor of shape (B, num_heads, T, T)
    """
    if attn_weights.dim() == 4:
        attn = attn_weights[0, head].detach().cpu().numpy()
    else:
        attn = attn_weights.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(attn, cmap="hot")
    plt.title(f"Attention Weights (Head {head})")
    plt.colorbar()
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.show()
