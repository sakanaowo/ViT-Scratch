import torch
from torch import nn

from patch_embeddings import num_patches, embed_dim, patches

position_embeddings = nn.Embedding(num_patches,
                                   embed_dim)  # "lookup table" with `num_patches` row, each row has a vector with `embed_dim` dimension
position_ids = torch.arange(num_patches).expand((1, -1))  # create tensor index for each patch above
# tensor [0,...,195].expand -> (1,196)

# print("shape position id:", position_ids.shape) # torch size ([1,196])

# flattening ~ prepare input for transformer (transformer work on sequence vector)
embeddings = patches.flatten(start_dim=2, end_dim=-1)  # (1,768,14,14)->(1,768,196)
embeddings = embeddings.transpose(1,
                                  2)  # convert to transformer input standard:batch_size,sequence_len,embed_dim (1,768,196) -> (1,196,768)

embeddings += position_embeddings(position_ids)  # add patch's info
# => embeddings shape = [1,196,768]

# print(embeddings.shape)
