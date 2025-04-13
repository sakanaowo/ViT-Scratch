from torch import nn, no_grad
from torchvision import transforms

from Document_ import Image_Preprocessing


def preprocess_image(image, image_size=224):
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = preprocess(image)  # Tensor shape (C,H,W) ~ (3,224,224)
    image_tensor = image_tensor.unsqueeze(0)  # batch size = 1 -> tensor.shape with batch = (1,3,224,224)
    return image_tensor


image_tensor = preprocess_image(image=Image_Preprocessing.img)
embed_dim = 768
patch_size = 16
image_size = 224
num_patches = (image_size // patch_size) ** 2

# divide img into patch and create embeddings

with no_grad():
    # divide img into batches and map to vector embedding
    patch_embedding = nn.Conv2d(
        in_channels=3,  # RGB
        out_channels=embed_dim,
        kernel_size=patch_size,
        stride=patch_size
    )
    patches = patch_embedding(image_tensor)
    # shape (1,768,14,14)

# print("Patch shape: ", patches.shape)
# print("Number of patches: ", num_patches)
#
# position_embeddings = nn.Embedding(num_patches,
#                                    embed_dim)  # "lookup table" with `num_patches` row, each row has a vector with `embed_dim` dimension
# position_ids = torch.arange(num_patches).expand((1, -1))  # create tensor index for each patch above
# # tensor [0,...,195].expand -> (1,196)
#
# # print("shape position id:", position_ids.shape) # torch size ([1,196])
#
# # flattening ~ prepare input for transformer (transformer work on sequence vector)
# embeddings = patches.flatten(start_dim=2, end_dim=-1)  # (1,768,14,14)->(1,768,196)
# embeddings = embeddings.transpose(1,
#                                   2)  # convert to transformer input standard:batch_size,sequence_len,embed_dim (1,768,196) -> (1,196,768)
#
# embeddings += position_embeddings(position_ids)  # add patch's info
# # => embeddings shape = [1,196,768]
