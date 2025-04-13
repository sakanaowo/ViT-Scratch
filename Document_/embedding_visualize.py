import matplotlib.pyplot as plt
import torch

from position_embedding import embeddings
from Image_Preprocessing import vision_model, processor, img

# patches_viz = embeddings[0].detach().numpy()  # shape: [196,768]
#
# plt.figure(figsize=(15, 8)) # 15x8 inches
# plt.imshow(patches_viz, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('Visualized Patches')
# plt.xlabel('Embedding Dimension')
# plt.ylabel('Number of Patches')
# plt.show()

vision_model.eval()
inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    patch_embeddings = vision_model.vision_model.embeddings(inputs.pixel_values)

print("patch_embedding shape:", patch_embeddings.shape)

patches_viz = embeddings[0].detach().numpy()  # shape: [196,768]

plt.figure(figsize=(15, 8)) # 15x8 inches
plt.imshow(patches_viz, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Visualized Patches')
plt.xlabel('Embedding Dimension')
plt.ylabel('Number of Patches')
plt.show()