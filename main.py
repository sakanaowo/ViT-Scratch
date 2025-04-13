from PIL import Image
import torch
from sympy.strategies.core import switch
from torchvision import transforms
from models.config import SiglipVisionConfig
from models.model import SiglipVisionModel
from utils.image_utils import get_image_fromURL, load_image
from utils.visualization import visualize_patch_embeddings, visualize_attention_map

if __name__ == "__main__":
    IMG_URL = input("Image URL plz: ")
    image_path = get_image_fromURL(IMG_URL)

    image_tensor = load_image(image_path)
    config = SiglipVisionConfig()
    model = SiglipVisionModel(config)
    output = model(image_tensor)
    print("Output shape:", output.shape)
    print("Do you want to visualize: ")
    n = int(input(f"1. Patch Embeddings\n"
                  f"2. Attention Mappings\n"))
    match n:
        case 1:
            visualize_patch_embeddings(output)
        case 2:
            visualize_attention_map(output)
        case _:
            exit(1)
