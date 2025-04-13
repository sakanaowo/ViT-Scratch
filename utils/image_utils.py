from PIL import Image
from torchvision import transforms
import torch


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # shape: (1, 3, 224, 224)


def load_image(path: str, image_size: int = 224) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return preprocess_image(img, image_size)


def create_dummy_image(image_size: int = 224) -> torch.Tensor:
    img = Image.new("RGB", (image_size, image_size), color="gray")
    return preprocess_image(img, image_size)


def get_image_fromURL(url):
    import requests
    from datetime import datetime
    import os

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    # folder = os.path.join("..", "Images")
    folder = "Images"
    os.makedirs(folder, exist_ok=True)
    file_name = "IMG" + datetime.now().strftime("%m-%d-%H-%M-%S") + ".jpg"
    file_path = os.path.join(folder, file_name)
    response = requests.get(url, headers=headers, allow_redirects=True)
    if response.status_code != 200:
        print("Failed to get image: ", response.status_code)
        return None
    else:
        with open(file_path, "wb") as f:
            f.write(response.content)
            print("Image loaded into: ", file_path)
        return file_path
