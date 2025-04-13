from PIL import Image

img = Image.open('Document_/Images/catImg.jpg')
# img.show()

from transformers import AutoProcessor, SiglipVisionModel, SiglipVisionConfig

processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
vision_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224",
                                                 config=SiglipVisionConfig(vision_use_head=False))
# print(vision_model)
