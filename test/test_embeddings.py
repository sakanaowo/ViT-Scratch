import torch

from Document_.Image_Preprocessing import vision_model
from Document_.patch_embeddings import image_tensor
from test.vision_embeddings import SiglipVisionEmbeddings, SiglipVisionConfig

embd = SiglipVisionEmbeddings(SiglipVisionConfig())
print(embd(image_tensor).shape)


def test_embeddings_equivalence():
    our_state_dict = embd.state_dict()
    hf_state_dict = {k.replace("vision_model.embeddings.", ""): v for k, v in vision_model.state_dict().items() if "vision_model.embeddings." in k}
    our_state_dict.update(hf_state_dict)
    embd.load_state_dict(our_state_dict)

    with torch.no_grad():
        our_output = embd(image_tensor)
        hf_output = vision_model.vision_model.embeddings(image_tensor)
        print("Max difference between our output and HF output:", torch.max(torch.abs(our_output - hf_output))) # =0, so they match!