```
siglip_vit/
├── models/
│   ├── __init__.py
│   ├── config.py               # Chứa SiglipVisionConfig
│   ├── embeddings.py           # Patch + Position Embedding
│   ├── attention.py            # Self-Attention + Multi-head
│   ├── mlp.py                  # Feedforward (MLP)
│   ├── encoder.py              # Encoder layer + Encoder stack
│   └── model.py                # VisionTransformer & VisionModel
│
├── utils/
│   ├── __init__.py
│   ├── image_utils.py          # preprocess_image, load image, create dummy img
│   └── visualization.py        # Show embeddings & attention maps
│
├── test/
│   ├── __init__.py
│   └── test_embeddings.py      # So sánh với HF, test shape
│
├── main.py                     # Chạy forward qua mô hình
├── README.md                   # Ghi chú mô tả project
└── requirements.txt            # torch, transformers, etc.
``` 
