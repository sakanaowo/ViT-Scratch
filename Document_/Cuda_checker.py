import torch

# Kiểm tra có GPU không
print("CUDA available:", torch.cuda.is_available())

# Nếu có, in thêm thông tin về GPU
if torch.cuda.is_available():
    print("Số lượng GPU:", torch.cuda.device_count())
    print("Tên GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    print("Không phát hiện GPU hỗ trợ CUDA.")
