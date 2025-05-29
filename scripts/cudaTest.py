import torch
print("CUDA available?", torch.cuda.is_available())
x = torch.rand(3,4, device="cuda")
print("Tensor on", x.device)
