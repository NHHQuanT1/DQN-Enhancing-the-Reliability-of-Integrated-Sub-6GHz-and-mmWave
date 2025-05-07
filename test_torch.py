import torch

device = torch.device("mps")
x = torch.tensor([1, 2, 3]).to(device)
print(x.device) 

