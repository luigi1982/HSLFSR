import torch
from models.epit import EPIT

f = EPIT(64)
print(sum(p.numel() for p in f.parameters()))
x = torch.randn((1, 1, 25, 32, 32))
y = f(x)

print(y.shape)

