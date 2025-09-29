import torch
from models.f3dun import F3DUN

f = F3DUN(64)
print(sum(p.numel() for p in f.parameters()))
x = torch.randn((1, 25, 32, 32))
y = f(x)
print(y.shape)


