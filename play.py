import torch
from models.SSR.mstpp import MST

f = MST()
print(sum(p.numel() for p in f.parameters()))
x = torch.randn((1, 25, 32, 32))
z = f(x)
print(z.shape)


