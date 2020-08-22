from __future__ import print_function
import torch
x = torch.empty(5, 3)
print(x)
y = torch.rand(5, 3)
print(y)
z = torch.zeros(5, 3, dtype=torch.long)
print(z)
l = torch.tensor([5.5, 3])
print(l)

z = z.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(z)

z = torch.randn_like(x, dtype=torch.float)    # 重载 dtype!
print(z)                                      # 结果size一致

print(z.size())

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print("result: ", result, "\n", " x+y:  ", x + y)

print("y.add(x):", y.add(x))

print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(16, -1)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype