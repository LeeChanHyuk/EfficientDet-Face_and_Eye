import torch

a =torch.ones(3,1)
b = torch.zeros(1,3)
c = []
c.append(a)
c.append(b)
d = []
d.append(a)
d.append(a)
d = torch.stack(d)
print(a)