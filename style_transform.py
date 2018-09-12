from model import vgg19

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

a = torch.zeros(2, 2, requires_grad=True)
b = torch.norm((a - 2) ** 2, 2)

#print(a.detach().numpy())
epochs = 100
optimizer = optim.SGD([a], lr=1e-3)

for i in range(epochs):
    optimizer.zero_grad()
    b.backward()
    optimizer.step()
    print('value of a: {}'.format(a.detach().numpy()))