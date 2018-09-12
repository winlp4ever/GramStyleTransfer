from model import vgg19

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

a = torch.zeros(2, 2, requires_grad=True)

#print(a.detach().numpy())
epochs = 100
optimizer = optim.SGD([a], lr=1e-3)

for i in range(epochs):
    optimizer.zero_grad()
    b = torch.norm((a - 2) ** 2, 2)
    b.backward()
    optimizer.step()
    print('value of a: {}'.format(a.detach().numpy()))


class framework(nn.Module):
    def __init__(self):
        super(framework, self).__init__()
        model = nn.Sequential(OrderedDict([
            ("layer_one", nn.Linear(1, 10)), ("layer_two", nn.Linear(10, 3))
        ]))
        print(model._modules["layer_one"])
        print(model._modules["layer_two"])

    def forward(self):
        x = torch.randn(5, 5)