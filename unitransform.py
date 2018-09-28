import argparse
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VGG
from model import make_layers
import torch.utils.model_zoo as model_zoo
from model import model_urls

class unitransform(nn.Module):
    def __init__(self, cfg, dspl_cfg):
        super(unitransform, self).__init__()
        self.features = make_layers(cfg, batch_norm=False)
        self.dspl = make_reversed_layers(dspl_cfg, batch_norm=False)
        self._init_weights()

    def forward(self, x):
        embed = self.features(x)
        output = self.dspl(embed)
        return embed, output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


cfg = {'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512]}
dspl_cfg = {'E': [512, 'U', 512, 512, 512, 256, 'U', 256, 256, 256, 128, 'U', 128, 64, 'U', 64, 3]}


def make_reversed_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 512 # to be changed
    for v in cfg:
        if v == 'M':
            layers.append(nn.Upsample(size=2, mode='nearest'))
        else:
            dspl_conv = nn.ConvTranspose2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [dspl_conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [dspl_conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg19_autoencoder():
    model = unitransform(cfg['E'], dspl_cfg['E'])
    model_dict = model.state_dict()

    pretrained_dict = model_zoo.load_url(model_urls['vgg19'])
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def train(model, device, train_loader, rl, lambd):
    model.train()
    for param in model.features.parameters():
        param.requires_grad=False

    optimizer = optim.Adam(model.dspl.parameters(), rl)

    for batch_idx, data in enumerate(train_loader):
        x = data.to(device)
        optimizer.zero_grad()
        embed, output = model.forward(x)
        output_embed, _ = model.forward(output)
        loss = F.mse_loss(input=output, target=x) + lambd * F.mse_loss(input=output_embed, target=embed)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    use_cuda = True
    traindir = '../datasets/coco'
    height = width = 224
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    batch_size = 32

    prep = transforms.Compose([transforms.Resize((height, width)),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul_(255)),
                               ])

    train_dataset = datasets.ImageFolder(
        traindir,
        prep
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, **kwargs
    )

