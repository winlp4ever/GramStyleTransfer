import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import numpy as np
from PIL import Image

import cv2

from model import VGG
from model import make_layers
from model import cfg
from model import model_urls


class vgg_extractor(VGG):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(vgg_extractor, self).__init__(features, num_classes, init_weights)
        self.ls_layers = None

    def set_ls_layers(self, ls_layers):
        self.ls_layers = ls_layers

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        assert self.ls_layers is not None, 'you have to define ls_layers first!'
        indx = 0
        curr_feat = x
        chosen_features = []
        for indx_ in self.ls_layers:
            curr_feat = self.features[indx:indx_](curr_feat)
            chosen_features.append(curr_feat)
            indx = indx_
        return chosen_features

def preprocess(fn, im_size=224):
    im = Image.open(fn)
    if isinstance(im_size, int):
        height = width = im_size
    elif isinstance(im_size, (tuple, list)) and len(im_size) == 2:
        height = im_size[0]
        width = im_size[1]
    else:
        raise ValueError('not an accepted dim!')
    prep = transforms.Compose([transforms.Resize((height, width)),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul_(255)),
                               ])
    return prep(im)


def postprocess(tensor):
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                                 transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                      std=[1, 1, 1]),
                                 transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                                 ])
    postpb = transforms.Compose([transforms.ToPILImage()])
    t = postpa(tensor)
    t[t>1]=1
    t[t<0]=0
    img = postpb(t)
    return img


def vgg19_bn_extractor(**kwargs):
    kwargs['init_weights'] = False

    model = vgg_extractor(make_layers(cfg['E'], batch_norm=False), **kwargs)
    model_dict = model.state_dict()

    pretrained_dict = model_zoo.load_url(model_urls['vgg19'])
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def mseloss(input, target):
    loss = 0
    for u, v in zip(input, target):
        loss += F.mse_loss(u, v)

    return loss


def synthesize(model, device, im, epochs, lr, im_size=224):
    for param in model.parameters():
        param.requires_grad = False

    if isinstance(im_size, int):
        height = width = im_size
    elif isinstance(im_size, (tuple, list)) and len(im_size) == 2:
        height = im_size[0]
        width = im_size[1]
    else:
        raise ValueError('not an accepted dim!')

    x = torch.zeros(3, height, width, requires_grad=True, device=device)

    feats = model.forward(im.float().to(device))
    feats = [u.detach() for u in feats]

    #optimizer = optim.SGD([x], lr=lr, momentum=momentum)
    optimizer = optim.LBFGS([x], max_iter=100, lr=lr)
    print('set up finished!')
    for epoch in range(1, epochs + 1):
        def closure():
            optimizer.zero_grad()
            output = model.forward(x)
            loss = mseloss(output, feats)
            loss.backward()
            return loss
        l = optimizer.step(closure)

        print('epoch:{}--loss:{}'.format(epoch, l), end='\r', flush=True)
        if epoch % 10 == 0:
            out_t = x.cpu()
            out_img = postprocess(out_t)
            print(type(out_img))
            out_img.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    E = vgg19_bn_extractor()
    #print(E.features)
    E = E.to(device)
    #ls_layers = (2, 9, 16, 29, 42)
    #ls_layers = (0, 7, 14, 27, 40)
    ls_layers = (1, 6, 11, 20, 29)
    E.set_ls_layers(ls_layers)
    filename = './images/im.jpg'
    resized_img = preprocess(filename, im_size=(224, 512))
    print(resized_img.shape)

    print('shape {}'.format(resized_img.shape))
    #print(resized_img)

    synthesize(E, device, resized_img, 100, 0.8, (224, 512))



