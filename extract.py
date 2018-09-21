import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import numpy as np

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


def crop_image(fn, new_width=224, new_height=224):
    im = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    dim = (new_width, new_height)
    return cv2.resize(im, dim, interpolation=cv2.INTER_AREA)


def vgg19_bn_extractor(**kwargs):
    kwargs['init_weights'] = False

    model = vgg_extractor(make_layers(cfg['E'], batch_norm=True), **kwargs)
    model_dict = model.state_dict()

    pretrained_dict = model_zoo.load_url(model_urls['vgg19_bn'])
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def mseloss(input, target):
    loss = 0
    for u, v in zip(input, target):
        loss += F.mse_loss(u, v)

    return loss


def synthesize(model, device, im, epochs, lr, momentum):
    for param in model.parameters():
        param.requires_grad = False

    x = torch.zeros(3, 224, 224, requires_grad=True, device='cuda')

    feats = model.forward(torch.from_numpy(im).float().to(device))
    feats = [u.detach() for u in feats]

    optimizer = optim.SGD([x], lr=lr, momentum=momentum)
    print('set up finished!')
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model.forward(x)
        loss = mseloss(output, feats)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print('epoch:{}--loss:{}'.format(epoch, loss.item()))
            #cv2.imshow('synthesized img', x.detach().numpy())
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    resynth_img = np.transpose(x.cpu().detach().numpy(), (1, 2, 0))
    cv2.imshow('resynthesized_img', resynth_img * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    device = torch.device("cuda")
    E = vgg19_bn_extractor()
    E = E.to(device)
    ls_layers = (2, 9, 16, 29, 42)
    E.set_ls_layers(ls_layers)
    filename = './images/im.jpg'
    resized_img = crop_image(filename)
    resized_img = np.transpose(resized_img, (2, 0, 1))
    resized_img = resized_img / 255
    print('shape {}'.format(resized_img.shape))
    print(resized_img)
    #cv2.imshow('crop image', resized_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #img = np.ones((3, 224, 224), dtype=float)
    synthesize(E, device, resized_img, 10000, 1e-3, 0.5)



