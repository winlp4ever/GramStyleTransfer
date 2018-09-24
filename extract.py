import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import numpy as np
from PIL import Image
import argparse
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
    t[t > 1] = 1
    t[t < 0] = 0
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


def mseloss(input, target, size_average=True):
    loss = 0
    for u, v in zip(input, target):
        loss += F.mse_loss(u, v, size_average)

    return loss


def style_loss(features):
    ls_grams = []
    for f in features:
        f_ = f.view(f.shape[1], -1)
        n = f_.shape[0]
        m = f_.shape[1]
        gram_f = torch.matmul(f_, torch.transpose(f_, 0, 1)) / (2 * n * m)
        ls_grams.append(gram_f)
    return ls_grams


def synthesize(model, device, im_c, im_s, epochs, lr, im_size=224, lambd=1.0):
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

    # content features
    feats_cnt = model.forward(im_c.float().to(device))
    feats_cnt = [u.detach() for u in feats_cnt]

    # style features
    feats_stl = model.forward(im_s.float().to(device))
    grams = style_loss(feats_stl)
    grams = [g.detach() for g in grams]

    # optimizer = optim.SGD([x], lr=lr, momentum=momentum)
    optimizer = optim.LBFGS([x], max_iter=100, lr=lr)
    print('set up finished!')
    for epoch in range(1, epochs + 1):
        def closure():
            optimizer.zero_grad()
            output = model.forward(x)
            out_grams = style_loss(output)
            l_c = mseloss(output, feats_cnt)
            l_s = lambd * mseloss(out_grams, grams, size_average=False)
            loss = l_c + l_s
            print('epoch:{0}/{1}--loss: total {2:6.2f} content {3:6.2f} style {4:6.2f} '.
                  format(epoch, epochs, loss, l_c, l_s), end='\r', flush=True)
            loss.backward()
            return loss

        optimizer.step(closure)

        if epoch % 10 == 0:
            out_t = x.cpu()
            out_img = postprocess(out_t)
            out_img.show()

    print('\nfinished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--size', type=int, nargs='*', default=(224,))
    parser.add_argument('--lambd', '-l', type=float, default=1e-2)

    args = parser.parse_args()
    size = args.size
    if len(size) < 2:
        size = size[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    E = vgg19_bn_extractor()
    # print(E.features)
    E = E.to(device)
    # ls_layers = (2, 9, 16, 29, 42)
    # ls_layers = (0, 7, 14, 27, 40)
    ls_layers = (1, 6, 11, 20, 29)
    E.set_ls_layers(ls_layers)
    fn_c = './images/cnt.jpg'
    img_c = preprocess(fn_c, im_size=size)
    print('shape {}'.format(img_c.shape))
    # print(resized_img)
    fn_s = './images/stl.jpg'
    img_s = preprocess(fn_s, im_size=size)


    synthesize(E, device, img_c, img_s, args.epochs, 0.8, size, lambd=args.lambd)
