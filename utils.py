import numpy as np
from PIL import Image
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

def sec2time(secs):
    secs = int(secs)
    if secs < 60:
        return "{}s".format(secs)
    if secs < 60 * 60:
        return "{}'{}s".format(secs // 60, secs % 60)
    return "{}h{}'{}s".format(secs // (60 * 60), secs % (60 * 60) // 60, secs % 60)


def preprocess(fn, im_size=224, subtract_mean=True):
    im = Image.open(fn)
    if isinstance(im_size, int):
        height = width = im_size
    elif isinstance(im_size, (tuple, list)) and len(im_size) == 2:
        height = im_size[0]
        width = im_size[1]
    else:
        raise ValueError('not an accepted dim!')
    ls_transforms = [transforms.Resize((height, width)),
                     transforms.ToTensor(),
                     transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                     transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                          std=[1, 1, 1]),
                     transforms.Lambda(lambda x: x.mul_(255)),]

    if not subtract_mean:
        del ls_transforms[3]
    prep = transforms.Compose(ls_transforms)

    return prep(im)


def postprocess(tensor, substract_mean=True):
    ls_transforms = [transforms.Lambda(lambda x: x.mul_(1. / 255)),
                                 transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                      std=[1, 1, 1]),
                                 transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                                 ]

    if not substract_mean:
        del ls_transforms[1]
    postpa = transforms.Compose(ls_transforms)
    postpb = transforms.Compose([transforms.ToPILImage()])
    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    return img


if __name__ == '__main__':
    print(sec2time(4600))