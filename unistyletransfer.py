import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from numpy.linalg import eigh

from utils import preprocess, postprocess
from unitransform import vgg19_autoencoder, load_checkpoint


def factorise(phi):
    m = np.mean(phi)
    phi_ = phi
    u = np.dot(phi_, phi_.T) + 1e-12
    w, v = eigh(u)
    w = np.sqrt(w)
    return m, w, v


def matching(phic, phis, alpha=1.0):
    mc, Dc, Ec = factorise(phic)
    ms, Ds, Es = factorise(phis)

    phic_ = np.dot(np.dot(Ec, np.diag(1 / Dc)), Ec.T)
    phic_ = np.dot(phic_, phic)
    phisc = np.dot(np.dot(Es, np.diag(Ds)), Es.T)
    phisc = np.dot(phisc, phic_)
    return alpha * phisc + (1 - alpha) * phic


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f')
    parser.add_argument('--level', '-l', type=int, default=0)
    parser.add_argument('--ckpt-path', '-c', default='./uni/checkpoints')
    parser.add_argument('--src-path', default='./images')
    parser.add_argument('--size', default=[224], nargs='*', type=int)

    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_path, str(args.level))

    if len(args.size) == 2:
        height = args.size[0]
        width = args.size[1]
    elif len(args.size) == 1:
        height = width = args.size[0]
    else:
        raise ValueError('unaccepted value of im size!')

    im = preprocess(os.path.join(args.src_path, args.filename + '.jpg'), im_size=(height, width), subtract_mean=False)
    im = torch.unsqueeze(im, 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vgg19_autoencoder().to(device)

    load_checkpoint(model, ckpt_path)

    _, o_t = model.forward(im.to(device), decode=True, level=args.level)
    print(o_t.shape)

    o_im = postprocess(o_t.cpu()[0], substract_mean=False)
    o_im.show()