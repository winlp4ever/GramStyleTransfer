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
    phi_ = phi - m
    u = np.dot(phi_, phi_.T) + 1e-12
    w, v = eigh(u)
    w[w < 0] = 1e-6
    w = np.sqrt(w)
    return m, w, v


def matching(phic, phis, alpha=1.0):
    mc, Dc12, Ec = factorise(phic)
    ms, Ds12, Es = factorise(phis)

    phic_ = np.dot(np.dot(Ec, np.diag(1 / Dc12)), Ec.T)
    phic_ = np.dot(phic_, phic)
    phisc = np.dot(np.dot(Es, np.diag(Ds12)), Es.T)
    phisc = np.dot(phisc, phic_)
    return alpha * phisc + (1 - alpha) * phic + ms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f')
    parser.add_argument('--style', '-s')
    parser.add_argument('--level', '-l', type=int, default=0)
    parser.add_argument('--ckpt-path', '-c', default='./uni/checkpoints')
    parser.add_argument('--src-path', default='./images')
    parser.add_argument('--size', default=[224], nargs='*', type=int)
    parser.add_argument('--alpha', default=1.0, type=float)

    args = parser.parse_args()

    ckpt_path = os.path.join(args.ckpt_path, str(args.level))

    if len(args.size) == 2:
        height = args.size[0]
        width = args.size[1]
    elif len(args.size) == 1:
        height = width = args.size[0]
    else:
        raise ValueError('unaccepted value of im size!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vgg19_autoencoder().to(device)

    load_checkpoint(model, ckpt_path)

    def process_im(fp):
        im = preprocess(fp, im_size=(height, width), subtract_mean=False)
        im = torch.unsqueeze(im, 0)
        embed = model.forward(im.to(device), decode=False, level=args.level)
        return embed

    em_c_ts = process_im(os.path.join(args.src_path, args.filename + '.jpg'))
    em_s_ts = process_im(os.path.join(args.src_path, args.style + '.jpg'))

    em_c = em_c_ts.cpu().detach().numpy()[0]
    em_s = em_s_ts.cpu().detach().numpy()[0]

    embed_shape = em_c.shape

    em_c = np.reshape(em_c, newshape=(embed_shape[0], -1))
    em_s = np.reshape(em_s, newshape=(embed_shape[0], -1))

    em_cs = matching(em_c, em_s, alpha=args.alpha)
    em_cs = np.reshape(em_cs, embed_shape)

    em = torch.from_numpy(em_cs).to(device)
    em = torch.unsqueeze(em, 0)

    o_t = model.forward(x=None, decode_fr_embed=True, embed_=em, level=args.level)
    o_im = postprocess(o_t.cpu()[0], substract_mean=False)
    o_im.show()