import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import os
from PIL import Image
from torchvision import transforms

from extract import preprocess, postprocess
from unitransform import vgg19_autoencoder, load_checkpoint


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

    im = preprocess(os.path.join(args.src_path, args.filename + '.jpg'), im_size=(height, width))
    im = torch.unsqueeze(im, 0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vgg19_autoencoder().to(device)

    load_checkpoint(model, ckpt_path)

    _, o_t = model.forward(im.to(device), decode=True, level=args.level)
    print(o_t.shape)

    o_im = postprocess(o_t.cpu()[0])
    o_im.show()





