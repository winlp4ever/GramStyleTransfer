import argparse
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from extract import make_layers
import torch.utils.model_zoo as model_zoo
from model import model_urls
import os
import glob
from tensorboardX import SummaryWriter
import time
import utils

#levels = [1, 6, 11, 20, 29]
levels = [2, 9, 16, 29, 42]
d_levels = [40, 33, 26, 13, 0]


class unitransform(nn.Module):
    def __init__(self, cfg, dspl_cfg):
        super(unitransform, self).__init__()
        self.slf_prep = nn.Conv2d(3, 3, 1, 1, 0)
        self.features = make_layers(cfg, batch_norm=False)
        self.dspl = make_reversed_layers(dspl_cfg, batch_norm=False)
        self._init_weights()
        self.writer = SummaryWriter()

    def forward(self, x, decode=False, level=5, decode_fr_embed=False, embed_=None):
        if decode_fr_embed:
            assert embed_ is not None, 'embed must not be of NoneType!'
            return self.dspl[d_levels[level]:](embed_)

        u = self.slf_prep(x)
        embed = self.features[:levels[level] + 1](u)
        if decode:
            output = self.dspl[d_levels[level]:](embed)
            return embed, output
        return embed

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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
    in_channels = 512  # to be changed
    for v in cfg:
        if v == 'U':
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        else:
            # dspl_conv = nn.ConvTranspose2d(in_channels, v, kernel_size=3, padding=0)
            dspl_conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=0)
            if batch_norm:
                layers += [nn.ReflectionPad2d(padding=1), dspl_conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [nn.ReflectionPad2d(padding=1), dspl_conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _to_reflective_padding(model):
    """
    Replace zero padding in model.features' layers by reflective padding
    :param model: Model to change
    :return:
    """
    new_features = []
    for lyr in model.features:
        if isinstance(lyr, nn.Conv2d):
            conv_ = nn.Conv2d(lyr.in_channels, lyr.out_channels, padding=0, kernel_size=3)
            conv_.weight = nn.Parameter(lyr.weight.float())
            conv_.bias = nn.Parameter(lyr.bias.float())
            new_features += [nn.ReflectionPad2d(1), conv_]
        else:
            new_features += [lyr]
    model.features = nn.Sequential(*new_features)


def vgg19_autoencoder():
    model = unitransform(cfg['E'], dspl_cfg['E'])
    model_dict = model.state_dict()

    pretrained_dict = model_zoo.load_url(model_urls['vgg19'])
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def save_checkpoint(args, state, epoch, level):
    path = os.path.join(args.ckpt_path, str(level))
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, 'checkpoint-{}.pth.tar'.format(epoch))
    torch.save(state, filename)


def load_checkpoint(model, ckpt_path, optimizer=None):
    max_ep = 0
    path = ''
    for fp in glob.glob(os.path.join(ckpt_path, '*')):
        fn = os.path.basename(fp)
        fn_ = fn.replace('-', ' ')
        fn_ = fn_.replace('.', ' ')
        epoch = int(fn_.split()[1])
        if epoch > max_ep:
            path = fp
            max_ep = epoch

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        # args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
        return checkpoint['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_path))
        return 0


def training(model, device, train_loader, optimizer, lambd, epoch, level, fix=False):
    model.train()
    for param in model.features.parameters():
        param.requires_grad = False

    if level > 0 and fix:
        for lyr in model.dspl[d_levels[level - 1]:]:
            if isinstance(lyr, nn.Conv2d):
                lyr.weight.requires_grad = False
                lyr.bias.requires_grad = False

    train_loss = 0
    begin = time.time()
    for batch_idx, (data, _) in enumerate(train_loader):
        x = data.to(device)
        optimizer.zero_grad()
        embed, output = model.forward(x, decode=True, level=level)
        output_embed = model.forward(output, level=level)
        loss = F.mse_loss(input=output, target=x) + lambd * F.mse_loss(input=output_embed, target=embed)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / len(train_loader.dataset)
        if batch_idx % 5 == 0:
            print('Train epoch {0} -- loss {1:.6f} [{2:.2f}%] in {3}'.
                  format(epoch, train_loss, 100. * batch_idx / len(train_loader), utils.sec2time(time.time() - begin)),
                  end='\r', flush=True)

    model.writer.add_scalar('train_loss', train_loss, global_step=epoch)


def testing(model, device, test_loader, epoch, level):
    model.eval()
    print('\n')
    test_loss = 0
    for batch_idx, (data, _) in enumerate(test_loader):
        x = data.to(device)
        _, output = model.forward(x, decode=True, level=level)
        loss = F.mse_loss(input=output, target=x)
        test_loss += loss.item() / len(test_loader.dataset)
        print('Eval: test loss {0:.6f} [{1:.2f}%]'.
              format(test_loss, 100. * batch_idx / len(test_loader)),
              end='\r', flush=True)
    print('\n')

    model.writer.add_scalar('test_loss', test_loss, global_step=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--traindir', help='dir where locates training dataset', default='../datasets/coco/train')
    parser.add_argument('--testdir', help='dir where locates testing dataset', default='../datasets/coco/test')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', help='initial learnning rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lambd', type=float, default=1.0)
    parser.add_argument('--ckpt-path', help='path to the saved checkpoint' , default='./uni/checkpoints')
    parser.add_argument('--resume', help='whether continue training or begin from scratch',
                        type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--gamma', help='const helps updating learning rate', type=float, default=5e-5)
    parser.add_argument('--level', type=int, default=4)
    parser.add_argument('--ckpt-level', type=int, default=4)

    args = parser.parse_args()

    use_cuda = True

    height = width = 224
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    prep = transforms.Compose([transforms.Resize((height, width)),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul_(255)),
                               ])



    # load datasets
    train_dataset, test_dataset = map(lambda dir: datasets.ImageFolder(
        dir, prep
    ), [args.traindir, args.testdir])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size // 2, shuffle=True, **kwargs
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vgg19_autoencoder()
    _to_reflective_padding(model)
    model = model.to(device)
    #print(model.dspl)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.dspl.parameters()), lr=args.lr)


    def lr_updater(optim_, ep_):
        """
        Update learning rate in optimizer object:
        $$lr = lr * 1 / (1 + gamma * epoch)$$
        :param optim_: pytorch optimizer object that contains learning rate attribute
        :param ep_: current epoch
        :return:
        """
        for param_group in optim_.param_groups:
            param_group['lr'] *= 1 / (1 + args.gamma * ep_)


    ckpt_path = os.path.join(args.ckpt_path, str(args.ckpt_level))
    if args.resume:
        fix = True
        if args.ckpt_level < args.level:
            last_epoch = 0
            load_checkpoint(model, ckpt_path)
        else:
            last_epoch = load_checkpoint(model, ckpt_path, optimizer)
    else:
        fix = False
        last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print('\ncurr learning rate: {0:.6f}'.format(optimizer.param_groups[0]['lr']))
        training(model, device, train_loader, optimizer, lambd=args.lambd,
                 epoch=epoch + last_epoch, level=args.level, fix=False)
        save_checkpoint(args,
                        {
                            'epoch': last_epoch + epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, epoch + last_epoch, args.level)
        lr_updater(optimizer, epoch + last_epoch)
        if (epoch + last_epoch) % 5 == 0:
            testing(model, device, test_loader, epoch + last_epoch, args.level)
