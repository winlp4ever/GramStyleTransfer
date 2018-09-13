# this code is copied from pytorch example directory with some slight modifies and is not intended to be shown to public. If you're looking at it, so sorry. I'm learning
# deep learning and because I work on several computers at the same time, upload to my git repository is the best way to keep all works syncing.
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import glob

from tensorboardX import SummaryWriter

import os
import shutil


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.writer = SummaryWriter()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def train_epoch(self, args, device, train_loader, optimizer, epoch):
        self.train()
        train_loss = 0
        train_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()), end='\r', flush=True)

            if epoch % args.sv_interval == 0:
                self.save_checkpoint(args,
                                     {
                                         'epoch': epoch,
                                         # 'arch': args.arch,
                                         'state_dict': self.state_dict(),
                                         'optimizer': optimizer.state_dict(),
                                     }, epoch, False)
            train_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss /= len(train_loader.dataset)
        train_correct = 100. * train_correct / len(train_loader.dataset)
        self.writer.add_scalar('train_loss', train_loss, global_step=epoch)
        self.writer.add_scalar('train_correct', train_correct, global_step=epoch)

    def save_checkpoint(self, args, state, epoch, is_best):
        filename = os.path.join(args.ckpt_path, 'checkpoint-{}.pth.tar'.format(epoch))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(args.ckpt_path, 'model_best.pth.tar'))

    def load_checkpoint(self, ckpt_path, optimizer):
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
            self.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(path, checkpoint['epoch']))
            return checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))
            return 0

    def test_epoch(self, device, test_loader, epoch=None):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        if epoch is not None:
            self.writer.add_scalar('test_loss', test_loss, global_step=epoch)
            self.writer.add_scalar('test_correct', correct * 100. / len(test_loader), global_step=epoch)

    def training_phase(self, args, device, train_loader, test_loader, optimizer):
        if args.resume:
            last_epoch = self.load_checkpoint(args.ckpt_path, optimizer)
        else:
            last_epoch = 0
        for epoch in range(1, args.epochs + 1):
            self.train_epoch(args, device, train_loader, optimizer, last_epoch + epoch)
            self.test_epoch(device, test_loader, last_epoch + epoch)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--sv-interval', type=int, default=10)
    parser.add_argument('--ckpt-path', '-c', default='./checkpoints/', nargs='?')
    parser.add_argument('--resume', type=bool, default=False, const=True, nargs='?')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    model.training_phase(args, device, train_loader, test_loader, optimizer)


if __name__ == '__main__':
    main()
