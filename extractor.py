import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

class VGG_Extractor(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG_Extractor, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, ls_layers):
        indx = 0
        chosen_features = []
        for name in ls_layers:
            indx_ = list(dict(self.features.named_children()).keys()).index(name)
            chosen_features.append(self.features[indx:indx_](x))
        return chosen_features

    def synthesis_via_optim(self):




cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_features(cfg, bn):
    i = 1
    j = 1
    features = OrderedDict()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            features['max_pool_{}'.format(i)] = nn.MaxPool2d(kernel_size=2, stride=2)
            i += 1
            j = 1
        else:
            features['conv_{}_{}'.format(i, j)] = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if bn:
                features['bn_{}_{}'.format(i, j)] = nn.BatchNorm2d(v)
            features['relu_{}_{}'.format(i, j)] = nn.ReLU(inplace=True)
            j += 1
            in_channels = v
    return nn.Sequential(features)
if __name__ == '__main__':
    model = VGG_Extractor(make_features(cfg['E'], bn=True))
    print(model)
    print(list(dict(model.features.named_children()).keys()).index('conv_1_1'))



