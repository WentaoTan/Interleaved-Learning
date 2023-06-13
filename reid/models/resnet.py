from __future__ import absolute_import

import random

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from torch.distributions import Uniform

__all__ = ['ResNet', 'resnet50']


class ResNet(nn.Module):
    __factory = {
        50: torchvision.models.resnet50,
    }

    def __init__(self, depth, Style_layer=['layer1']):
        super(ResNet, self).__init__()
        self.depth = depth
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        del resnet.fc, resnet.avgpool
        # self.base = nn.Sequential(
        #     resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        #     resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.base = resnet
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.Style_layer = Style_layer
        if Style_layer:
            self.style = UBS()
        self.feat_bn0 = nn.BatchNorm1d(2048)
        self.feat_bn0.bias.requires_grad_(False)
        init.constant_(self.feat_bn0.weight, 1)
        init.constant_(self.feat_bn0.bias, 0)

    def forward(self, x, style=False):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        if 'layer1' in self.Style_layer and style:
            x = self.style(x)
        x = self.base.layer2(x)
        if 'layer2' in self.Style_layer and style:
            x = self.style(x)
        x = self.base.layer3(x)
        if 'layer3' in self.Style_layer and style:
            x = self.style(x)
        x = self.base.layer4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        bn_x = self.feat_bn0(x)
        return F.normalize(bn_x, dim=1)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


class UBS(nn.Module):

    def __init__(self, p=1.0, rho=3.0, eps=1e-6):
        super().__init__()
        self.p = p
        self.rho = rho
        self.eps = eps

    def __repr__(self):
        return f'UBS(rho={self.rho}, p={self.p})'

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        mu_1 = x.mean(dim=[2, 3], keepdim=True)
        std_1 = x.std(dim=[2, 3], keepdim=True)

        mu_mu = mu_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        mu_std = mu_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        std_mu = std_1.mean(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        std_std = std_1.std(dim=0, keepdim=True).squeeze(0).squeeze(1).squeeze(1)
        mu_std.data.clamp_(min=self.eps)
        std_std.data.clamp_(min=self.eps)

        Distri_mu = Uniform(mu_mu - self.rho * mu_std, mu_mu + self.rho * mu_std)
        Distri_std = Uniform(std_mu - self.rho * std_std, std_mu + self.rho * std_std)

        mu_b = Distri_mu.sample([B, ])
        sig_b = Distri_std.sample([B, ])
        mu_b = mu_b.unsqueeze(2).unsqueeze(2)
        sig_b = sig_b.unsqueeze(2).unsqueeze(2)
        mu_b, sig_b = mu_b.detach(), sig_b.detach()

        return x_normed * sig_b + mu_b
