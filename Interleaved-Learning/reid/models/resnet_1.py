# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math
import random
from torch.nn import functional as F
import faiss
import torch
import torch.utils.model_zoo as model_zoo

from einops import rearrange


__all__ = ['ResNet', 'resnet50', ]

from torch.nn import init

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'ibn_50x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
}

import math
from torch.nn.parameter import Parameter
from torch.distributions import Normal, Uniform
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, with_ibn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3], args=None):
        self.inplanes = 64
        super().__init__()
        # self.AmpIN = nn.InstanceNorm2d(3, affine=True, track_running_stats=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], with_ibn=False)
        # self.layer1.append(Bottleneck(256, 64, with_ibn=True))
        # self.register_buffer('attention1', torch.zeros(64))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, with_ibn=False)
        # self.layer2.append(Bottleneck(512, 128, with_ibn=True))
        # self.layer2.append(nn.BatchNorm2d(512))
        # self.register_buffer('attention2', torch.zeros(128))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, with_ibn=False)
        # self.layer3.append(Bottleneck(1024, 256, with_ibn=True))
        # self.layer3.append(nn.BatchNorm2d(1024))
        # self.register_buffer('attention3', torch.zeros(256))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        # self.layer4.append(Bottleneck(2048, 512, with_ibn=False))
        # self.layer4.append(nn.BatchNorm2d(2048))
        # self.layer4 = self._make_double_layer(Double_BasicBlock, 512, layers[3], stride=last_stride)
        # self.register_buffer('attention4', torch.zeros(512))
        self.args = args


        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.style_layers = ['layer1']
        if self.style_layers:
            self.style = UBS()
        self.feat_bn0 = nn.BatchNorm1d(2048)
        self.feat_bn0.bias.requires_grad_(False)
        # init.constant_(self.feat_bn0.weight, 1)
        # init.constant_(self.feat_bn0.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, with_ibn=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, with_ibn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, with_ibn=with_ibn))

        return nn.Sequential(*layers)


    def forward1(self, x, phase=None, targets=4, style=False, Recon=False, returnx=False, stage='l1'):
        if self.training:
            if stage == 'l1':
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                return x
            elif stage == 'l2':
                x = self.layer2(x)
                return x
            else:
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                bn_x = self.feat_bn0(x)

                return x, F.normalize(bn_x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            bn_x = self.feat_bn0(x)
            return F.normalize(bn_x)

    def forward(self, x, phase=None, targets=4, style=False, Recon=False, returnx=False, stage='l1', domain_id=4):

        # if phase is not None: x = torch.cat([x, phase], dim=0)

        # x_phase = self._phase(x)
        # x = torch.cat([x, x_phase], dim=0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        # x = self._stem(x)
        # shape = self._stem(phase)
        #
        # x = torch.cat([x, shape], dim=1)
        x = self.layer1(x)

        # B, C, H, W = x.size()
        # ori_x, aug_x = x.chunk(2, dim=0)
        # pair_inputs1 = F.normalize(ori_x.reshape(B // 2, C, -1), dim=-1)
        # pair_inputs2 = F.normalize(aug_x.reshape(B // 2, C, -1), dim=-1)
        # attention_weight = (pair_inputs1 * pair_inputs2).sum(dim=-1)
        # attention_weight = attention_weight.reshape(B // 2, C).unsqueeze(-1).unsqueeze(-1)
        # weighted_ori_x = ori_x * attention_weight
        # # weighted_aug_x = aug_x * attention_weight
        # x = torch.cat([weighted_ori_x,aug_x],dim=0)

        # if self.training:
        #     ori_out, aug_out = x.chunk(2, dim=0)
        #     B, C, H, W = ori_out.size()
        #     pair_inputs1 = F.normalize(ori_out.reshape(B, C, -1), dim=-1)
        #     pair_inputs2 = F.normalize(aug_out.reshape(B, C, -1), dim=-1)
        #     attention_weight = (pair_inputs1 * pair_inputs2).sum(dim=-1).mean(0)
        #     self.attention1.data.copy_(0.9 * self.attention1 + 0.1 * attention_weight)

        if self.style is not None and style and 'layer1' in self.style_layers:
            # targets = torch.cat([targets,targets],dim=0)
            # x = torch.cat([x, self.style(x)], dim=0)
            x = self.style(x)
        x = self.layer2(x)

        # if self.training:
        #     ori_out, aug_out = x.chunk(2, dim=0)
        #     B, C, H, W = ori_out.size()
        #     pair_inputs1 = F.normalize(ori_out.reshape(B, C, -1), dim=-1)
        #     pair_inputs2 = F.normalize(aug_out.reshape(B, C, -1), dim=-1)
        #     attention_weight = (pair_inputs1 * pair_inputs2).sum(dim=-1).mean(0)
        #     self.attention2.data.copy_(0.9 * self.attention2 + 0.1 * attention_weight)
        if self.style is not None and style and 'layer2' in self.style_layers:
            # orix, adax = x.chunk(2, dim=0)
            # x = torch.cat([orix, self.style(adax)], dim=0)
            x = self.style(x)
        x = self.layer3(x)

        # if self.training:
        #     ori_out, aug_out = x.chunk(2, dim=0)
        #     B, C, H, W = ori_out.size()
        #     pair_inputs1 = F.normalize(ori_out.reshape(B, C, -1), dim=-1)
        #     pair_inputs2 = F.normalize(aug_out.reshape(B, C, -1), dim=-1)
        #     attention_weight = (pair_inputs1 * pair_inputs2).sum(dim=-1).mean(0)
        #     self.attention3.data.copy_(0.9 * self.attention3 + 0.1 * attention_weight)
        if self.style is not None and style and 'layer3' in self.style_layers:
            x = self.style(x)
        x = self.layer4(x)

        # if self.training:
        #     ori_out, aug_out = x.chunk(2, dim=0)
        #     B, C, H, W = ori_out.size()
        #     pair_inputs1 = F.normalize(ori_out.reshape(B, C, -1), dim=-1)
        #     pair_inputs2 = F.normalize(aug_out.reshape(B, C, -1), dim=-1)
        #     attention_weight = (pair_inputs1 * pair_inputs2).sum(dim=-1).mean(0)
        #     self.attention4.data.copy_(0.9 * self.attention4 + 0.1 * attention_weight)

        # B, C, H, W = x.size()
        # pairs = x.reshape(B // 2, 2, C, H, W)
        # pair_inputs1 = F.normalize(pairs[:, 0, :, :, :].squeeze().reshape(B // 2, C, -1), dim=-1)
        # pair_inputs2 = F.normalize(pairs[:, 1, :, :, :].squeeze().reshape(B // 2, C, -1), dim=-1)
        # attention_weight = (pair_inputs1*pair_inputs2).sum(dim=-1).repeat(2,1,1)
        # attention_weight = attention_weight.reshape(B,C).unsqueeze(-1).unsqueeze(-1)
        # x, shape = x.chunk(2, dim=0)
        # B, C, H, W = x.size()
        # pair_inputs1 = F.normalize(x.reshape(B, C, -1), dim=-1)
        # pair_inputs2 = F.normalize(shape.reshape(B, C, -1), dim=-1)
        # attention_weight = (pair_inputs1 * pair_inputs2).sum(dim=-1)
        # attention_weight = attention_weight.reshape(B, C).unsqueeze(-1).unsqueeze(-1)
        # x = x * attention_weight
        # if self.training:x, shape = x.chunk(2, dim=0)


        # B, C, H, W = x.size()
        # pairs = x.reshape(B // 2, 2, C, H, W)
        # pair_inputs1 = F.normalize(pairs[:, 0, :, :, :].squeeze().reshape(B // 2, C, -1), dim=-1)
        # pair_inputs2 = F.normalize(pairs[:, 1, :, :, :].squeeze().reshape(B // 2, C, -1), dim=-1)
        # attention_weight = (pair_inputs1 * pair_inputs2).sum(dim=-1).repeat(2, 1, 1)
        # attention_weight = attention_weight.reshape(B, C).unsqueeze(-1).unsqueeze(-1)
        # x, shape = x.chunk(2, dim=0)
        # B, C, H, W = x.size()
        # pair_inputs1 = F.normalize(x.reshape(B, C, -1), dim=-1)
        # pair_inputs2 = F.normalize(shape.reshape(B, C, -1), dim=-1)
        # attention_weight = (pair_inputs1 * pair_inputs2).sum(dim=-1)
        # attention_weight = attention_weight.reshape(B, C).unsqueeze(-1).unsqueeze(-1)
        '''similarity_max match'''
        # B, C, H, W = x.size()
        # x3d = F.normalize(x.reshape((B, C, H * W)), dim=-1)
        # attention_weight = []
        # for i in range(B):
        #     cur_feature = x3d[i].unsqueeze(0).repeat(B, 1, 1)
        #     a = (cur_feature * x3d).sum(dim=-1)
        #     a[i] = 0
        #     index = a.sum(-1).argmax()
        #     attention_weight.append(a[index].unsqueeze(0))
        # attention_weight = torch.cat(attention_weight, dim=0)
        # attention_weight = attention_weight.reshape(B, C).unsqueeze(-1).unsqueeze(-1)
        # x = x * attention_weight

        # x = x.reshape((B, C, H * W))
        # sim = torch.einsum('bcl,bcl->bcl', [x, x])
        # a = torch.nn.MultiheadAttention(512, 1, bias=False, batch_first=True)

        '''crossAttention'''
        # B, C, H, W = x.size()
        # x3d = x.reshape((B, C, H * W))
        # x3d = x3d.transpose(-1, -2)
        # x3d = self.crossAttention(x3d, self.dict.unsqueeze(0).repeat(B, 1, 1))
        # x3d = x3d.transpose(-1, -2)
        # x = x3d.reshape((B, C, H, W))



        if self.feat_bn0 is not None:
            # global_f = x
            # x = self.project_head(x)
            # part_list = []
            # B, C, H, W = x.size()
            # banch = H // 4
            # for i in range(4):
            #     part_feature = x[:, :, i * banch:(i + 1) * banch, :]
            #     part_feature = self.avgpool(part_feature)
            #     part_feature = part_feature.view(B, -1).unsqueeze(1)
            #     sim = torch.einsum('blc,nc->bln', [part_feature, self.PartMemory])
            #     sim = F.softmax(sim, -1)
            #     att_part_f = torch.einsum('bln,nc->bc', [sim, self.PartMemory]) + part_feature.squeeze(1)
            #     part_list.append(att_part_f)
            # part_f = torch.cat(part_list, dim=1)
            # # B, C, H, W = x.size()
            # # x3d = x.reshape((B, C, H * W)).permute(0, 2, 1)
            # # # return self.classifier(x3d.permute(0, 2, 1).mean(-1))
            # #
            # # sim = torch.einsum('bcl,nl->bcn', [x3d, self.PartMemory])
            # # sim = F.softmax(sim, -1)
            # #
            # # reconx = torch.einsum('bcn,nl->bcl', [sim, self.PartMemory]) + x3d
            # # x = reconx.permute(0, 2, 1).reshape((B,C,H,W))
            # global_f = self.avgpool(global_f)
            # global_f = global_f.view(global_f.size(0), -1)
            # if style:
            #     x = global_f + part_f
            # else:
            #     x = global_f
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            bn_x = self.feat_bn0(x)
        else:

            # B, C, H, W = x.size()
            # component_dim = 8
            # x3d = x.reshape((B, C, H * W)).permute(0, 2, 1)
            #
            # u, s, v = torch.svd(x3d)
            # main_component = torch.matmul(
            #     torch.matmul(u[:, :, :component_dim], torch.diag_embed(s[:, :component_dim])),
            #     v[:, :, :component_dim].mT)
            # x = main_component.permute(0, 2, 1).reshape((B, C, H, W))
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

            if self.training:
                return F.normalize(x)
            else:
                prob = self.classifier(x)
                return prob

            # return F.normalize(x)

        if self.training:

            return F.normalize(bn_x)
        else:
            return F.normalize(bn_x)

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
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
def resnet50(last_stride=1, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(last_stride, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), False)
    return model