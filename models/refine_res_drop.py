# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.model_helper import FpnAdapter, weights_init
from layers.dropblock.dropblock import DropBlock2D
from layers.dropblock.scheduler import LinearScheduler

def add_extras(size, in_channel, batch_norm=False):
    # Extra layers added to resnet for feature scaling
    layers = []
    layers += [nn.Conv2d(in_channel, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)]
    return layers


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, dropblock, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

class BottleneckDrop(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, dropblock, stride=1, downsample=None):
        super(BottleneckDrop, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropblock = dropblock

    def forward(self, x):
        residual = x

        out = self.dropblock(self.conv1(x))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropblock(self.conv2(out))
        out = self.bn2(out)
        out = self.relu(out)

        out = self.dropblock(self.conv3(out))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        residual = self.dropblock(residual)

        out += residual
        out = self.relu(out)

        return out

class RefineResnetDrop(nn.Module):
    def __init__(self, block, block_drop, num_blocks, size, drop_prob=0.1, block_size=7, nr_steps=2e3):
        super(RefineResnetDrop, self).__init__()
        self.inplanes = 64
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=nr_steps  # 2e3
        )
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_drop, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_drop, 512, num_blocks[3], stride=2)
        self.inchannel = block.expansion * 512
        self.extras = nn.ModuleList(add_extras(str(size), self.inchannel))
        self.smooth1 = nn.Conv2d(
            self.inchannel, 512, kernel_size=3, stride=1, padding=1)
        self.fpn = FpnAdapter([512, 1024, 512, 256], 4)
        self._init_modules()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.dropblock, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.dropblock))

        return nn.Sequential(*layers)

    def _init_modules(self):
        self.extras.apply(weights_init)
        self.smooth1.apply(weights_init)

    def forward(self, x):
        self.dropblock.step()  # increment number of iterations
        # Bottom-up
        odm_sources = list()
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        x = c5
        c5_ = self.smooth1(c5)
        arm_sources = [c3, c4, c5_]
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                arm_sources.append(x)
        odm_sources = self.fpn(arm_sources)
        return arm_sources, odm_sources


def RefineResnet50Drop(size, channel_size='48', drop_prob=0.1, block_size=7, nr_steps=5e3):
    return RefineResnetDrop(Bottleneck, BottleneckDrop, [3, 4, 6, 3], size, drop_prob=drop_prob, block_size=block_size, nr_steps=nr_steps)




