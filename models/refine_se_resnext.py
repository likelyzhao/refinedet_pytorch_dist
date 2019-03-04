# -*- coding: utf-8 -*-
# Written by yq_yao
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.model_helper import FpnAdapter, weights_init


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


class SEBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(SEBottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(planes * 4, planes // 4)
        self.fc2 = nn.Linear(planes // 4, planes * 4)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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

        se = self.global_avg(out)
        se = se.view(se.size(0), -1)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        se = se.view(se.size(0), se.size(1), 1, 1)

        out = out * se.expand_as(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RefineSeResneXt(nn.Module):
    def __init__(self, block, num_blocks, size, head7x7=False, baseWidth=4, cardinality=32):
        super(RefineSeResneXt, self).__init__()
        self.inplanes = 64
        self.cardinality = cardinality
        self.baseWidth = baseWidth

        self.head7x7 = head7x7

        if self.head7x7:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, groups=8, bias=False)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, groups=16, bias=False)
            self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.inchannel = block.expansion * 512
        self.extras = nn.ModuleList(add_extras(str(size), self.inchannel))
        self.smooth1 = nn.Conv2d(
            self.inchannel, 512, kernel_size=3, stride=1, padding=1)
        self.fpn = FpnAdapter([512, 1024, 512, 256], 4)
        self._init_modules()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct SE_ResNeXt
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)

    def _init_modules(self):
        self.extras.apply(weights_init)
        self.smooth1.apply(weights_init)

    def forward(self, x):
        # Bottom-up
        odm_sources = list()
        if self.head7x7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
        c1 = self.maxpool(x)
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


def RefineSeResneXt50(size, channel_size='48'):
    return RefineSeResneXt(SEBottleneck, [3, 4, 6, 3], size)

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = RefineSeResneXt50(size=300)
    print(model)
    with torch.no_grad():
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        model.cuda()
        model(x.cuda())
