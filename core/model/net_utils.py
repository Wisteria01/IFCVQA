# --------------------------------------------------------
# IFCVQA（Image Feature Cropping VQA）
# Licensed under The MIT License [see LICENSE for details]
# Written by Ziteng Xu
# --------------------------------------------------------

import torch.nn as nn
import torch


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0.8, use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.GELU()

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class CNN(nn.Module):
    def __init__(self, num_classes, dropout_r=0., use_relu=True):
        super(CNN, self).__init__()
        self.use_relu = use_relu

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, num_classes)  # 根据输入图像尺寸调整

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x
         