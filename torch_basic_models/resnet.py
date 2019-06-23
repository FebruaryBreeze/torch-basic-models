import json
from pathlib import Path
from typing import List
from typing import Type

import box
import jsonschema
import torch_utils
from torch import nn
from torch.nn import functional

from .batch_norm_2d import load_default_batch_norm_2d
from .blocks import conv_bn_nl
from .configs import ResNetConfig
from .layers import Classifier, Squeeze, GlobalPooling, InplaceReLU


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, norm_layer: Type[nn.BatchNorm2d]):
        super().__init__()

        mid_channels = out_channels // 4
        no_linear = InplaceReLU
        self.blocks = nn.Sequential(
            conv_bn_nl(in_channels, mid_channels, kernel=1, stride=1, norm_layer=norm_layer, no_linear=no_linear),
            conv_bn_nl(mid_channels, mid_channels, kernel=3, stride=stride, norm_layer=norm_layer, no_linear=no_linear),
            conv_bn_nl(mid_channels, out_channels, kernel=1, stride=1, norm_layer=norm_layer, no_linear=None),
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels),
            )

    def forward(self, x):
        out = self.blocks(x) + self.shortcut(x)
        out = functional.relu(out, inplace=True)
        return out


def build_blocks(in_channels: int, layers_list: List[int], stride_list: List[int], norm_layer: Type[nn.BatchNorm2d]):
    blocks = []
    channels_list = [256, 512, 1024, 2048]
    for layers, out_channels, first_stride in zip(layers_list, channels_list, stride_list):
        for i in range(layers):
            stride = first_stride if i == 0 else 1
            blocks.append(Bottleneck(in_channels, out_channels, stride=stride, norm_layer=norm_layer))
            in_channels = out_channels
    return blocks, in_channels


@box.register(tag='model')
class ResNet(nn.Module):
    with open(str(Path(__file__).parent / 'schema' / 'resnet_config.json')) as f:
        schema = json.load(f)

    def __init__(self, config: ResNetConfig):
        super().__init__()

        in_channels = 64
        stride_list = config.stride_list
        norm_layer = load_default_batch_norm_2d()

        first_block = nn.Sequential(
            conv_bn_nl(3, in_channels, kernel=7, stride=stride_list[0], norm_layer=norm_layer, no_linear=InplaceReLU),
            nn.MaxPool2d(kernel_size=3, stride=stride_list[1], padding=1)
        )

        blocks, out_channels = build_blocks(
            in_channels=in_channels,
            layers_list=config.layers_list,
            stride_list=stride_list[2:],
            norm_layer=norm_layer
        )

        last_block = nn.Sequential(
            GlobalPooling(),
            Squeeze(),
            Classifier(out_channels, config.feature_dim, dropout=config.dropout_ratio)
        )

        self.blocks = nn.Sequential(first_block, *blocks, last_block)
        torch_utils.initialize_weights(self, norm_layer=norm_layer)

    def forward(self, x):
        return self.blocks(x)

    @classmethod
    def factory(cls, config: dict = None):
        jsonschema.validate(config or {}, cls.schema)
        return cls(config=ResNetConfig(values=config))
