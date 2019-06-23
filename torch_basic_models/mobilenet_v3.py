"""
MobileNetV3
refer to paper https://arxiv.org/abs/1905.02244
also refer to https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
"""

import functools
import json
from pathlib import Path

import box
import jsonschema
import torch.nn as nn
import torch_utils

from .batch_norm_2d import load_default_batch_norm_2d
from .blocks import conv_bn_nl, conv_bn_se_nl
from .configs import MobileNetV3Config
from .layers import HSigmoid, HSwish, Squeeze, GlobalPooling, InplaceReLU, SELayer, Classifier


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, exp_channels: int,
                 se: bool = False, no_linear=InplaceReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()

        se_layer = functools.partial(SELayer, reduction=4, sigmoid=HSigmoid) if se else None  # h-sigmoid in figure 4
        self.blocks = nn.Sequential(
            # pixel wise
            conv_bn_nl(in_channels, exp_channels, kernel=1, stride=1, norm_layer=norm_layer, no_linear=no_linear),
            # depth wise
            conv_bn_se_nl(exp_channels, exp_channels, kernel, stride, groups=exp_channels,
                          norm_layer=norm_layer, se_layer=se_layer, no_linear=no_linear),
            # pixel wise
            conv_bn_nl(exp_channels, out_channels, kernel=1, stride=1, norm_layer=norm_layer, no_linear=None),
        )

        self.residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.residual:
            return self.blocks(x) + x
        else:
            return self.blocks(x)


def make_divisible(x: float, divisible_by: int = 8):
    return int(round(x / divisible_by) * divisible_by)


def build_blocks(in_channels: int, stride_list: list, width_multiple: float, norm_layer=nn.BatchNorm2d):
    blocks_setting = [
        # kernel, exp, out, se, no linear, stride
        [3, 16, 16, False, InplaceReLU, 1],
        [3, 64, 24, False, InplaceReLU, stride_list[1]],
        [3, 72, 24, False, InplaceReLU, 1],
        [5, 72, 40, True, InplaceReLU, stride_list[2]],
        [5, 120, 40, True, InplaceReLU, 1],
        [5, 120, 40, True, InplaceReLU, 1],
        [3, 240, 80, False, HSwish, stride_list[3]],
        [3, 200, 80, False, HSwish, 1],
        [3, 184, 80, False, HSwish, 1],
        [3, 184, 80, False, HSwish, 1],
        [3, 480, 112, True, HSwish, 1],
        [3, 672, 112, True, HSwish, 1],
        [5, 672, 112, True, HSwish, 1],
        [5, 672, 160, True, HSwish, stride_list[4]],
        [5, 960, 160, True, HSwish, 1],
    ]
    blocks = []

    for kernel, exp, out, se, no_linear, stride in blocks_setting:
        out_channels = make_divisible(out * width_multiple)
        exp_channels = make_divisible(exp * width_multiple)
        blocks.append(InvertedResidual(
            in_channels, out_channels, kernel, stride, exp_channels,
            se=se, no_linear=no_linear, norm_layer=norm_layer
        ))
        in_channels = out_channels
    return blocks, in_channels


@box.register(tag='model')
class MobileNetV3(nn.Module):
    def __init__(self, config: MobileNetV3Config):
        super().__init__()

        in_channels = 16
        last_channels = 1280
        norm_layer = load_default_batch_norm_2d()

        first_block = conv_bn_nl(in_channels=3, out_channels=in_channels, kernel=3, stride=config.stride_list[0],
                                 norm_layer=norm_layer, no_linear=HSwish)
        blocks, out_channels = build_blocks(in_channels, config.stride_list, config.width_multiple, norm_layer)

        last_conv = make_divisible(960 * config.width_multiple)
        last_block = nn.Sequential(
            conv_bn_nl(out_channels, last_conv, kernel=1, stride=1, no_linear=HSwish, norm_layer=norm_layer),
            GlobalPooling(),
            Squeeze(),
            nn.Linear(last_conv, last_channels),
            HSwish(),
            Classifier(last_channels, config.feature_dim, dropout=config.dropout_ratio)
        )

        self.blocks = nn.Sequential(first_block, *blocks, last_block)
        torch_utils.initialize_weights(self)

    def forward(self, x):
        return self.blocks(x)

    @classmethod
    def factory(cls, config: dict = None):
        jsonschema.validate(config or {}, cls.schema)
        return cls(config=MobileNetV3Config(values=config))

    with open(str(Path(__file__).parent / 'schema' / 'mobilenet_v3_config.json')) as f:
        schema = json.load(f)
