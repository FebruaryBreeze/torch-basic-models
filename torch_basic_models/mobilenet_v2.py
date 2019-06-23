import json
from pathlib import Path

import box
import jsonschema
import torch.nn as nn
import torch_utils

from . import layers
from .batch_norm_2d import load_default_batch_norm_2d
from .blocks import conv_bn_nl
from .configs import MobileNetV2Config


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, expansion: float,
                 no_linear=layers.InplaceReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()

        exp_channels = int(in_channels * expansion)
        self.blocks = nn.Sequential(
            # pixel wise
            conv_bn_nl(in_channels, exp_channels, kernel=1, stride=1, norm_layer=norm_layer, no_linear=no_linear),
            # depth wise
            conv_bn_nl(exp_channels, exp_channels, kernel, stride, groups=exp_channels, norm_layer=norm_layer,
                       no_linear=no_linear),
            # pixel wise
            conv_bn_nl(exp_channels, out_channels, kernel=1, stride=1, norm_layer=norm_layer, no_linear=None),
        )

        self.residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.residual:
            return self.blocks(x) + x
        else:
            return self.blocks(x)


def build_blocks(in_channels: int, stride_list: list, t: float, width_multiple: float, no_linear, norm_layer):
    block_settings = [
        # t, c, n, s
        [1, 16, 1, 1],
        [t, 24, 2, stride_list[1]],
        [t, 32, 3, stride_list[2]],
        [t, 64, 4, stride_list[3]],
        [t, 96, 3, 1],
        [t, 160, 3, stride_list[4]],
        [t, 320, 1, 1],
    ]

    blocks = []
    for expansion, channels, times, first_stride in block_settings:
        out_channels = int(channels * width_multiple)

        for i in range(times):
            stride = first_stride if i == 0 else 1
            blocks.append(InvertedResidual(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expansion=expansion,
                kernel=3,
                norm_layer=norm_layer,
                no_linear=no_linear
            ))

            in_channels = out_channels
    return blocks, in_channels


@box.register(tag='model')
class MobileNetV2(nn.Module):
    def __init__(self, config: MobileNetV2Config):
        super().__init__()
        width_multiple = config.width_multiple
        norm_layer = load_default_batch_norm_2d()
        no_linear = getattr(layers, config.no_linear)

        # building first layer
        in_channels = int(32 * width_multiple)
        first_block = conv_bn_nl(3, in_channels, kernel=3, stride=config.stride_list[0],
                                 norm_layer=norm_layer, no_linear=no_linear)

        blocks, out_channels = build_blocks(
            in_channels=in_channels,
            stride_list=config.stride_list,
            t=config.expansion_ratio,
            width_multiple=config.width_multiple,
            no_linear=no_linear,
            norm_layer=norm_layer
        )

        # building inverted residual blocks
        last_channels = int(1280 * max(width_multiple, 1.0))
        last_block = nn.Sequential(
            conv_bn_nl(out_channels, last_channels, 1, stride=1, norm_layer=norm_layer, no_linear=no_linear),
            layers.GlobalPooling(),
            layers.Squeeze(),
            layers.Classifier(last_channels, config.feature_dim, dropout=config.dropout_ratio)
        )

        self.blocks = nn.Sequential(first_block, *blocks, last_block)
        torch_utils.initialize_weights(self, norm_layer=norm_layer)

    def forward(self, x):
        return self.blocks(x)

    @classmethod
    def factory(cls, config: dict = None):
        jsonschema.validate(config or {}, cls.schema)
        return cls(config=MobileNetV2Config(values=config))

    with open(str(Path(__file__).parent / 'schema' / 'mobilenet_v2_config.json')) as f:
        schema = json.load(f)
