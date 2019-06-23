import torch.nn as nn

from .layers import InplaceReLU, SELayer


def conv_bn_nl(in_channels: int, out_channels: int, kernel: int, stride: int, groups: int = 1,
               conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, no_linear=InplaceReLU):
    return nn.Sequential(
        conv_layer(in_channels, out_channels, kernel, stride, padding=(kernel - 1) // 2, groups=groups, bias=False),
        norm_layer(out_channels),
        no_linear() if no_linear is not None else nn.Sequential()
    )


def conv_bn_se_nl(in_channels: int, out_channels: int, kernel: int, stride: int, groups: int = 1,
                  conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, se_layer=SELayer, no_linear=InplaceReLU):
    return nn.Sequential(
        conv_layer(in_channels, out_channels, kernel, stride, padding=(kernel - 1) // 2, groups=groups, bias=False),
        norm_layer(out_channels),
        se_layer(out_channels) if se_layer is not None else nn.Sequential(),
        no_linear() if no_linear is not None else nn.Sequential()
    )
