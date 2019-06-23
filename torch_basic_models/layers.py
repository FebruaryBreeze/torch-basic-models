import torch
from torch import nn
from torch.nn import functional


class GlobalPooling(nn.Module):
    def forward(self, x: torch.Tensor):
        return functional.avg_pool2d(x, (int(x.size(2)), int(x.size(3))))


class Squeeze(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view((x.size(0), -1))


class UnSqueeze(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view((x.size(0), -1, 1, 1))


class View(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        # noinspection PyUnresolvedReferences
        return x.view((*x.shape[:min(x.dim(), self.dim) - 1], -1, *([1] * (self.dim - x.dim()))))


class Classifier(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor):
        if self.dropout:
            x = functional.dropout(x, p=self.dropout, training=self.training, inplace=True)
        return self.linear(x)


class InplaceReLU(nn.ReLU):
    def __init__(self):
        super().__init__(inplace=True)


class InplaceReLU6(nn.ReLU6):
    def __init__(self):
        super().__init__(inplace=True)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class HSigmoid(nn.Module):
    """
    hard sigmoid
    refer to MobileNetV3 https://arxiv.org/abs/1905.02244
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return functional.relu6(x + 3., inplace=self.inplace) / 6.


class HSwish(nn.Module):
    """
    hard swish
    refer to MobileNetV3 https://arxiv.org/abs/1905.02244
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.h_sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.h_sigmoid(x)


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Networks
    refer to https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_channels: int, reduction: int = 4, no_linear=InplaceReLU, sigmoid=nn.Sigmoid, bias=False):
        super().__init__()
        self.se = nn.Sequential(
            GlobalPooling(),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=bias),
            no_linear(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=bias),
            sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)
