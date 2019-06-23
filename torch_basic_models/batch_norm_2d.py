from typing import Type

import box
import torch.nn as nn

_default_batch_norm_2d_key = 'default_batch_norm_2d'


def load_default_batch_norm_2d() -> Type[nn.BatchNorm2d]:
    return box.load(name=_default_batch_norm_2d_key, default=nn.BatchNorm2d)


def set_default_batch_norm_2d(batch_norm_2d: Type[nn.BatchNorm2d]):
    return box.register(obj=batch_norm_2d, name=_default_batch_norm_2d_key)


def reset_default_batch_norm_2d():
    box.erase(name=_default_batch_norm_2d_key)
