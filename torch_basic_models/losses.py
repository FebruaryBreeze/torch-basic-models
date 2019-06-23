import json
from pathlib import Path

import box
import jsonschema
import torch
import torch.nn as nn
import torch.nn.functional as functional

from .configs import CrossEntropyLossConfig
from .configs import L2LossConfig
from .configs import LabelSmoothingLossConfig


@box.register(tag='model')
class CrossEntropyLoss(nn.CrossEntropyLoss):
    with open(str(Path(__file__).parent / 'schema' / 'cross_entropy_loss_config.json')) as f:
        schema = json.load(f)

    def __init__(self, config: CrossEntropyLossConfig = None):
        super().__init__()

    @classmethod
    def factory(cls, config: dict = None):
        jsonschema.validate(config or {}, cls.schema)
        return cls(config=CrossEntropyLossConfig(config))


@box.register(tag='model')
class LabelSmoothingLoss(nn.Module):
    with open(str(Path(__file__).parent / 'schema' / 'label_smoothing_loss_config.json')) as f:
        schema = json.load(f)

    def __init__(self, config: LabelSmoothingLossConfig):
        super().__init__()
        self.smooth_ratio = config.smooth_ratio

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        num_classes = logits.size(1)
        positive = 1.0 - self.smooth_ratio
        negative = self.smooth_ratio / (num_classes - 1)

        one_hot = torch.zeros_like(logits).scatter(1, target.long().view((-1, 1)), 1)
        soft_target = one_hot * positive + (1.0 - one_hot) * negative

        log_probabilities = functional.log_softmax(logits, dim=1)
        return functional.kl_div(log_probabilities, soft_target, reduction='sum') / logits.size(0)

    @classmethod
    def factory(cls, config: dict = None):
        jsonschema.validate(config or {}, cls.schema)
        return cls(config=LabelSmoothingLossConfig(values=config))


@box.register(tag='model')
class L2Loss(nn.Module):
    def __init__(self, config: L2LossConfig = None):
        super().__init__()
        self.normalize = config.normalize

    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        if self.normalize:
            return (predict - target).pow(2).mean()
        else:
            return (predict - target).pow(2).sum(1).mean(0) / 2

    @classmethod
    def factory(cls, config: dict = None):
        jsonschema.validate(config or {}, cls.schema)
        return cls(config=L2LossConfig(config))

    with open(str(Path(__file__).parent / 'schema' / 'l2_loss_config.json')) as f:
        schema = json.load(f)


@box.register(tag='model')
class NormalizedL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        return (functional.normalize(predict) - functional.normalize(target)).pow(2).sum(1).mean() / 2

    @classmethod
    def factory(cls, config: dict = None):
        jsonschema.validate(config or {}, CrossEntropyLoss.schema)
        return cls()
