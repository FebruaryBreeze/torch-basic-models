import json
import math
from pathlib import Path

import box
import jsonschema
import torch
import torch.nn as nn
import torch.nn.functional as f

from .configs import ArcFaceConfig


@box.register(tag='model')
class ArcFace(nn.Module):
    """
    ArcFace loss for face recognition
    refer to https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py#L10
    """
    with open(str(Path(__file__).parent / 'schema' / 'arc_face_config.json')) as f:
        schema = json.load(f)

    def __init__(self, config: ArcFaceConfig):
        super().__init__()
        self.feature_dim = config.feature_dim
        self.num_classes = config.num_classes
        self.s = config.s
        self.m = config.m

        self.weight = nn.Parameter(torch.Tensor(self.num_classes, self.feature_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, feature: torch.Tensor, label: torch.Tensor):
        cosine = f.linear(f.normalize(feature), f.normalize(self.weight))
        sine = (1.0 - cosine.pow(2)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine, dtype=torch.uint8)  # label to byte tensor
        one_hot.scatter_(1, label.view((-1, 1)).long(), 1)

        output = torch.where(one_hot, phi, cosine)
        output *= self.s

        return output

    @classmethod
    def factory(cls, config: dict):
        jsonschema.validate(config or {}, cls.schema)
        return cls(config=ArcFaceConfig(values=config))
