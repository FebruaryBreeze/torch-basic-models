from . import layers
from .__version__ import __version__
from .batch_norm_2d import set_default_batch_norm_2d, load_default_batch_norm_2d, reset_default_batch_norm_2d
from .face import ArcFace
from .losses import CrossEntropyLoss, LabelSmoothingLoss, L2Loss, NormalizedL2Loss
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .resnet import ResNet
