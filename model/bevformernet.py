import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("../../bev_simple_fusion")


from torchvision.models.resnet import resnet18
from efficientnet_pytorch import EfficientNet

EPS = 1e-4

from functools import partial
from einops.layers.torch import Rearrange, Reduce


from nets.ops.modules import MSDeformAttn, MSDeformAttn3D