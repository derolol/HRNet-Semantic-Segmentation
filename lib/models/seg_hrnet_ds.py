# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from models.seg_hrnet import HighResolutionNet

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)

class HighResolutionNetDS(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.hrnet = HighResolutionNet(config, **kwargs)
        self.hrnet.init_weights(config.MODEL.PRETRAINED)

    def forward(self, x):

        print(x.shape)
        x = F.interpolate(x, scale_factor=0.5)
        x = F.interpolate(x, scale_factor=0.5)
        x = F.interpolate(x, scale_factor=0.5)
        x = F.interpolate(x, scale_factor=0.5)
        x = self.hrnet(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = F.interpolate(x, scale_factor=2, mode='bilinear')

        return x

def get_seg_model(cfg, **kwargs):
    model = HighResolutionNetDS(cfg, **kwargs)
    return model
