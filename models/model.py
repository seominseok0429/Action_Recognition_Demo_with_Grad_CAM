from __future__ import division
import torch
from torch import nn
from models import resnext
import pdb

def generate_model():

    from models.resnext import get_fine_tuning_parameters
    model = resnext.resnet101(
            num_classes=51,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=64,
            input_channels=3)
    

    model = model.cuda()
    model = nn.DataParallel(model)
    return model

