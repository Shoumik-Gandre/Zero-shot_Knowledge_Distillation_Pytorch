import torch
from architectures.resmlp import ResMLP
from architectures.lenet import LeNet5
from architectures.resnet import ResNet18

import rtdl

def get_model(name: str) -> torch.nn.Module:
    match name:
        case 'lenet':
            return LeNet5()
        case 'resmlp':
            return ResMLP(32*32, 10)
        case 'resnet':
            return ResNet18(10, 2) 
        case 'rtdl-resnet':
            return rtdl.ResNet.make_baseline(
                d_in=32*32,
                d_main=128,
                d_hidden=256,
                dropout_first=0.2,
                dropout_second=0.0,
                n_blocks=2,
                d_out=10,
            )
    raise ValueError()