import torch
from architectures.resmlp import ResMLP
from architectures.lenet import LeNet5
from architectures.resnet import ResNet18

def get_model(name: str) -> torch.nn.Module:
    match name:
        case 'lenet':
            return LeNet5()
        case 'resmlp':
            return ResMLP(32*32, 10)
        case 'resnet':
            return ResNet18(10, 1) 
    raise ValueError()