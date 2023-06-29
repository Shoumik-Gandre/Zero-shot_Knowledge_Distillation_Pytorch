from typing import Tuple
import torch
from architectures.resmlp import ResMLP
from architectures.lenet import LeNet5
from architectures.resnet import ResNet18
from architectures.factory import ArchitectureFactory


# def get_model(name: str, input_dims: Tuple[int, ...], output_dims: int) -> torch.nn.Module:
#     match name:
#         case 'lenet':
#             return LeNet5()
#         case 'resmlp':
#             return ResMLP(sum(input_dims), output_dims)
#         case 'resnet':
#             return ResNet18(output_dims, input_dims[0]) 
#         case 'rtdl-resnet':
#             return rtdl.ResNet.make_baseline(
#                 d_in=sum(input_dims),
#                 d_main=128,
#                 d_hidden=256,
#                 dropout_first=0.2,
#                 dropout_second=0.0,
#                 n_blocks=2,
#                 d_out=output_dims,
#             )
#     raise ValueError()