from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
import operator
from typing import Mapping, Tuple
from torch import nn

from architectures.lenet import LeNet5
from architectures.resmlp import ResMLP
from architectures.resnet import ResNet18
from architectures.rtdl_resnet import ResNet as RTDLResNet


@dataclass
class ArchitectureFactory:
    input_dims: Tuple[int, ...]
    output_dims: int
    name: str

    def produce(self) -> nn.Module:
        match self.name:
            case 'lenet':
                return LeNet5()
            case 'resmlp':
                return ResMLP(reduce(operator.mul, (self.input_dims)), self.output_dims)
            case 'resnet':
                return ResNet18(self.output_dims, self.input_dims[0]) 
            case 'rtdl-resnet':
                return RTDLResNet.make_baseline(
                    d_in=reduce(operator.mul, (self.input_dims)),
                    d_main=128,
                    d_hidden=256,
                    dropout_first=0.2,
                    dropout_second=0.0,
                    n_blocks=2,
                    d_out=self.output_dims,
                )
        raise ValueError()