import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Callable


ModuleType = Union[str, Callable[..., nn.Module]]
ModuleType0 = Union[str, Callable[[], nn.Module]]


class ReGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError('The size of the last dimension must be even.')
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


def make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == 'ReGLU':
            cls = ReGLU
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError:
                raise ValueError(
                    f'There is no such module as {module_type} in torch.nn'
                )
        return cls(*args)
    else:
        return module_type(*args)


class ResNet(nn.Module):
    """The ResNet model used in the paper "Revisiting Deep Learning Models for Tabular Data" [1].

    **Input shape**: ``(n_objects, n_features)``.

    The following scheme describes the architecture:

    .. code-block:: text

        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)

                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)

          Head: (in) -> Norm -> Activation -> Linear -> (out)

    Attributes:
        blocks: the main blocks of the model (`torch.nn.Sequential` of `ResNet.Block`)
        head: (optional) the last module (`ResNet.Head`)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                d_in=x.shape[1],
                d_out=1,
                n_blocks=2,
                d_main=3,
                d_hidden=4,
                dropout_first=0.25,
                dropout_second=0.0,
            )
            assert module(x).shape == (len(x), 1)

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
            self,
            *,
            d_main: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType0,
            activation: ModuleType0,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        """The output module of `ResNet`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType0,
            activation: ModuleType0,
        ) -> None:
            super().__init__()
            self.normalization = make_nn_module(normalization, d_in)
            self.activation = make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType0,
        activation: ModuleType0,
    ) -> None:
        """
        Note:
            Use the `make_baseline` method instead of the constructor unless you need
            more control over the architecture.
        """
        super().__init__()

        self.first_layer = nn.Linear(d_in, d_main)
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = (
            None
            if d_out is None
            else ResNet.Head(
                d_in=d_main,
                d_out=d_out,
                bias=True,
                normalization=normalization,
                activation=activation,
            )
        )

    @classmethod
    def make_baseline(
        cls,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
    ) -> 'ResNet':
        """A simplified constructor for building baseline ResNets.

        Features:

        * all activations are ``ReLU``
        * all normalizations are ``BatchNorm1d``

        Args:
            d_in: the input size
            d_out: the output size of `ResNet.Head`. If `None`, then the output of
                ResNet will be the output of the last block, i.e. the model will be
                backbone-only.
            n_blocks: the number of blocks
            d_main: the input size (or, equivalently, the output size) of each block
            d_hidden: the output size of the first linear layer in each block
            dropout_first: the dropout rate of the first dropout layer in each block.
            dropout_second: the dropout rate of the second dropout layer in each block.
                The value `0.0` is a good starting point.
        Return:
            resnet
        """
        return cls(
            d_in=d_in,
            d_out=d_out,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization='BatchNorm1d',
            activation='ReLU',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1, -1)
        print(f" Shape of S: {x.shape}")
        print(f" Shape of first layer: {self.first_layer.weight.shape}")
        x = self.first_layer(x)
        x = self.blocks(x)
        if self.head is not None:
            x = self.head(x)
        return x