import torch


class SimpleMLP(torch.nn.Module):

    def __init__(self, in_dims, out_dims) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(in_dims, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, out_dims)
        )
    
    def forward(self, x):
        return self.sequential(x)
        