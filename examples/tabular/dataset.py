from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset
import torch
import pandas as pd


class TabularDataset(Dataset):

    def __init__(self, path: Path, preprocessor) -> None:
        super().__init__()
        self.df = pd.read_csv(path)
        self.preprocessor = preprocessor
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.preprocessor(self.df).iloc[index, :-1].to_numpy())
        y = torch.from_numpy(self.df.iloc[index, -1])
        return x, y
    
    def __len__(self):
        return self.df.shape[0]