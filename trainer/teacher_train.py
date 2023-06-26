from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import nn


@dataclass
class TeacherTrainerHyperparams:
    epochs: int
    batch_size: int
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None


class TeacherTrainer:
    """Train and save trainer model"""

    def __init__(
        self, 
        model: nn.Module,
        hyperparams: TeacherTrainerHyperparams,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        model_path: Path,
        device=torch.device('cuda')
    ):

        self.acc = 0

        self.model_path = model_path
        
        self.net = model.to(device)
        self.hyperparams = hyperparams

        self.train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=hyperparams.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        self.eval_dataloader = DataLoader(
            dataset=eval_dataset, 
            batch_size=hyperparams.batch_size, 
            num_workers=0
        )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

    def train_step(self, epoch):
        """Train step for teacher"""

        self.net.train()
        for i, (images, labels) in enumerate(pbar := tqdm(self.train_dataloader)):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.hyperparams.optimizer.zero_grad()
            output = self.net(images)
            loss = self.criterion(output, labels)

            loss.backward()
            self.hyperparams.optimizer.step()
            if self.hyperparams.lr_scheduler:
                self.hyperparams.lr_scheduler.step()
            pbar.set_description(f"Epoch {epoch} | Loss {loss.item():.4f}")

    def eval_step(self):
        """Validation step for the teacher"""
        self.net.eval()
        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.eval_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.net(images)
                avg_loss += self.criterion(output, labels).sum()
                pred = output.argmax(dim=1)
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

        avg_loss /= len(self.eval_dataloader.dataset) # type: ignore
        self.acc = float(total_correct) / len(self.eval_dataloader.dataset) # type: ignore

        print('Test Avg. Loss: %f, Accuracy: %f' %
              (avg_loss.item(), self.acc))  # type: ignore

    def train(self):
        """Trainer run function"""

        for epoch in range(1, self.hyperparams.epochs+1):
            self.train_step(epoch)
            self.eval_step()
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net, self.model_path)
