from dataclasses import dataclass
from pathlib import Path

import torchvision
import torch
from torch.autograd import Variable
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm 


@dataclass
class StudentTrainerHyperparams:
    epochs: int
    batch_size: int
    teacher_temperature: float
    optimizer: torch.optim.Optimizer


class StudentTrainer:

    def __init__(
            self,
            teacher: torch.nn.Module,
            student: torch.nn.Module,
            model_save_path: Path,
            train_dataset: Dataset,
            test_dataset: Dataset,
            hyperparams: StudentTrainerHyperparams,
            device = torch.device('cuda')
        ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.save_path = model_save_path
        model_save_path.parent.mkdir(parents=True, exist_ok=True)

        self.train_dataloader = DataLoader(
            dataset=train_dataset, 
            batch_size=hyperparams.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        self.eval_dataloader = DataLoader(
            dataset=test_dataset, 
            batch_size=hyperparams.batch_size, 
            num_workers=0
        )
        self.hyperparams = hyperparams
        self.criterion_train = torch.nn.BCELoss()
        self.criterion_test = torch.nn.CrossEntropyLoss()
        self.acc = 0.0
        self.device = device

    def train_step(self):
        self.student.train()
        self.teacher.eval()
        for i, (images, labels) in enumerate(pbar := tqdm(self.train_dataloader)):
            images = images.to(self.device) 
            labels = labels.to(self.device)
            teacher_output = F.softmax(self.teacher(images) / self.hyperparams.teacher_temperature, dim=1)
            student_output = F.softmax(self.student(images), dim=1)

            self.hyperparams.optimizer.zero_grad()
            loss = self.criterion_train(student_output, teacher_output.detach())
            loss.backward()
            self.hyperparams.optimizer.step()
            pbar.set_description(f'Loss {loss.item()}')

    def eval_step(self):
        self.student.eval()
        total_correct = 0
        avg_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.eval_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = F.softmax(self.student(images), dim=1)

                avg_loss += self.criterion_test(outputs, labels).item()
                prediction = outputs.argmax(dim=1)
                total_correct += prediction.eq(labels.data.view_as(prediction)).sum()

        avg_loss /= len(self.eval_dataloader.dataset) # type: ignore
        acc = float(total_correct) / len(self.eval_dataloader.dataset) # type: ignore
        print('Test Avg. Loss: %f, Accuracy: %f' %
              (avg_loss.item(), acc)) # type: ignore

    def train(self):
        for epoch in range(1, self.hyperparams.epochs+1):
            print(f"Epoch [{epoch}/{self.hyperparams.epochs}]")
            self.train_step()
            self.eval_step()
        torch.save(self.student, self.save_path)
