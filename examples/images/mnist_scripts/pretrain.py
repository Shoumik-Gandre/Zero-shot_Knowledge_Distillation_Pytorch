import random
import argparse
from pathlib import Path
import numpy as np
import torch.backends.cudnn
import torchvision

from zskd import ZeroShotKDHyperparams
from zskd.core2 import ZeroShotKDClassification
from zskd.classifier_weights import extract_classifier_weights
from trainer.teacher_train import TeacherTrainerHyperparams, TeacherTrainer
from trainer.student_train import StudentTrainerHyperparams, StudentTrainer
from trainer.utils import transformer
from save_method import save_synthesized_images_labelwise
from architectures import ArchitectureFactory


def main():
    args = argparse.Namespace()
    # Deterministic Behavior
    seed = 0
    torch.cuda.set_device(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_dims = (1, 32, 32)
    num_labels = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_factory = ArchitectureFactory(
        name=args.teacher, 
        input_dims=input_dims, 
        output_dims=num_labels
    )
    
    student_factory = ArchitectureFactory(
        name=args.student, 
        input_dims=input_dims, 
        output_dims=num_labels
    )

    # load teacher network
    teacher = teacher_factory.produce()

    # Test Dataset
    train_dataset = torchvision.datasets.MNIST(
            root=args.real_data_path, 
            train=True, 
            transform=transformer(args.dataset)[0], 
            download=True
    )
    # Test Dataset
    eval_dataset = torchvision.datasets.MNIST(
            root=args.real_data_path, 
            train=False, 
            transform=transformer(args.dataset)[1], 
            download=True
    )

    teacher_trainer_hyperparams = TeacherTrainerHyperparams(
        epochs=10, 
        batch_size=256, 
        optimizer=torch.optim.Adam(teacher.parameters())
    )
    
    teacher_trainer = TeacherTrainer(
        model=teacher,
        hyperparams=teacher_trainer_hyperparams,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_path=args.teacher_path
    )
    
    print('[BEGIN] Train Teacher Model')
    teacher_trainer.train()
    print('[END] Train Teacher Model')


if __name__ == "__main__":
    main()

