import random
import argparse
from pathlib import Path
import numpy as np
import torch.backends.cudnn
import torchvision

from architectures import get_model
from zskd import ZeroShotKDClassification, ZeroShotKDHyperparams
from trainer.teacher_train import TeacherTrainerHyperparams, TeacherTrainer
from trainer.student_train import StudentTrainerHyperparams, StudentTrainer
from trainer.utils import transformer
from save_method import save_synthesized_images_labelwise
from architectures import ArchitectureFactory


ARCHITECTURES = [
    'lenet',
    'resnet',
    'resmlp',
    'rtdl-resnet'
]


def handle_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', 
        type=str,
        default='cifar10', 
        choices=['mnist', 'cifar10', 'cifar100'],
        help='Select the dataset'
    )

    parser.add_argument(
        '--teacher',
        type=str,
        choices=ARCHITECTURES,
        default='lenet'
    )
    
    parser.add_argument(
        '--student',
        type=str,
        choices=ARCHITECTURES,
        default='lenet'
    )

    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=24000,
        help='Number of DIs crafted per category'
    )
    parser.add_argument(
        '--beta', 
        type=list,
        default=[0.1, 1.], 
        help='Beta scaling vectors'
    )

    parser.add_argument(
        '--temperature', 
        type=int, 
        default=20,
        help='Temperature for distillation'
    )

    parser.add_argument(
        '--batch-size', 
        type=int,
        default=100, 
        help='batch_size'
    )

    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.01, 
        help='learning rate'
    )

    parser.add_argument(
        '--iterations', 
        type=int, 
        default=1500,
        help='number of iterations to optimize a batch of data impressions'
    )

    parser.add_argument(
        '--student-path', 
        type=Path,
        help='save path for student network'
    )

    parser.add_argument(
        '--teacher-path',
        type=Path,
    )

    parser.add_argument(
        '--synthetic-data-path',
        type=Path
    )

    parser.add_argument(
        '--real-data-path',
        type=Path
    )

    parser.add_argument(
        '--train-teacher', 
        action=argparse.BooleanOptionalAction, 
        help='Set flag to Train teacher network'
    )

    parser.add_argument(
        '--synthesize-data', 
        action=argparse.BooleanOptionalAction,
        help='generate synthesized images from ZSKD??'
    )

    parser.add_argument(
        '--train-student', 
        action=argparse.BooleanOptionalAction, 
        help='Set flag to Train student network'
    )

    return parser.parse_args()


def get_dataset(name: str, train: bool, data_path):
    match name:
        case 'mnist':
            train_transforms, test_transforms = transformer('mnist')
            if train:
                return torchvision.datasets.MNIST(
                    root=data_path, 
                    train=True, 
                    transform=train_transforms, 
                    download=True
                ) 
            else:
                return torchvision.datasets.MNIST(
                    root=data_path, 
                    train=False, 
                    transform=test_transforms, 
                    download=True
                )


def main():
    args = handle_args()
    # Deterministic Behavior
    seed = 0
    torch.cuda.set_device(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    teacher_factory = ArchitectureFactory(
        name=args.teacher, 
        input_dims=(1, 32, 32), 
        output_dims=10
    )
    
    student_factory = ArchitectureFactory(
        name=args.student, 
        input_dims=(1, 32, 32), 
        output_dims=10
    )

    # load teacher network
    if args.train_teacher:
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

    teacher = torch.load(args.teacher_path)
    student = student_factory.produce()

    # perform Zero-shot Knowledge distillation
    if args.synthesize_data:
        zskd_hyperparams = ZeroShotKDHyperparams(
            learning_rate=args.lr,
            iterations=args.iterations,
            batch_size=args.batch_size,
            temperature=args.temperature,
            num_samples=args.num_samples,
            beta=args.beta
        )
        zskd = ZeroShotKDClassification(
            teacher=teacher.eval(), 
            hyperparams=zskd_hyperparams,
            dimensions=(1, 32, 32),
            num_classes=10
        )

        print('\n[BEGIN] Zero Shot Knowledge Distillation For Image Classification')
        args.synthetic_data_path.parent.mkdir(parents=True, exist_ok=True)
        file_count_labelwise = np.zeros(10, dtype=int)

        for batch_idx, synthetic_batch in enumerate(zskd.synthesize_batch()):
            print(f"Batch [{batch_idx + 1}/{zskd_hyperparams.num_samples // zskd_hyperparams.batch_size}]")
            x = synthetic_batch[0].detach().cpu()
            y = synthetic_batch[1].argmax(dim=1).detach().cpu().numpy()
            
            save_synthesized_images_labelwise(
                inputs=x, 
                labels=y, 
                file_counts=file_count_labelwise, 
                root_dir=args.synthetic_data_path
            )
        
        print('[END] Zero Shot Knowledge Distillation For Image Classification')

    if args.train_student:
        # train student network

        # Set Hyperparameters
        st_hyperparams = StudentTrainerHyperparams(
            epochs=20,
            batch_size=256,
            teacher_temperature=20.0,
            optimizer=torch.optim.Adam(params=student.parameters())
        )

        # Synthetic Dataset
        synthetic_dataset = torchvision.datasets.ImageFolder(
            root=args.synthetic_data_path, 
            transform=transformer(args.dataset)[0]
        )

        # Test Dataset
        eval_dataset = torchvision.datasets.MNIST(
            root=args.real_data_path, 
            train=False, 
            transform=transformer(args.dataset)[1], 
            download=True
        )

        student_trainer = StudentTrainer(
            teacher=teacher.eval(),
            student=student,
            model_save_path=args.student_path,
            train_dataset=synthetic_dataset,
            test_dataset=eval_dataset,
            hyperparams=st_hyperparams
        )
        print('\n[BEGIN] Train Student Model')
        student_trainer.train()
        print('[END] Train Student Model')


if __name__ == "__main__":
    main()

