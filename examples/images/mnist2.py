import random
import argparse
from pathlib import Path
import numpy as np
import torch.backends.cudnn
import torchvision

from zskd import DataImpressionHyperparams
from zskd.synthesize import DataImpressionSynthesizer
from zskd.classifier_weights import extract_classifier_weights
from trainer.teacher_train import TeacherTrainerHyperparams, TeacherTrainer
from trainer.student_train import StudentTrainerHyperparams, StudentTrainer
from trainer.utils import transformer
from save_method import save_synthesized_images_labelwise
from architectures import ArchitectureFactory


ARCHITECTURES = [
    'lenet',
    'resnet',
    'resmlp',
]


def handle_args():
    """
    There are three stages for this project for Zero-Shot Knowledge Distillation: 
    1. pretrain: Obtain a pretrained model.
        Dependencies:
        - teacher network architecture
        - real data path for training the teacher
    2. synthesize: Generate Data Impressions
        Dependencies:
        - teacher network architecture
        - path to pretrained teacher model
        - data impression hyperparameters
        - synthetic data path to save the data
    3. distill: Use data impressions in knowledge distillation training for student network.
        Dependencies:
        - Real Data Path for evaluating the trained student network
        - Synthetic Data Path for performing knowledge distillation
        - teacher network architecture
        - path to pretrained teacher model
        - student network architecture
        - path to save student model
    """
    parser = argparse.ArgumentParser()
    # Architectures
    parser.add_argument('--teacher', type=str, choices=ARCHITECTURES, default='lenet')
    parser.add_argument('--student', type=str, choices=ARCHITECTURES, default='lenet')
    # Data Impression Hyperparameters
    parser.add_argument('--num-samples', type=int, default=24000, help='Number of DIs crafted per category')
    parser.add_argument('--beta', nargs='+', default=[0.1, 1.], help='Beta scaling vectors')
    parser.add_argument('--temperature', type=float, default=20.0, help='Temperature for distillation')
    parser.add_argument('--batch-size', type=int, default=100, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--iterations', type=int, default=1500, help='number of iterations to optimize a batch of data impressions')
    # File Paths
    parser.add_argument('--real-data-path', type=Path, help='dataset root')
    parser.add_argument('--student-path', type=Path, help='save path for student network')
    parser.add_argument('--teacher-path', type=Path, help='save path for teacher network')
    parser.add_argument('--synthetic-data-path', type=Path, help='save dir for synthetic data impressions')
    # Flags
    parser.add_argument('--pretrain', action=argparse.BooleanOptionalAction, help='Set flag to Train teacher network')
    parser.add_argument('--synthesize', action=argparse.BooleanOptionalAction, help='Use this flag to synthesize the data impressions')
    parser.add_argument('--distill', action=argparse.BooleanOptionalAction, help='Set flag to Train student network by distilling the teacher network')

    return parser.parse_args()


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

    input_dims = (1, 32, 32)
    num_labels = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_factory = ArchitectureFactory(name=args.teacher, input_dims=input_dims, output_dims=num_labels)
    student_factory = ArchitectureFactory(name=args.student, input_dims=input_dims, output_dims=num_labels)

    # load teacher network
    if args.pretrain:
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

    teacher: torch.nn.Module = torch.load(args.teacher_path, map_location=device)
    student = student_factory.produce()

    # perform Zero-shot Knowledge distillation
    if args.synthesize:
        
        synthesizer_hyperparams = DataImpressionHyperparams(
            learning_rate=args.lr,
            iterations=args.iterations,
            batch_size=args.batch_size,
            temperature=args.temperature,
            num_samples=args.num_samples,
            beta=args.beta
        )

        synthesizer = DataImpressionSynthesizer(
            teacher=teacher.eval(), 
            hyperparams=synthesizer_hyperparams,
            dimensions=input_dims,
            num_classes=num_labels,
            transfer_criterion=torch.nn.BCELoss(),
            extract_classifier_weights=extract_classifier_weights,
            device=device
        )

        print('\n[BEGIN] Zero Shot Knowledge Distillation For Image Classification')
        args.synthetic_data_path.parent.mkdir(parents=True, exist_ok=True)
        file_count_labelwise = np.zeros(10, dtype=int)        

        for batch_idx, synthetic_batch in enumerate(synthesizer.iter_synthesize()):
            print(f"Batch [{batch_idx + 1}/{synthesizer_hyperparams.num_samples // synthesizer_hyperparams.batch_size}]")
            x = synthetic_batch[0].detach().cpu()
            y = synthetic_batch[1].argmax(dim=1).detach().cpu().numpy()
            
            save_synthesized_images_labelwise(inputs=x, labels=y, 
                file_counts=file_count_labelwise, root_dir=args.synthetic_data_path)
        
        print('[END] Zero Shot Knowledge Distillation For Image Classification')

    if args.distill:
        # train student network

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

        # Set Hyperparameters
        st_hyperparams = StudentTrainerHyperparams(
            epochs=20,
            batch_size=256,
            teacher_temperature=args.temperature,
            optimizer=torch.optim.Adam(params=student.parameters())
        )

        student_trainer = StudentTrainer(
            teacher=teacher.eval(),
            student=student,
            model_save_path=args.student_path,
            train_dataset=synthetic_dataset,
            test_dataset=eval_dataset,
            hyperparams=st_hyperparams,
            device=device
        )
        print('\n[BEGIN] Train Student Model')
        student_trainer.train()
        print('[END] Train Student Model')


if __name__ == "__main__":
    main()

