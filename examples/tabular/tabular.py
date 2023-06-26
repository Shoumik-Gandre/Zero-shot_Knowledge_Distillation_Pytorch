import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch.backends.cudnn
from torch.utils.data import TensorDataset

from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from architectures import ResMLP
from zskd import ZeroShotKDClassification, ZeroShotKDHyperparams
from trainer.teacher_train import TeacherTrainer, TeacherTrainerHyperparams
from trainer.student_train import StudentTrainerHyperparams, StudentTrainer
from trainer.utils import transformer


def handle_args():
    parser = argparse.ArgumentParser()

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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}")

    args = handle_args()

        # Load Csv
    df = pd.read_csv(args.real_data_path, index_col=0)
    inputs = df[df.columns[df.columns != 'target']]
    labels = df['target']
    
    categorical_columns = [
        column for column in inputs.columns 
        if inputs[column].dtype != 'float64' and column != 'f_27'
    ]
    numerical_columns = [
        column for column in inputs.columns 
        if inputs[column].dtype == 'float64'
    ]

    preprocess = ColumnTransformer(
        [
            ('categorical_columns', OneHotEncoder(drop='first', sparse_output=False), categorical_columns),
            ('numerical_columns', FunctionTransformer(lambda col: col, feature_names_out='one-to-one'), numerical_columns)
        ],
        remainder='drop'
    )
    preprocess.set_output(transform='pandas')
    preprocess.fit(inputs, labels)
    inputs: pd.DataFrame = preprocess.transform(inputs)  # type: ignore
    x_train, x_eval, y_train, y_eval = train_test_split(inputs, labels, test_size=0.25, random_state=0)
    # Train Dataset
    train_dataset = TensorDataset(
        torch.from_numpy(x_train.to_numpy()).type(torch.float),
        torch.from_numpy(y_train.to_numpy()).type(torch.long)
    )
    
    # Test Dataset
    eval_dataset = TensorDataset(
        torch.from_numpy(x_eval.to_numpy()).type(torch.float),
        torch.from_numpy(y_eval.to_numpy()).type(torch.long)
    )

    # load teacher network
    if args.train_teacher:
        teacher = ResMLP(185, 2).to(device)
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

    teacher = torch.load(args.teacher_path, map_location=device).to(device)
    student = ResMLP(185, 2).to(device)

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
            dimensions=(185,),
            num_classes=2
        )

        print('\n[BEGIN] Zero Shot Knowledge Distillation For Tabular Classification')
        args.synthetic_data_path.mkdir(parents=True, exist_ok=True)
        with open(args.synthetic_data_path / 'x.out', 'w+') as x_file, open(args.synthetic_data_path / 'y.out', 'w+') as y_file:
            for synthetic_batch in zskd.synthesize_batch():
                x = synthetic_batch[0].detach().cpu().numpy()
                y = synthetic_batch[1].argmax(dim=1).detach().cpu().numpy()

                # Write Save code here for x and y
                np.savetxt(x_file, x)
                np.savetxt(y_file, y)
        
        print('[END] Zero Shot Knowledge Distillation For Tabular Classification')

    if args.train_student:
        # train student network

        # Set Hyperparameters
        st_hyperparams = StudentTrainerHyperparams(
            epochs=20,
            batch_size=256,
            teacher_temperature=20.0,
            optimizer=torch.optim.Adam(
                params=student.parameters(), 
                lr=0.01, 
                weight_decay=1e-4
            )
        )

        # Synthetic Dataset
        with open(args.synthetic_data_path / 'x.out', 'r') as x_file, open(args.synthetic_data_path / 'y.out', 'r') as y_file:
            synthetic_dataset = TensorDataset(
                torch.from_numpy(np.loadtxt(x_file)).type(torch.float),
                torch.from_numpy(np.loadtxt(y_file)).type(torch.long)
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
    # Deterministic Behavior
    seed = 0
    torch.cuda.set_device(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()


# Learning rate=0.0001: 0.182300
# No weight decay, lr=0.01, epochs=20 - 0.435600
# No weight decay, lr=0.01, epochs=100 - 0.433400


# Under the following hyperparameters:
# epochs=20
# lr=0.01
# batch size 256

# LeNet teacher accuracy - 0.988100
# ResMLP student accuracy = 0.309900

# ResMLP Teacher accuracy = 0.980500
# LeNet Student accuracy = 0.579000
