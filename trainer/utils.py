from typing import Tuple
import torch
import torchvision.transforms as transforms

class KeepChannelsTransform:
    """Rotate by one of the given angles."""

    def __init__(self, channels: Tuple[int, ...]):
        self.channels = channels

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.channels, :]
    

def transformer(dataset) -> Tuple[transforms.Compose, transforms.Compose]:
    if dataset == 'mnist':
        trans=transforms.Compose([  
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            KeepChannelsTransform((0,)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_trans, test_trans = trans, trans
        
    elif dataset == 'cifar10' or dataset == 'cifar100':
        train_trans = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        test_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        
    else:
        raise ValueError("Dataset does not have transforms")
    
    return train_trans, test_trans


def adjust_learning_rate(optimizer, epoch):
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr