from dataclasses import dataclass
from typing import Iterator, Tuple, Callable

from tqdm import tqdm
import torch
import torch.nn.functional as F

from zskd.class_sim_matrix import compute_class_similarity_matrix
from zskd.hyperparams import DataImpressionHyperparams
from zskd.classifier_weights import extract_classifier_weights2


ExtractClassifierWeightsFn = Callable[[torch.nn.Module], torch.Tensor]


@dataclass
class DataImpressionSynthesizer:
    """Zero Shot Knowledge Distillation for a classification task
    input:
    teacher - the pre-trained model that we want to extract a dataset from
    hyperparams - ZeroShotKDHyperparams
    dimensions - shape of the generated images
    num_classes - Total number of classes
    """
    teacher: torch.nn.Module
    hyperparams: DataImpressionHyperparams
    dimensions: Tuple[int, ...]
    num_classes: int
    transfer_criterion: torch.nn.Module
    extract_classifier_weights: ExtractClassifierWeightsFn = extract_classifier_weights2
    device: torch.device = torch.device('cuda')

    def synthesize_batch(self, concentration: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """A Generator function that returns the optimized input value and the sampled dirichlet y values batchwise
        returns: 
        (1) x - Tensor of shape (BATCH_SIZE, *dimensions)
        (2) y - Tensor of shape (BATCH_SIZE, num_classes)
        """

        # sampling target label from Dirichlet distribution
        dirichlet_distribution = (
            torch
                .distributions
                .Dirichlet(concentration)
        )

        y: torch.Tensor = (
            dirichlet_distribution
                .rsample((self.hyperparams.batch_size,)) 
        ).to(self.device)

        # optimization for images
        inputs = (
            torch
                .randn(
                    size=(self.hyperparams.batch_size, *self.dimensions),
                    requires_grad=True,
                    device=self.device
                )
        )
        optimizer = torch.optim.Adam([inputs], self.hyperparams.learning_rate)
        self.teacher.eval()
        for _ in (pbar := tqdm(range(self.hyperparams.iterations))):
            output = F.softmax(self.teacher(inputs) / self.hyperparams.temperature, dim=1)
            loss: torch.Tensor = self.transfer_criterion(output, y.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss:.4f}")
        
        return inputs, y.detach()


    def iter_synthesize(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """A Generator function that returns the optimized input value and the sampled dirichlet y values batchwise
        returns: 
        (1) x - Tensor of shape (BATCH_SIZE, *dimensions)
        (2) y - Tensor of shape (BATCH_SIZE, num_classes)
        """

        # Get Classifier Weights
        classifier_weights = self.extract_classifier_weights(self.teacher)
        class_similarity_matrix = (
            compute_class_similarity_matrix(classifier_weights)
                .clamp(min=1e-6, max=1.0)
        )

        # Generate Synthetic Images
        for label in range(self.num_classes):
            for beta in self.hyperparams.beta:
                concentration = beta * class_similarity_matrix[label]

                # Samples per label, batch and beta
                N = (
                    self.hyperparams.num_samples 
                    / len(self.hyperparams.beta) 
                    / self.hyperparams.batch_size 
                    / self.num_classes
                )
                assert N.is_integer()  # Divisibility Check
                N = int(N)

                for _ in range(N):
                    yield self.synthesize_batch(concentration)

    
    def iter_concentrations(self) -> Iterator[torch.Tensor]:
        # Get Classifier Weights
        classifier_weights = self.extract_classifier_weights(self.teacher)
        class_similarity_matrix = (
            compute_class_similarity_matrix(classifier_weights)
                .clamp(min=1e-6, max=1.0)
        )

        # Generate Synthetic Images
        for label in range(self.num_classes):
            for beta in self.hyperparams.beta:
                concentration = beta * class_similarity_matrix[label]

                # Samples per label, batch and beta
                N = (
                    self.hyperparams.num_samples 
                    / len(self.hyperparams.beta) 
                    / self.hyperparams.batch_size 
                    / self.num_classes
                )
                assert N.is_integer()  # Divisibility Check
                N = int(N)

                for _ in range(N):
                    yield concentration
