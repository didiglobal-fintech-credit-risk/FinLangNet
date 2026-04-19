"""Loss functions for multi-task credit risk training.

The training objective in FinLangNet combines two loss components that jointly
address class imbalance and hard-sample difficulty, following the paper's
Dynamically Weighted Hybrid Loss (Section 3.3):

  L_total = (1/n) Σ_i ω_i [ β(y'_i - y_i)^2 + (1-β) L_WLL,i ]

where:
  - L_WLL (Weighted Logarithmic Loss) assigns higher penalties to the minority
    class (defaulters) to counteract severe class imbalance in credit data.
  - ω_i (Dynamic Hard Example Mining weight) up-weights samples where the model
    struggles most, measured by the L2 norm of their gradient contributions.

In this implementation:
  - DiceBCELoss approximates the WLL-style focal loss with a combined BCE +
    Dice objective that is robust to class imbalance.
  - FocalTverskyLoss provides the hard-sample emphasis through the Tversky
    index with an additional focal exponent γ.
  - DynamicWeightAverage balances the two loss components dynamically based
    on their per-sample gradient norms, implementing the ω_i schedule.
  - MultiLoss handles the weighted combination of multiple loss terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for hard-sample emphasis under class imbalance.

    The Tversky index generalizes the Dice coefficient by weighting false
    positives (α) and false negatives (β) separately. The focal exponent γ
    further down-weights easy examples, analogous to focal loss.

    Formula:
        Tversky = (TP + smooth) / (TP + α*FP + β*FN + smooth)
        L_FT    = (1 - Tversky)^γ
    """

    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """Compute Focal Tversky Loss.

        Args:
            inputs: Predicted probabilities, shape (N, 1) or (N,).
            targets: Binary ground-truth labels, same shape as inputs.
            smooth: Laplace smoothing constant to avoid division by zero.
            alpha: Weight applied to false positives.
            beta:  Weight applied to false negatives.
            gamma: Focal exponent; set > 1 to further penalize hard examples.

        Returns:
            Scalar loss tensor.
        """
        inputs  = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky       = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        focal_tversky = (1 - tversky) ** gamma
        return focal_tversky


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy Loss.

    Adds a Dice coefficient loss term to BCE to improve gradient signal for
    severely imbalanced binary classification tasks. The Dice term directly
    optimizes the overlap between predictions and targets, while BCE provides
    per-sample log-likelihood supervision.

    Formula:
        Dice_Loss = 1 - (2*|P∩T| + smooth) / (|P| + |T| + smooth)
        L         = BCE + Dice_Loss
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1.0,
    ) -> torch.Tensor:
        """Compute Dice + BCE Loss.

        Args:
            inputs:  Predicted probabilities, shape (N, 1) or (N,).
            targets: Binary ground-truth labels, same shape as inputs.
            smooth:  Smoothing constant for numerical stability.

        Returns:
            Scalar loss tensor.
        """
        inputs  = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss    = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        bce_loss     = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return bce_loss + dice_loss


class MultiLoss(nn.Module):
    """Weighted linear combination of multiple loss terms.

    Provides a simple fixed-weight wrapper used to combine DiceBCELoss and
    FocalTverskyLoss into a single scalar before dynamic re-weighting.

    Args:
        loss_weights (list of float, optional): Per-loss scalar weights.
            Defaults to equal weighting if not provided.
    """

    def __init__(self, loss_weights=None):
        super(MultiLoss, self).__init__()
        self.loss_weights = loss_weights

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of losses.

        Args:
            *losses: Variable number of scalar loss tensors.

        Returns:
            Weighted sum as a scalar tensor.
        """
        if self.loss_weights is None:
            self.loss_weights = [1.0] * len(losses)
        total_loss = sum(w * loss for w, loss in zip(self.loss_weights, losses))
        return total_loss


class DynamicWeightAverage:
    """Dynamic loss balancing via gradient-norm-based weighting.

    Implements the Dynamic Hard Example Mining schedule from the paper
    (Section 3.3). For each training step, the relative weight of each loss
    component is proportional to the α-th power of its per-sample gradient
    norm, causing the optimizer to focus more on components where the model
    currently struggles.

    Weight update rule:
        g_k = ||∂L_k / ∂ŷ||_2^α
        w_k = g_k / Σ_j g_j

    Args:
        n_losses (int): Number of loss components to balance.
        alpha (float): Exponent controlling sensitivity to gradient magnitude.
                       Higher values amplify the dominance of larger gradients.
                       Default: 2.0.
    """

    def __init__(self, n_losses: int, alpha: float = 2.0):
        self.weights = [1.0 / n_losses] * n_losses
        self.alpha   = alpha

    def update(self, loss_gradients: list) -> None:
        """Update weights based on current per-loss gradient norms.

        Args:
            loss_gradients: List of gradient tensors, one per loss component.
                            Each tensor should correspond to ∂L_k/∂ŷ for the
                            current batch.
        """
        normed_grads = [
            torch.norm(g.detach(), 2).pow(self.alpha).item()
            for g in loss_gradients
        ]
        normed_sum   = sum(normed_grads)
        self.weights = [ng / normed_sum for ng in normed_grads]
