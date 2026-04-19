"""Training metric logging utilities for FinLangNet.

Provides functions for:
  - Writing timestamped log entries to a daily log file.
  - Computing and logging the full suite of binary classification metrics used
    in FinLangNet evaluation (KS, AUC-ROC, Gini, Accuracy, Recall, F1, AUC-PR).

The primary evaluation metric is the Kolmogorov-Smirnov (KS) statistic, which
measures the maximum divergence between the cumulative distributions of positive
and negative classes (Massey Jr., 1951).  KS is the industry-standard metric for
credit scoring model evaluation.
"""

import datetime
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    recall_score,
    f1_score,
    average_precision_score,
)


def write_log(w: str) -> None:
    """Append a timestamped message to today's log file.

    Log files are written to the `logs/` directory and named by date (MMDD).

    Args:
        w: The message string to log.
    """
    file_name = 'logs/' + datetime.date.today().strftime('%m%d') + "_{}.log"
    t0   = datetime.datetime.now().strftime('%H:%M:%S')
    info = f"{t0} : {w}"
    print(info)
    with open(file_name, 'a') as f:
        f.write(info + '\n')


def calculate_metrics(
    true_values: list,
    predictions: list,
    epoch: int,
    num_epochs: int,
    train_loss: float,
    val_loss: float,
    label: str,
) -> float:
    """Compute and log binary classification metrics for one prediction head.

    Computes the following metrics and writes them to the log:
      - AUC-ROC: area under the ROC curve.
      - KS: max(TPR − FPR), the primary credit-scoring metric.
      - Gini: 2 * AUC − 1, a normalized discrimination coefficient.
      - Accuracy, Recall, F1: at a 0.5 classification threshold.
      - AUC-PR: area under the precision-recall curve.

    Args:
        true_values: Ground-truth binary labels.
        predictions: Predicted probabilities in [0, 1].
        epoch:       Current epoch index (0-based).
        num_epochs:  Total number of training epochs.
        train_loss:  Average training loss for the current epoch.
        val_loss:    Cumulative validation loss for the current epoch.
        label:       Name of the prediction head (e.g. "dob90dpd7").

    Returns:
        KS statistic (float) for the current epoch and head.
    """
    fpr, tpr, _ = roc_curve(true_values, predictions)
    auc_value   = auc(fpr, tpr)
    ks          = max(tpr - fpr)
    gini        = 2 * auc_value - 1

    # Threshold-based metrics at the default 0.5 operating point
    val_preds_class = [1 if x > 0.5 else 0 for x in predictions]
    accuracy        = accuracy_score(true_values, val_preds_class)
    recall          = recall_score(true_values, val_preds_class)
    f1              = f1_score(true_values, val_preds_class)
    auc_pr          = average_precision_score(true_values, predictions)

    msg = (
        f'{label}: Epoch [{epoch + 1}/{num_epochs}] '
        f'- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
        f'Val AUC_roc: {auc_value:.4f}, Val KS: {ks:.4f}, '
        f'Val Gini: {gini:.4f}, Val Accuracy: {accuracy:.4f}, '
        f'Val Recall: {recall:.4f}, Val F1: {f1:.4f}, '
        f'Val AUC_PR: {auc_pr:.4f}'
    )
    write_log(msg)
    return ks
