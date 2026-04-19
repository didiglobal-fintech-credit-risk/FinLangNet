"""FinLangNet training script.

Trains the FinLangNet model for multi-scale credit risk prediction across
7 delinquency horizons (dob45dpd7, dob90dpd7, dob90dpd30, dob120dpd7,
dob120dpd30, dob180dpd7, dob180dpd30).

Training details (per Table 6 in the paper):
  - Optimizer:  AdamW, lr=5e-4, betas=(0.9, 0.999), weight_decay=0.01
  - Scheduler:  StepLR with step_size=3, gamma=0.2
  - Epochs:     12
  - Batch size: 512
  - Loss:       DiceBCELoss + FocalTverskyLoss, balanced via DynamicWeightAverage
  - Early stopping patience: 10 epochs (monitored on validation loss)
  - Model selection: best validation KS on the dob90dpd7 head (primary target τ=1)
"""

from Model.sentences_load import train_dataloader, val_dataloader
import torch
import torch.nn.functional as F
from Model.train_loss import FocalTverskyLoss, DiceBCELoss, MultiLoss, DynamicWeightAverage
from Model.FinLangNet import MyModel_FinLangNet
from Model.metric_record import write_log, calculate_metrics
from torch.optim.lr_scheduler import StepLR

# Baseline comparison models (used for ablation / benchmarking)
from other_models.gru_deepfm           import MyModel_GRU
from other_models.gru_attention_deepfm import MyModel_GRU_Attention
from other_models.lstm_deepfm          import MyModel_LSTM
from other_models.transformer_deepfm   import MyModel_Transformer
from other_models.stack_gru_deepfm     import MyModel_StackGRU

# Automatically select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model() -> None:
    """Train FinLangNet with dynamic loss balancing and early stopping.

    The training loop:
      1. Iterates over mini-batches and computes per-head losses.
      2. Dynamically re-weights the two loss components (DiceBCE and
         FocalTversky) using gradient-norm-based DynamicWeightAverage.
      3. Evaluates on the validation set after each epoch and computes
         KS / AUC metrics for all 7 prediction heads.
      4. Saves the model checkpoint with the best dob90dpd7 KS score.
      5. Stops early if validation loss does not improve for 10 epochs.

    Label layout (from the dataloader):
      labels[:, [3, 5, 7, 9]]       → standard reference labels (label_stander)
      labels[:, [0,1,2,4,6,8,10]]   → ground-truth for 7 prediction heads (label_gt)

    The standard labels are used to filter valid evaluation samples for
    longer-horizon heads (e.g., dob120 samples only exist for users who
    have been in the portfolio for at least 120 days).
    """
    early_stop_patience = 10
    best_val_loss       = float('inf')
    best_ks             = 0.0
    best_epoch          = 0

    # Loss functions and dynamic balancer
    loss_weighter = DynamicWeightAverage(2)
    criterion1    = DiceBCELoss()
    criterion2    = FocalTverskyLoss()

    # Instantiate FinLangNet (swap class name to benchmark other models)
    model     = MyModel_FinLangNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01
    )
    scheduler  = StepLR(optimizer, step_size=3, gamma=0.2)
    num_epochs = 12

    for epoch in range(num_epochs):
        loss_weights = [1.0, 1.0]
        multi_loss   = MultiLoss(loss_weights)

        model.train()
        train_loss = 0.0
        train_cnt  = 0

        write_log(f'Epoch: {epoch + 1}')
        print(f"Current lr: {optimizer.state_dict()['param_groups'][0]['lr']}")

        for step, (
            dz_categorica_feature, dz_numeric_feature, person_feature,
            labels, len_dz, cfrnid, bv1_score, date_time_credit,
            inquery_feature, creditos_feature, len_inquery, len_creditos,
        ) in enumerate(train_dataloader):

            # Move tensors to device
            dz_categorica_feature = dz_categorica_feature.to(device)
            dz_numeric_feature    = dz_numeric_feature.to(device)
            person_feature        = person_feature.to(device)
            labels                = labels.float().to(device)
            inquery_feature       = inquery_feature.float().to(device)
            creditos_feature      = creditos_feature.float().to(device)

            # Reference labels for filtering valid long-horizon samples
            label_stander = labels[:, [3, 5, 7, 9]]
            # Ground-truth for the 7 prediction heads
            label_gt      = labels[:, [0, 1, 2, 4, 6, 8, 10]]

            optimizer.zero_grad()
            outputs = model(
                dz_categorica_feature, dz_numeric_feature, person_feature,
                len_dz, inquery_feature, creditos_feature, len_inquery, len_creditos,
            )

            total_loss = 0
            loss_log   = []
            for idx, output in enumerate(outputs):
                label          = label_gt[:, idx].unsqueeze(-1)
                dice_loss      = criterion1(output, label)
                focal_loss     = criterion2(output, label)

                # Compute per-loss gradient norms for dynamic weighting
                dice_grad  = torch.autograd.grad(dice_loss,  output, retain_graph=True, only_inputs=True)[0]
                focal_grad = torch.autograd.grad(focal_loss, output, retain_graph=True, only_inputs=True)[0]
                loss_weighter.update([dice_grad, focal_grad])

                loss = (
                    dice_loss  * loss_weighter.weights[0]
                    + focal_loss * loss_weighter.weights[1]
                )
                loss_log.append(loss.item())
                total_loss += multi_loss(dice_loss, focal_loss)

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

            if step % 100 == 0:
                msg = (
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{step}], '
                    f'dob45dpd7: {loss_log[0]:.4f}, dob90dpd7: {loss_log[1]:.4f}, '
                    f'dob90dpd30: {loss_log[2]:.4f}, dob120dpd7: {loss_log[3]:.4f}, '
                    f'dob120dpd30: {loss_log[4]:.4f}, dob180dpd7: {loss_log[5]:.4f}, '
                    f'dob180dpd30: {loss_log[6]:.4f}, Total Loss: {total_loss.item():.4f}'
                )
                write_log(msg)

            train_cnt += 1

        scheduler.step()
        train_loss /= train_cnt

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_cnt  = 0

        # Prediction and ground-truth collectors for all 7 heads
        val_preds = {name: [] for name in ['dob45dpd7','dob90dpd7','dob90dpd30',
                                            'dob120dpd7','dob120dpd30','dob180dpd7','dob180dpd30']}
        val_true  = {name: [] for name in val_preds}

        with torch.no_grad():
            for val_step, (
                dz_categorica_feature, dz_numeric_feature, person_feature,
                labels, len_dz, cfrnid, bv1_score, date_time_credit,
                inquery_feature, creditos_feature, len_inquery, len_creditos,
            ) in enumerate(val_dataloader):

                dz_categorica_feature = dz_categorica_feature.to(device)
                dz_numeric_feature    = dz_numeric_feature.to(device)
                person_feature        = person_feature.to(device)
                labels                = labels.float().to(device)
                inquery_feature       = inquery_feature.float().to(device)
                creditos_feature      = creditos_feature.float().to(device)

                label_stander = labels[:, [3, 5, 7, 9]]
                label_gt      = labels[:, [0, 1, 2, 4, 6, 8, 10]]

                outputs    = model(
                    dz_categorica_feature, dz_numeric_feature, person_feature,
                    len_dz, inquery_feature, creditos_feature, len_inquery, len_creditos,
                )
                total_loss = 0
                loss_log   = []
                for idx, output in enumerate(outputs):
                    label      = label_gt[:, idx].unsqueeze(-1)
                    step_loss  = multi_loss(criterion1(output, label), criterion2(output, label))
                    loss_log.append(step_loss.item())
                    total_loss += step_loss

                val_loss += total_loss.item()

                if val_step % 1000 == 0:
                    msg = (
                        f'Valid Epoch [{epoch+1}/{num_epochs}], Step [{val_step}], '
                        f'dob45dpd7: {loss_log[0]:.4f}, dob90dpd7: {loss_log[1]:.4f}, '
                        f'dob90dpd30: {loss_log[2]:.4f}, dob120dpd7: {loss_log[3]:.4f}, '
                        f'dob120dpd30: {loss_log[4]:.4f}, dob180dpd7: {loss_log[5]:.4f}, '
                        f'dob180dpd30: {loss_log[6]:.4f}, Total Loss: {total_loss.item():.4f}'
                    )
                    write_log(msg)

                # Collect predictions for the three unconditional heads
                val_preds['dob45dpd7'].extend(outputs[0].tolist())
                val_true['dob45dpd7'].extend(label_gt[:, 0].unsqueeze(-1).tolist())
                val_preds['dob90dpd7'].extend(outputs[1].tolist())
                val_true['dob90dpd7'].extend(label_gt[:, 1].unsqueeze(-1).tolist())
                val_preds['dob90dpd30'].extend(outputs[2].tolist())
                val_true['dob90dpd30'].extend(label_gt[:, 2].unsqueeze(-1).tolist())

                # Long-horizon heads are filtered to users with sufficient history
                for head_idx, (head_name, stander_col) in enumerate([
                    ('dob120dpd7', 0), ('dob120dpd30', 1),
                    ('dob180dpd7', 2), ('dob180dpd30', 3),
                ], start=3):
                    valid_idx = [i for i, v in enumerate(label_stander[:, stander_col].tolist()) if v == 1]
                    val_preds[head_name].extend([outputs[head_idx].tolist()[i] for i in valid_idx])
                    val_true[head_name].extend([label_gt[:, head_idx].unsqueeze(-1).tolist()[i] for i in valid_idx])

                val_cnt += 1

        # Compute and log KS / AUC for all heads
        ks_dob90dpd7 = calculate_metrics(val_true['dob90dpd7'],  val_preds['dob90dpd7'],  epoch, num_epochs, train_loss, val_loss, 'dob90dpd7')
        calculate_metrics(val_true['dob45dpd7'],  val_preds['dob45dpd7'],  epoch, num_epochs, train_loss, val_loss, 'dob45dpd7')
        calculate_metrics(val_true['dob90dpd30'], val_preds['dob90dpd30'], epoch, num_epochs, train_loss, val_loss, 'dob90dpd30')
        calculate_metrics(val_true['dob120dpd7'], val_preds['dob120dpd7'], epoch, num_epochs, train_loss, val_loss, 'dob120dpd7')
        calculate_metrics(val_true['dob180dpd7'], val_preds['dob180dpd7'], epoch, num_epochs, train_loss, val_loss, 'dob180dpd7')
        calculate_metrics(val_true['dob120dpd30'],val_preds['dob120dpd30'],epoch, num_epochs, train_loss, val_loss, 'dob120dpd30')
        calculate_metrics(val_true['dob180dpd30'],val_preds['dob180dpd30'],epoch, num_epochs, train_loss, val_loss, 'dob180dpd30')

        # Save checkpoint for best dob90dpd7 KS (primary deployment target, τ=1)
        if ks_dob90dpd7 > best_ks:
            best_ks    = ks_dob90dpd7
            best_epoch = epoch
            torch.save(model.state_dict(), './model.pth')
            print(f'Saved best model at epoch {epoch}, KS={ks_dob90dpd7:.4f}')

        # Early stopping on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
        elif epoch - best_epoch >= early_stop_patience:
            print('Early stopping triggered.')
            break


if __name__ == '__main__':
    train_model()
