"""FinLangNet: Core Model Definition.

This module implements the full FinLangNet architecture described in:
  "A Unified Framework for Modeling Heterogeneous Financial Data
   via Dual-Granularity Prompting"

FinLangNet processes heterogeneous financial data through a dual-module design:
  1. Non-Sequential Module (DeepFM): captures static feature interactions
     from basic personal information via Factorization Machine + DNN.
  2. Sequential Module (SRG – Sequence Representation Generator): encodes
     temporal behavioral sequences (loan behavior, credit inquiries, account
     records) using a Transformer encoder augmented with a dual-prompt mechanism:
       - Feature-level Prompt (phi_c): a learnable CLS token prepended to each
         channel's token sequence, capturing channel-specific global patterns.
       - User-level Prompt (P_s): aggregates cross-channel representations into
         a holistic user-level behavioral profile.

Multi-scale credit risk prediction:
  The fused static + sequential representation O = [O_m; O_bs] is projected
  through shared branch layers into 7 prediction heads, each targeting a
  distinct (days-on-book, days-past-due) delinquency horizon:
    dob45dpd7, dob90dpd7, dob90dpd30, dob120dpd7,
    dob120dpd30, dob180dpd7, dob180dpd30.
  A DependencyLayer enforces monotonic ordering between horizons that share
  the same DPD threshold (e.g., dob90dpd7 feeds into dob120dpd7).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Modules.encode import Encoder, EncoderLayer
from Modules.attention import AttentionLayer, FullAttention
from Modules.embedding import PositionalEmbedding


class DependencyLayer(nn.Module):
    """Linear projection layer with optional additive dependency from a prior head.

    Used in the multi-scale prediction heads to enforce the constraint that
    longer-horizon risk scores are at least as high as shorter-horizon ones
    for the same DPD threshold (monotonic ordering across DOB windows).

    Attributes:
        linear (nn.Linear): Linear projection from hidden dim to output dim.
        dependent (bool): If True, adds a dependency residual from a prior head.
    """

    def __init__(self, input_size: int, output_size: int):
        super(DependencyLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dependent = False  # Set to True for non-dpd7 heads during model init

    def forward(self, x: torch.Tensor, dependency: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size).
            dependency: Optional additive residual from the preceding head,
                        shape (batch_size, output_size). Added only when
                        self.dependent is True.

        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        x = self.linear(x)
        if self.dependent and dependency is not None:
            x = x + dependency
        return x


class MyModel_FinLangNet(nn.Module):
    """FinLangNet: Dual-module credit risk model with dual-granularity prompting.

    Architecture overview (see Figure 1 in the paper):

    Non-Sequential Module (DeepFM):
      - Input: static personal features m ∈ R^M (basic info, demographics)
      - FM component: 1st-order sparse embeddings + 2nd-order interaction via
        the inner-product shortcut: 0.5*(||Σ v_i||^2 - Σ||v_i||^2)
      - DNN component: deep MLP over flattened 2nd-order embeddings
      - Output: O_m ∈ R^256

    Sequential Module – SRG (Sequence Representation Generator):
      Processes three temporal data sources independently:
        (a) Loan Behavior (dz): categorical + numeric features discretized
            into tokens and encoded by a Transformer encoder.  The CLS token
            prepended to the sequence serves as the Feature-level Prompt (phi_c)
            that aggregates channel-specific global patterns.
        (b) Credit Inquiry Records (inquery): encoded by a separate Transformer.
        (c) Account / Credit-line Records (creditos): encoded by a third Transformer.
      Outputs are projected to R^256 per source and concatenated.
      Output: O_bs ∈ R^768 (3 × 256) — the user-level sequential representation.

    Fusion & Multi-scale Prediction:
      x_concat = [O_m ∈ R^256 ; O_bs ∈ R^768] → total input dim = 1024
      7 shared-branch heads predict P(overdue | DOB window, DPD threshold):
        dob45dpd7, dob90dpd7, dob90dpd30, dob120dpd7,
        dob120dpd30, dob180dpd7, dob180dpd30

    Args:
        embedding_dim (int): Token embedding dimension. Default: 64.
        hidden_dim (int): Transformer hidden dimension. Default: 64.
        num_layers (int): Number of base Transformer encoder layers per source
                          (actual depth = num_layers * 2). Default: 2.
        num_head (int): Number of attention heads. Default: 8.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_head: int = 8,
    ):
        super(MyModel_FinLangNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_head = num_head

        # ── Credit Inquiry Records (inquery) channel ──────────────────────────
        self.inquery_category_index = list(range(3 + 2))  # categorical feature indices
        self.inquery_fe_len = 3 + 2                        # total feature channels

        # ── Account / Credit-line Records (creditos) channel ──────────────────
        self.creditos_category_index = list(range(8 + 21))
        self.creditos_fe_len = 8 + 21

        self.encode_dim = 128  # vocabulary size for token embeddings

        # ── Token embedding tables for the inquery channel ────────────────────
        self.inquery_embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=self.encode_dim, embedding_dim=self.embedding_dim)
            for _ in range(self.inquery_fe_len)
        ])

        # ── Token embedding tables for the creditos channel ───────────────────
        self.creditos_embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=self.encode_dim, embedding_dim=self.embedding_dim)
            for _ in range(self.creditos_fe_len)
        ])

        # Feature-level Prompt (phi_c) for the inquery channel.
        # This learnable CLS token is prepended to the token sequence and,
        # after encoding, its output vector aggregates channel-global patterns.
        self.inquery_cls_token_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        # Actual encoder depth is doubled for richer temporal modeling
        self.num_layers_circulo = self.num_layers * 2

        # ── Transformer encoder for inquery sequences ─────────────────────────
        self.inquery_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, output_attention=True, num_head=self.num_head),
                        self.hidden_dim,
                        self.num_head,
                    ),
                    self.hidden_dim,
                    self.num_head,
                )
                for _ in range(self.num_layers_circulo)
            ],
            norm_layer=torch.nn.LayerNorm(self.hidden_dim),
        )

        # ── Transformer encoder for creditos sequences ────────────────────────
        self.creditos_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, output_attention=True, num_head=self.num_head),
                        self.hidden_dim,
                        self.num_head,
                    ),
                    self.hidden_dim,
                    self.num_head,
                )
                for _ in range(self.num_layers_circulo)
            ],
            norm_layer=torch.nn.LayerNorm(self.hidden_dim),
        )

        self.inquery_dropout = nn.Dropout(0.1)
        self.creditos_dropout = nn.Dropout(0.1)

        # Project CLS-token hidden states to the shared 256-d representation
        self.inquery_lstm_linear = nn.Linear(self.hidden_dim, 256)
        self.creditos_lstm_linear = nn.Linear(self.hidden_dim, 256)

        # ── Loan Behavior (dz) channel ─────────────────────────────────────────
        # Feature-level Prompt (phi_c) for the creditos channel
        self.creditos_cls_token_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        # Combined categorical + numeric discretized features for loan behavior
        self.category_fe_len = 4 + 237

        self.category_embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=self.encode_dim, embedding_dim=self.embedding_dim)
            for _ in range(self.category_fe_len)
        ])

        # Feature-level Prompt (phi_c) for the loan behavior (dz) channel
        self.time_cls_token_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

        # ── Transformer encoder for loan behavior sequences ───────────────────
        self.dz_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, output_attention=True, num_head=self.num_head),
                        self.hidden_dim,
                        self.num_head,
                    ),
                    self.hidden_dim,
                    self.num_head,
                )
                for _ in range(self.num_layers_circulo)
            ],
            norm_layer=torch.nn.LayerNorm(self.hidden_dim),
        )

        self.dz_dropout = nn.Dropout(0.1)
        self.dz_lstm_linear = nn.Linear(self.hidden_dim, 256)

        # GELU activation for non-linear transformations after encoding
        self.act = F.gelu

        # ── Non-Sequential Module (DeepFM) – static personal features ─────────
        self.person_fe_len = 3 + 8  # number of static personal feature fields

        # Embedding tables for static features (shared across FM and DNN)
        self.personal_embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=32, embedding_dim=self.embedding_dim)
            for _ in range(self.person_fe_len)
        ])
        self.personal_feature_linear = nn.Linear(self.person_fe_len * self.embedding_dim, 128)

        # FM 1st-order term: linear part y_FM^(1)
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(32, 1) for _ in range(self.person_fe_len)
        ])

        # FM 2nd-order term: interaction part y_FM^(2) via inner products of latent vectors V_j
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(32, self.embedding_dim) for _ in range(self.person_fe_len)
        ])

        # DNN component: models high-order non-linear correlations
        hid_dims = [256, 128]
        self.all_dims = [self.person_fe_len * self.embedding_dim] + hid_dims
        for i in range(1, len(self.all_dims)):
            setattr(self, f'linear_{i}',     nn.Linear(self.all_dims[i - 1], self.all_dims[i]))
            setattr(self, f'batchNorm_{i}',  nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, f'activation_{i}', nn.ReLU())
            setattr(self, f'dropout_{i}',    nn.Dropout(0.2))

        # Fuses FM 1st-order, FM 2nd-order interaction, and DNN outputs → O_m ∈ R^256
        self.person_final_linear = nn.Linear(self.person_fe_len + self.embedding_dim + 128, 256)

        # ── Multi-scale credit risk prediction heads ───────────────────────────
        # Each head predicts P(overdue | τ) for a specific (DOB, DPD) combination.
        # Naming convention: dob{X}dpd{Y} = days-on-book X, days-past-due threshold Y.
        self.multihead_name = [
            'dob45dpd7', 'dob90dpd7', 'dob90dpd30',
            'dob120dpd7', 'dob120dpd30', 'dob180dpd7', 'dob180dpd30',
        ]

        # Fused input: O_m(256) + dz(256) + inquery(256) + creditos(256) = 1024
        in_size = 256 + 256 + 256 + 256
        hidden_sizes = [512, 256, 128, 32]
        self.dropout_prob = 0.2
        self.multihead_dict = nn.ModuleDict()

        # Build shared branch layers (weights are reused across all heads)
        branch_layers = {}
        for hidden_size in hidden_sizes:
            layers = [
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(p=self.dropout_prob),
            ]
            branch_layers[hidden_size] = nn.Sequential(*layers)
            in_size = hidden_size

        # Assemble per-head Sequential with DependencyLayer for cross-horizon ordering.
        # For dpd > 7 heads, `dependent=True` enables additive residual from the
        # corresponding dpd=7 head, enforcing P(dpd30) ≥ P(dpd7) at the same DOB.
        for name in self.multihead_name:
            layers = []
            _, dpd_str = name.split('dpd')
            for hs in hidden_sizes:
                layers.append(branch_layers[hs])
            dep_layer = DependencyLayer(hidden_sizes[-1], 1)
            dep_layer.dependent = (dpd_str != '7')  # dpd30 heads depend on dpd7
            layers.append(dep_layer)
            self.multihead_dict[name] = nn.Sequential(*layers)

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        dz_categorica_feature: torch.Tensor,
        dz_numeric_feature: torch.Tensor,
        person_feature: torch.Tensor,
        len_dz: torch.Tensor,
        x_inquery: torch.Tensor,
        x_creditos: torch.Tensor,
        inquery_length: torch.Tensor,
        creditos_length: torch.Tensor,
    ):
        """Forward pass through the full FinLangNet model.

        Args:
            dz_categorica_feature: Loan behavior categorical tokens,
                shape (BS, num_cat_channels, seq_len).
            dz_numeric_feature: Loan behavior discretized numeric tokens,
                shape (BS, num_num_channels, seq_len).
            person_feature: Static personal feature indices,
                shape (BS, person_fe_len).
            len_dz: Actual sequence lengths for loan behavior,
                shape (BS,).
            x_inquery: Credit inquiry token sequences,
                shape (BS, inquery_fe_len, seq_len).
            x_creditos: Account record token sequences,
                shape (BS, creditos_fe_len, seq_len).
            inquery_length: Actual lengths for inquery sequences, shape (BS,).
            creditos_length: Actual lengths for creditos sequences, shape (BS,).

        Returns:
            list[torch.Tensor]: 7 prediction tensors, one per head (in the order
            defined by self.multihead_name), each of shape (BS, 1) with values
            in (0, 1) representing delinquency probabilities.
        """
        # ── SRG: Credit Inquiry Sequences ────────────────────────────────────
        # Embed each feature channel independently, then sum across channels
        # (feature-level token fusion before the Transformer encoder).
        inquery_embedded_features = []
        for i in range(self.inquery_fe_len):
            embedded = self.inquery_embedding_layers[i](
                torch.as_tensor(x_inquery[:, i, :], dtype=torch.long)
            )
            inquery_embedded_features.append(embedded)

        # Prepend Feature-level Prompt (phi_c) as CLS token
        inquery_cls_tokens = self.inquery_cls_token_embedding.expand(x_inquery.size(0), -1, -1)
        inquery_seq = torch.sum(torch.stack(inquery_embedded_features), dim=0)  # (BS, L, D)
        inquery_seq = torch.cat((inquery_cls_tokens, inquery_seq), dim=1)       # (BS, L+1, D)

        inquery_enc_out, _ = self.inquery_encoder(inquery_seq, attn_mask=None)
        inquery_enc_out = self.act(inquery_enc_out)
        inquery_enc_out = self.inquery_dropout(inquery_enc_out)
        # Extract CLS-token output as the channel representation
        inquery_out = self.inquery_lstm_linear(inquery_enc_out[:, 0, :])  # (BS, 256)

        # ── SRG: Account / Credit-line Sequences ─────────────────────────────
        creditos_embedded_features = []
        for i in range(self.creditos_fe_len):
            embedded = self.creditos_embedding_layers[i](
                torch.as_tensor(x_creditos[:, i, :], dtype=torch.long)
            )
            creditos_embedded_features.append(embedded)

        creditos_cls_tokens = self.creditos_cls_token_embedding.expand(x_creditos.size(0), -1, -1)
        creditos_seq = torch.sum(torch.stack(creditos_embedded_features), dim=0)
        creditos_seq = torch.cat((creditos_cls_tokens, creditos_seq), dim=1)

        creditos_enc_out, _ = self.creditos_encoder(creditos_seq, attn_mask=None)
        creditos_enc_out = self.act(creditos_enc_out)
        creditos_enc_out = self.creditos_dropout(creditos_enc_out)
        creditos_out = self.creditos_lstm_linear(creditos_enc_out[:, 0, :])  # (BS, 256)

        # ── SRG: Loan Behavior Sequences ─────────────────────────────────────
        # Concatenate categorical and discretized-numeric feature channels
        dz_feature_input = torch.cat((dz_categorica_feature, dz_numeric_feature), dim=1)
        dz_embedded_features = []
        for i in range(self.category_fe_len):
            embedded = self.category_embedding_layers[i](
                torch.as_tensor(dz_feature_input[:, i, :], dtype=torch.long)
            )
            dz_embedded_features.append(embedded)

        # Prepend Feature-level Prompt (phi_c) as CLS token for loan behavior
        time_cls_tokens = self.time_cls_token_embedding.expand(dz_feature_input.size(0), -1, -1)
        dz_seq = torch.sum(torch.stack(dz_embedded_features), dim=0)
        dz_seq = torch.cat((time_cls_tokens, dz_seq), dim=1)

        dz_enc_out, _ = self.dz_encoder(dz_seq, attn_mask=None)
        dz_enc_out = self.act(dz_enc_out)
        dz_enc_out = self.dz_dropout(dz_enc_out)
        dz_out = self.dz_lstm_linear(dz_enc_out[:, 0, :])  # (BS, 256)

        # ── Non-Sequential Module (DeepFM): Static Personal Features ─────────
        person_feature_cat = person_feature

        # FM 1st-order: y_FM^(1) = Σ_j w_j * x_j  (linear term)
        fm_1st_sparse_res = [
            emb(torch.as_tensor(person_feature_cat[:, i], dtype=torch.long).unsqueeze(1)).view(-1, 1)
            for i, emb in enumerate(self.fm_1st_order_sparse_emb)
        ]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)   # (BS, person_fe_len)

        # FM 2nd-order: y_FM^(2) = 0.5*(||Σ v_j||^2 - Σ||v_j||^2)  (pairwise interactions)
        fm_2nd_order_res = [
            emb(torch.as_tensor(person_feature_cat[:, i], dtype=torch.long).unsqueeze(1))
            for i, emb in enumerate(self.fm_2nd_order_sparse_emb)
        ]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)     # (BS, person_fe_len, emb_dim)

        sum_embed        = torch.sum(fm_2nd_concat_1d, 1)
        square_sum_embed = sum_embed * sum_embed
        sum_square_embed = torch.sum(fm_2nd_concat_1d * fm_2nd_concat_1d, 1)
        sub = (square_sum_embed - sum_square_embed) * 0.5          # (BS, emb_dim)

        # DNN component: high-order non-linear feature interactions
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)              # (BS, person_fe_len * emb_dim)
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, f'linear_{i}')(dnn_out)
            dnn_out = getattr(self, f'batchNorm_{i}')(dnn_out)
            dnn_out = getattr(self, f'activation_{i}')(dnn_out)
            dnn_out = getattr(self, f'dropout_{i}')(dnn_out)

        # Fuse FM and DNN outputs → O_m ∈ R^256
        person_cat_out = torch.cat((fm_1st_sparse_res, sub, dnn_out), dim=1)
        person_out = self.person_final_linear(person_cat_out)      # (BS, 256)

        # ── Fusion: concatenate all module outputs ────────────────────────────
        # x_concat = [O_dz ; O_m ; O_inquery ; O_creditos] ∈ R^1024
        x_concat = torch.cat((dz_out, person_out, inquery_out, creditos_out), dim=1)

        # ── Multi-scale prediction: 7 heads over shared branch layers ─────────
        final_output = []
        output_history = {}  # stores sigmoid outputs for dependency propagation

        for name in self.multihead_name:
            dob = int(name.replace('dob', '').split('dpd')[0])
            dpd = int(name.split('dpd')[1])

            head_output = self.multihead_dict[name](x_concat)

            # Add residual from shorter-horizon head (same DPD, smaller DOB)
            # This implements the monotonic ordering constraint across time horizons.
            for prev_name in self.multihead_name:
                prev_dob = int(prev_name.replace('dob', '').split('dpd')[0])
                prev_dpd = int(prev_name.split('dpd')[1])
                if dob > prev_dob and dpd == prev_dpd:
                    head_output = head_output + output_history[prev_name]

            head_output = self.sigmoid(head_output)
            final_output.append(head_output)
            output_history[name] = head_output

        return final_output


def get_parameter_number(model: nn.Module) -> dict:
    """Returns total and trainable parameter counts for a model.

    Args:
        model: A PyTorch nn.Module instance.

    Returns:
        dict with keys 'Total' and 'Trainable'.
    """
    total_num     = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
