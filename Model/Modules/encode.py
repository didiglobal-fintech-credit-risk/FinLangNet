"""Transformer encoder stack for the FinLangNet SRG module.

Provides the standard Transformer encoder architecture used by all three
temporal sequence channels (loan behavior, credit inquiries, account records)
in the Sequence Representation Generator (SRG):

  Encoder
  └── N × EncoderLayer
        ├── Multi-Head Self-Attention  (with residual + LayerNorm)
        └── Position-wise Feed-Forward (with residual + LayerNorm)

The CLS token prepended to each sequence by the SRG serves as the
Feature-level Prompt (phi_c); its final hidden state is extracted after
encoding to represent the full channel-level behavioral pattern.
"""

import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Stack of Transformer encoder layers with optional convolutional downsampling.

    Args:
        attn_layers (list[EncoderLayer]): Ordered list of encoder layer modules.
        conv_layers (list, optional): Convolutional downsampling layers inserted
            between attention layers (used in some time-series variants; not
            used in the default FinLangNet configuration).
        norm_layer (nn.Module, optional): Final normalization layer applied
            after the last encoder layer (typically nn.LayerNorm).
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm        = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """Run the full encoder stack.

        Args:
            x (torch.Tensor): Input sequence of shape (B, L+1, D), where the
                first position (index 0) holds the CLS/prompt token.
            attn_mask: Optional attention mask passed to each layer.
            tau, delta: Unused; kept for interface compatibility with
                        time-series variants.

        Returns:
            Tuple of:
              - x (torch.Tensor): Encoded sequence, shape (B, L+1, D).
                  x[:, 0, :] is the CLS-token representation used downstream.
              - attns (list): Attention weight tensors from each layer.
        """
        attns = []

        if self.conv_layers is not None:
            # Interleaved attention + convolution (not used in default config)
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta_i = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta_i)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer: self-attention + feed-forward.

    Implements the standard Pre-LN / Post-LN Transformer encoder layer:

        x' = LayerNorm(x + Dropout(Attention(x)))
        x'' = LayerNorm(x' + Dropout(FFN(x')))

    where FFN is a two-layer 1-D convolution expanding to d_ff then back to d_model.

    Args:
        attention (nn.Module): Multi-head attention module (AttentionLayer).
        d_model (int): Model / hidden dimension.
        d_ff (int, optional): Feed-forward inner dimension. Defaults to 4 * d_model.
        dropout (float): Dropout probability. Default: 0.1.
        activation (str): Activation function for the FFN; "relu" or "gelu".
    """

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.attention  = attention
        # Point-wise feed-forward implemented as 1-D convolutions (equivalent to
        # two linear layers applied independently to each position).
        self.conv1      = nn.Conv1d(in_channels=d_model, out_channels=d_ff,  kernel_size=1)
        self.conv2      = nn.Conv1d(in_channels=d_ff,    out_channels=d_model, kernel_size=1)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """Apply self-attention followed by position-wise feed-forward.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d_model).
            attn_mask: Optional mask forwarded to the attention module.
            tau, delta: Passed through for interface compatibility.

        Returns:
            Tuple of:
              - output (torch.Tensor): Shape (B, L, d_model).
              - attn: Attention weights from the inner attention module.
        """
        # ── Self-Attention sub-layer ──────────────────────────────────────────
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)        # residual connection
        y = x = self.norm1(x)              # post-attention LayerNorm

        # ── Feed-Forward sub-layer ────────────────────────────────────────────
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn     # residual + final LayerNorm
