"""Attention mechanisms for the FinLangNet Transformer encoder.

Implements the scaled dot-product multi-head attention used inside the
Sequence Representation Generator (SRG) module.  Two classes are provided:

  - FullAttention: scaled dot-product attention with optional causal masking.
  - AttentionLayer: wraps FullAttention with Q/K/V linear projections and
    the output projection, following the standard Transformer design.
  - TriangularCausalMask: upper-triangular boolean mask for autoregressive use
    (not used during FinLangNet inference, included for generality).
"""

import torch
import torch.nn as nn
from math import sqrt
import numpy as np


class TriangularCausalMask:
    """Upper-triangular causal mask for autoregressive attention.

    Creates a boolean mask of shape (B, 1, L, L) where True entries indicate
    positions that should be masked out (i.e., future tokens).

    Args:
        B (int): Batch size.
        L (int): Sequence length.
        device (str): Target device. Default: "cpu".
    """

    def __init__(self, B: int, L: int, device: str = "cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask


class FullAttention(nn.Module):
    """Scaled dot-product attention with optional masking.

    Computes:
        Attention(Q, K, V) = softmax(scale * Q K^T) V

    where scale = 1/√d_k.  An optional boolean mask can be applied to zero
    out (−∞) selected positions before the softmax.

    Args:
        mask_flag (bool): If True, apply causal or padding mask. Default: True.
        factor (int): Unused sparsity factor (kept for API compatibility).
        scale (float, optional): Override the default 1/√d_k scaling.
        attention_dropout (float): Dropout probability on attention weights.
        output_attention (bool): If True, also return the attention weight matrix.
        num_head (int): Number of attention heads (used when applying a padding mask).
    """

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: float = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
        num_head: int = 8,
    ):
        super(FullAttention, self).__init__()
        self.scale            = scale
        self.mask_flag        = mask_flag
        self.output_attention = output_attention
        self.dropout          = nn.Dropout(attention_dropout)
        self.num_head         = num_head

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask,
        tau=None,
        delta=None,
    ):
        """Compute scaled dot-product attention.

        Args:
            queries: Shape (B, L, H, d_k).
            keys:    Shape (B, S, H, d_k).
            values:  Shape (B, S, H, d_v).
            attn_mask: Boolean padding mask of shape (B, L) or a
                       TriangularCausalMask instance; None for no masking.
            tau, delta: Unused – kept for interface compatibility.

        Returns:
            Tuple of:
              - output:  Shape (B, L, H, d_v), attended value vectors.
              - attn:    Shape (B, H, L, S) attention weights if
                         output_attention=True, else None.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        # Compute raw attention scores: (B, H, L, S)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            # attn_mask is a 2D padding mask (B, L): True = pad position to mask out
            batch_size, len_mask = attn_mask.size()
            mask = attn_mask.view(batch_size, 1, len_mask, 1).expand(-1, self.num_head, -1, -1)
            scores.masked_fill_(mask, -np.inf)

        # Softmax + dropout → weighted sum of values
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class AttentionLayer(nn.Module):
    """Multi-head attention layer with Q/K/V and output projections.

    Wraps a core attention module (e.g. FullAttention) with the standard
    linear projections from d_model to (d_keys * n_heads) for queries and keys,
    and (d_values * n_heads) for values, followed by an output projection back
    to d_model.

    Args:
        attention:  Core attention module implementing the attention operation.
        d_model:    Model dimension (input and output size).
        n_heads:    Number of attention heads.
        d_keys:     Per-head key dimension. Defaults to d_model // n_heads.
        d_values:   Per-head value dimension. Defaults to d_model // n_heads.
    """

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: int = None,
        d_values: int = None,
    ):
        super(AttentionLayer, self).__init__()

        d_keys   = d_keys   or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention  = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection   = nn.Linear(d_values * n_heads, d_model)
        self.n_heads          = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask,
        tau=None,
        delta=None,
    ):
        """Project inputs and apply multi-head attention.

        Args:
            queries: Shape (B, L, d_model).
            keys:    Shape (B, S, d_model).
            values:  Shape (B, S, d_model).
            attn_mask: Mask passed through to the inner attention module.
            tau, delta: Passed through unchanged for interface compatibility.

        Returns:
            Tuple of:
              - output: Shape (B, L, d_model).
              - attn:   Attention weights from the inner module.
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H       = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys    = self.key_projection(keys).view(B, S, H, -1)
        values  = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn
