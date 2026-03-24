from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def full_causal_attention(q: Tensor, k: Tensor, v: Tensor, *, enable_gqa: bool) -> Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        is_causal=True,
        enable_gqa=enable_gqa,
    )


def local_causal_sliding_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    window: int,
    enable_gqa: bool,
) -> Tensor:
    seqlen = q.size(-2)
    if window >= seqlen:
        return full_causal_attention(q, k, v, enable_gqa=enable_gqa)

    i = torch.arange(seqlen, device=q.device)
    j = torch.arange(seqlen, device=q.device)
    causal = j[None, :] <= i[:, None]
    within_window = j[None, :] >= (i[:, None] - window + 1)
    allowed = causal & within_window
    attn_bias = torch.full((seqlen, seqlen), float("-inf"), device=q.device, dtype=q.dtype)
    attn_bias = attn_bias.masked_fill(allowed, 0.0)

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_bias,
        is_causal=False,
        enable_gqa=enable_gqa,
    )
