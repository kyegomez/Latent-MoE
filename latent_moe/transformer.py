"""A small decoder-only Transformer with Grouped-Query Attention + LatentMoE.

This wires the :class:`~latent_moe.main.LatentMoE` feed-forward layer into a
modern, minimal LLM stack:

    * RMSNorm (pre-norm)
    * Grouped-Query Attention (GQA) with rotary position embeddings (RoPE) and a
      causal mask, via ``F.scaled_dot_product_attention``.
    * LatentMoE as the token-mixing FFN in every block.

Each block's MoE refreshes its ``aux_loss`` / ``z_loss`` on the forward pass; the
model sums them across layers into :attr:`MoETransformer.aux_loss` so a trainer
can simply add it to the cross-entropy loss.

Like the rest of the repo this favors clarity over raw performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from latent_moe.main import (
    LatentMoE,
    LatentMoEConfig,
    Variant,
    detect_devices,
)


# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #
@dataclass
class TransformerConfig:
    """Configuration for :class:`MoETransformer`.

    Attributes:
        vocab_size: Size of the token vocabulary.
        d_model: Residual-stream / hidden dimension ``d``.
        n_layers: Number of Transformer blocks.
        n_heads: Number of query heads. Must divide ``d_model``.
        n_kv_heads: Number of key/value heads for GQA. Must divide ``n_heads``
            (``n_kv_heads == n_heads`` recovers plain multi-head attention;
            ``n_kv_heads == 1`` is multi-query attention).
        max_seq_len: Maximum sequence length (sizes the RoPE cache).
        d_ff: Expert intermediate width ``m`` in each LatentMoE FFN.
        rope_base: RoPE base frequency (``theta``).
        norm_eps: RMSNorm epsilon.
        n_experts, top_k, alpha, n_shared, variant, moe_impl,
        aux_loss_coef, z_loss_coef: Forwarded to each block's
            :class:`~latent_moe.main.LatentMoEConfig`.
        device: Force placement (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``); ``None``
            auto-detects. Shared with the MoE layers so the model stays coherent.
        multi_gpu: Shard each MoE's routed experts across all visible CUDA GPUs.
    """

    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    max_seq_len: int = 2048

    # -- FFN / MoE. ----------------------------------------------------------- #
    d_ff: int = 1408
    n_experts: int = 16
    top_k: int = 4
    alpha: int = 2
    n_shared: int = 1
    variant: Variant = "acc"
    moe_impl: str = "grouped"
    aux_loss_coef: float = 1e-2
    z_loss_coef: float = 1e-3

    # -- Attention / misc. ---------------------------------------------------- #
    rope_base: float = 10000.0
    norm_eps: float = 1e-6

    # -- Placement. ----------------------------------------------------------- #
    device: Optional[str] = None
    multi_gpu: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by "
                f"n_heads={self.n_heads}."
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} must be divisible by "
                f"n_kv_heads={self.n_kv_heads}."
            )
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError(
                "head_dim (d_model / n_heads) must be even for RoPE."
            )

    def moe_config(self) -> LatentMoEConfig:
        """Build the :class:`LatentMoEConfig` for one block's FFN."""
        return LatentMoEConfig(
            d=self.d_model,
            m=self.d_ff,
            n_experts=self.n_experts,
            top_k=self.top_k,
            alpha=self.alpha,
            n_shared=self.n_shared,
            variant=self.variant,
            impl=self.moe_impl,  # type: ignore[arg-type]
            aux_loss_coef=self.aux_loss_coef,
            z_loss_coef=self.z_loss_coef,
            device=self.device,
            multi_gpu=self.multi_gpu,
        )


# ----------------------------------------------------------------------------- #
# Normalization
# ----------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean-subtraction, single scale)."""

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (x * norm) * self.weight


# ----------------------------------------------------------------------------- #
# Rotary position embeddings
# ----------------------------------------------------------------------------- #
def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Precompute the ``(cos, sin)`` tables for RoPE.

    Args:
        seq_len: Number of positions to cache.
        head_dim: Per-head dimension (must be even).
        base: RoPE base frequency.
        device: Device to build the cache on.

    Returns:
        ``(cos, sin)``, each of shape ``(1, 1, seq_len, head_dim)`` so they
        broadcast over ``(batch, heads, seq, head_dim)``.
    """
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, device=device).float()
            / head_dim
        )
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq, head_dim)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return cos, sin


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the two halves of the last dim: ``[x1, x2] -> [-x2, x1]``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to ``x`` of shape ``(b, heads, seq, head_dim)``."""
    return (x * cos) + (_rotate_half(x) * sin)


# ----------------------------------------------------------------------------- #
# Grouped-Query Attention
# ----------------------------------------------------------------------------- #
def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Repeat each KV head ``n_rep`` times so it pairs with ``n_rep`` query heads.

    Args:
        x: Key/value tensor of shape ``(b, n_kv_heads, seq, head_dim)``.
        n_rep: Number of query heads per KV head (``n_heads // n_kv_heads``).

    Returns:
        Tensor of shape ``(b, n_kv_heads * n_rep, seq, head_dim)``.
    """
    if n_rep == 1:
        return x
    b, n_kv, s, hd = x.shape
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, s, hd)
    return x.reshape(b, n_kv * n_rep, s, hd)


class LayerKVCache:
    """Per-layer key/value cache for autoregressive decoding.

    Stores the post-RoPE key/value tensors of every position seen so far, so each
    decode step only computes K/V for the new token(s) and attends over the
    concatenation. Turns generation from ``O(n^2)`` into ``O(n)``.

    Keys/values are kept *before* the GQA head-repeat (shape
    ``(b, n_kv_heads, seq, head_dim)``) so the cache holds the compact KV heads.
    """

    def __init__(self) -> None:
        self.k: Optional[Tensor] = None
        self.v: Optional[Tensor] = None

    @property
    def seq_len(self) -> int:
        """Number of positions currently cached."""
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Append new K/V along the sequence axis and return the full cache.

        Args:
            k: New keys ``(b, n_kv_heads, s_new, head_dim)``.
            v: New values ``(b, n_kv_heads, s_new, head_dim)``.

        Returns:
            The full cached ``(k, v)`` including the newly appended positions.
        """
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat((self.k, k), dim=2)
            self.v = torch.cat((self.v, v), dim=2)
        return self.k, self.v

    def reset(self) -> None:
        """Empty the cache."""
        self.k = None
        self.v = None


class GroupedQueryAttention(nn.Module):
    """Causal Grouped-Query Attention with RoPE.

    Query heads outnumber key/value heads by a factor of
    ``n_heads // n_kv_heads``, shrinking the KV cache and KV projections while
    keeping full query resolution.

    Args:
        config: The parent :class:`TransformerConfig`.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        d = config.d_model
        self.q_proj = nn.Linear(
            d, self.n_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            d, self.n_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            d, self.n_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, d, bias=False
        )

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: Optional[LayerKVCache] = None,
    ) -> Tensor:
        """Run causal GQA.

        Args:
            x: Input of shape ``(b, seq, d_model)``.
            cos: RoPE cosine table ``(1, 1, seq, head_dim)`` for *these* query
                positions (already sliced to account for any cache offset).
            sin: RoPE sine table ``(1, 1, seq, head_dim)``.
            cache: Optional per-layer KV cache. When given, the new keys/values
                are appended and attention runs over the full cached sequence.

        Returns:
            Output of shape ``(b, seq, d_model)``.
        """
        b, s, _ = x.shape

        q = (
            self.q_proj(x)
            .view(b, s, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )  # (b, H, s, hd)
        k = (
            self.k_proj(x)
            .view(b, s, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )  # (b, Hkv, s, hd)
        v = (
            self.v_proj(x)
            .view(b, s, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )  # (b, Hkv, s, hd)

        # RoPE is applied *before* caching so each key is rotated for its own
        # absolute position exactly once.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Append to (and read back) the KV cache. Compact KV heads are cached.
        if cache is not None:
            k, v = cache.update(k, v)
        kv_len = k.shape[2]

        # Expand KV heads to match the query heads (GQA).
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        if kv_len == s:
            # No cache / full-sequence prefill: use the fused causal fast path.
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
        else:
            # Incremental decode: the s new queries sit at positions
            # [cache_len, cache_len + s). Each attends to all earlier keys plus
            # the causal part of the new block.
            cache_len = kv_len - s
            q_pos = torch.arange(s, device=x.device) + cache_len
            k_pos = torch.arange(kv_len, device=x.device)
            attn_mask = (
                k_pos[None, :] <= q_pos[:, None]
            )  # (s, kv_len) bool
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask
            )

        out = out.transpose(1, 2).reshape(
            b, s, self.n_heads * self.head_dim
        )
        return self.o_proj(out)


# ----------------------------------------------------------------------------- #
# Transformer block
# ----------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """Pre-norm block: ``x + Attn(norm(x))`` then ``x + MoE(norm(x))``.

    Args:
        config: The parent :class:`TransformerConfig`.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.moe_norm = RMSNorm(config.d_model, config.norm_eps)
        self.moe = LatentMoE(config.moe_config())

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: Optional[LayerKVCache] = None,
    ) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.moe(self.moe_norm(x))
        return x


# ----------------------------------------------------------------------------- #
# Full model
# ----------------------------------------------------------------------------- #
class MoETransformer(nn.Module):
    """A decoder-only Transformer LM with GQA attention and LatentMoE FFNs.

    Args:
        config: A :class:`TransformerConfig`.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )
        # Weight tying (embedding <-> output projection).
        self.lm_head.weight = self.tok_emb.weight

        # GPT-2-style init: small normal weights so init logits are ~uniform
        # (init loss ~ ln(vocab)) instead of saturated.
        self.apply(self._init_weights)

        # Collected router-regularization loss, refreshed each forward.
        self.aux_loss: Tensor = torch.zeros(())

        # Resolve devices, place non-MoE params on the primary device (each MoE
        # already placed / sharded itself), and cache RoPE on the primary device.
        self.primary_device, _ = detect_devices(
            config.device, config.multi_gpu
        )
        self._place_non_moe()

        head_dim = config.d_model // config.n_heads
        cos, sin = build_rope_cache(
            config.max_seq_len,
            head_dim,
            config.rope_base,
            self.primary_device,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize Linear / Embedding weights ~ N(0, 0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ #
    def _place_non_moe(self) -> None:
        """Move every non-MoE parameter onto the primary device.

        The MoE sub-layers place (and possibly shard) themselves in their own
        constructors, so we deliberately avoid ``self.to(...)`` which would
        collapse that sharding.
        """
        dev = self.primary_device
        self.tok_emb.to(dev)
        self.norm.to(dev)
        self.lm_head.to(dev)
        for blk in self.blocks:
            blk.attn_norm.to(dev)
            blk.attn.to(dev)
            blk.moe_norm.to(dev)

    # ------------------------------------------------------------------ #
    def init_kv_cache(self) -> list[LayerKVCache]:
        """Create a fresh, empty KV cache (one :class:`LayerKVCache` per block)."""
        return [LayerKVCache() for _ in self.blocks]

    # ------------------------------------------------------------------ #
    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
        cache: Optional[list[LayerKVCache]] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Run the model.

        Args:
            idx: Token ids of shape ``(b, seq)``.
            targets: Optional next-token targets ``(b, seq)``. When given, the
                returned loss is cross-entropy plus the summed MoE aux/z losses.
            cache: Optional KV cache from :meth:`init_kv_cache`. When supplied,
                ``idx`` is treated as the tokens *following* what is already
                cached, and each block appends to its cache in place.

        Returns:
            ``(logits, loss)``. ``logits`` has shape ``(b, seq, vocab_size)``;
            ``loss`` is ``None`` when ``targets`` is ``None``.
        """
        _, s = idx.shape
        cache_len = cache[0].seq_len if cache is not None else 0
        if cache_len + s > self.config.max_seq_len:
            raise ValueError(
                f"position {cache_len + s} exceeds max_seq_len "
                f"{self.config.max_seq_len}."
            )
        idx = idx.to(self.primary_device)

        x = self.tok_emb(idx)  # (b, s, d)
        # Slice the RoPE tables at the current absolute positions.
        cos = self.rope_cos[:, :, cache_len : cache_len + s, :]
        sin = self.rope_sin[:, :, cache_len : cache_len + s, :]

        aux = torch.zeros((), device=self.primary_device)
        for i, blk in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            x = blk(x, cos, sin, layer_cache)
            aux = aux + blk.moe.aux_loss + blk.moe.z_loss
        self.aux_loss = aux

        x = self.norm(x)
        logits = self.lm_head(x)  # (b, s, vocab)

        loss: Optional[Tensor] = None
        if targets is not None:
            targets = targets.to(self.primary_device)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss = ce + aux
        return logits, loss

    # ------------------------------------------------------------------ #
    def _sample(
        self,
        logits: Tensor,
        temperature: float,
        top_k: Optional[int],
    ) -> Tensor:
        """Sample the next token from last-step logits ``(b, vocab)``."""
        if temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            vals, _ = torch.topk(logits, k, dim=-1)
            logits = logits.masked_fill(
                logits < vals[:, [-1]], -float("inf")
            )
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> Tensor:
        """Autoregressively sample ``max_new_tokens`` continuations.

        With ``use_cache`` (default), the prompt is processed once to fill the KV
        cache and each new step feeds only the single latest token — ``O(n)``
        overall instead of the ``O(n^2)`` of re-encoding the whole prefix.

        Args:
            idx: Prompt token ids ``(b, seq)``.
            max_new_tokens: Number of tokens to append.
            temperature: Softmax temperature (``0`` -> greedy argmax).
            top_k: If set, sample only from the ``top_k`` most likely tokens.
            use_cache: Use the KV cache for fast decoding.

        Returns:
            Token ids ``(b, seq + max_new_tokens)``.
        """
        self.eval()
        idx = idx.to(self.primary_device)

        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.max_seq_len :]
                logits, _ = self(idx_cond)
                nxt = self._sample(
                    logits[:, -1, :], temperature, top_k
                )
                idx = torch.cat((idx, nxt), dim=1)
            return idx

        # Cached path: prefill the prompt, then decode one token at a time.
        cache = self.init_kv_cache()
        logits, _ = self(idx, cache=cache)  # prefill
        for i in range(max_new_tokens):
            nxt = self._sample(logits[:, -1, :], temperature, top_k)
            idx = torch.cat((idx, nxt), dim=1)
            if i < max_new_tokens - 1:
                logits, _ = self(nxt, cache=cache)  # decode step
        return idx

    # ------------------------------------------------------------------ #
    def num_params(self, non_embedding: bool = False) -> int:
        """Total parameter count (tied head not double-counted)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n
