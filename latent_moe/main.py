"""
LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
================================================================================

A single-file, dependency-light PyTorch implementation of the LatentMoE
architecture from Elango et al. (NVIDIA, 2026), arXiv:2601.18089.

Core idea
---------
A standard MoE routes/computes experts in the model hidden dimension ``d``.
LatentMoE instead projects each token into a smaller *latent* dimension
``l = d / alpha`` via a shared down-projection, runs all routed experts inside
that latent space, then projects back up to ``d``. Because dispatch traffic and
expert weights now live in ``l`` rather than ``d``, both all-to-all communication
volume and per-expert weight-loading memory cost drop by a factor of ``alpha``.

Those savings are reinvested (Design Principle V): the total number of routed
experts ``N`` is scaled by ``alpha`` (``N' = alpha * N``), which exponentially
expands the space of expert combinations. Two configurations exist:

    * l-MoE_eff : keep top-k ``K`` fixed  -> matches baseline accuracy at lower
                  inference cost (cheaper communication + memory).
    * l-MoE_acc : scale top-k to ``K' = alpha * K`` -> matches baseline inference
                  cost while improving accuracy (the recommended, Pareto-optimal
                  variant).

The intermediate FFN width ``m`` is held constant so the effective non-linear
budget ``U_eff ~ K * m`` is preserved (Design Principle III). Shared experts and
the router continue to operate in the original dimension ``d``, as they are not
the memory/communication bottleneck.

Reference eq. (2) of the paper:

    l-MoE_acc(x) = W_up @ ( sum_{i in TopK'} p'_i * E_i(W_down @ x ; l) )
                   + sum_{j in shared} E_j(x ; d)

This file favors clarity over kernel-level performance: experts are looped in
Python rather than fused, and no expert-parallel all-to-all is implemented.
The tensor shapes and cost structure, however, mirror the paper exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Variant = Literal["acc", "eff"]


# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #
@dataclass
class LatentMoEConfig:
    """Configuration for a :class:`LatentMoE` layer.

    Attributes:
        d: Model hidden dimension (the residual-stream width).
        m: Intermediate feed-forward dimension of each expert. Held constant to
            preserve the effective non-linear budget ``U_eff ~ K * m``.
        n_experts: Base number of routed experts ``N`` (before latent scaling).
        top_k: Base number of active experts per token ``K`` (before scaling).
        alpha: Compression / scaling factor ``alpha = d / l``. Must divide ``d``.
            The paper finds ``alpha <= 4`` preserves quality for typical models.
        n_shared: Number of shared experts ``S`` that always process every token
            in the *full* dimension ``d``.
        variant: ``"acc"`` scales both experts and top-k by ``alpha`` (recommended,
            iso-cost, higher accuracy). ``"eff"`` scales only the expert count,
            keeping top-k fixed (iso-accuracy, lower cost).
        bias: Whether expert / projection linear layers use bias terms.
    """

    d: int = 2048
    m: int = 1408
    n_experts: int = 64
    top_k: int = 6
    alpha: int = 4
    n_shared: int = 2
    variant: Variant = "acc"
    bias: bool = False

    # -- Derived quantities (populated in __post_init__). --------------------- #
    latent_dim: int = 0  # l = d / alpha
    n_experts_scaled: int = 0  # N' = alpha * N
    top_k_scaled: int = 0  # K' = alpha * K (acc) or K (eff)

    def __post_init__(self) -> None:
        if self.d % self.alpha != 0:
            raise ValueError(
                f"alpha={self.alpha} must divide d={self.d} so that "
                f"the latent dimension l = d / alpha is an integer."
            )
        if self.variant not in ("acc", "eff"):
            raise ValueError(
                f"variant must be 'acc' or 'eff', got {self.variant!r}"
            )

        self.latent_dim = self.d // self.alpha  # l
        self.n_experts_scaled = (
            self.alpha * self.n_experts
        )  # N' = alpha * N

        # l-MoE_acc scales top-k by alpha; l-MoE_eff keeps it fixed.
        if self.variant == "acc":
            self.top_k_scaled = self.alpha * self.top_k
        else:
            self.top_k_scaled = self.top_k

        if self.top_k_scaled > self.n_experts_scaled:
            raise ValueError(
                f"Effective top_k ({self.top_k_scaled}) cannot exceed the number "
                f"of routed experts ({self.n_experts_scaled})."
            )


# ----------------------------------------------------------------------------- #
# Expert
# ----------------------------------------------------------------------------- #
class Expert(nn.Module):
    """A single gated feed-forward expert (SwiGLU-style).

    Maps ``in_dim -> m -> in_dim`` with a gated non-linearity:

        h = act(FC1(x)) * gate(x)   (element-wise, both project in_dim -> m)
        y = FC2(h)                  (projects m -> in_dim)

    Routed experts use ``in_dim = l`` (latent); shared experts use ``in_dim = d``.
    The intermediate width ``m`` is identical in both cases, keeping ``U_eff``
    invariant to the latent compression.

    Args:
        in_dim: Input/output dimension (``l`` for routed, ``d`` for shared).
        m: Intermediate feed-forward width.
        bias: Whether the linear layers include bias terms.
    """

    def __init__(
        self, in_dim: int, m: int, bias: bool = False
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            in_dim, m, bias=bias
        )  # W_FC1 : (m, in_dim)
        self.gate = nn.Linear(
            in_dim, m, bias=bias
        )  # W_gate: (m, in_dim)
        self.fc2 = nn.Linear(
            m, in_dim, bias=bias
        )  # W_FC2 : (in_dim, m)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the expert.

        Args:
            x: Tokens of shape ``(..., in_dim)``.

        Returns:
            Tensor of shape ``(..., in_dim)``.
        """
        return self.fc2(F.silu(self.fc1(x)) * self.gate(x))


# ----------------------------------------------------------------------------- #
# LatentMoE layer
# ----------------------------------------------------------------------------- #
class LatentMoE(nn.Module):
    """LatentMoE feed-forward layer (drop-in replacement for a standard MoE FFN).

    Pipeline for a token ``x`` of dimension ``d``:

        1. Down-project:  z   = W_down @ x            (d -> l)
        2. Route:         p'  = softmax(W_r @ x)      (router reads full x, R^N')
        3. Select:        TopK' experts by router score.
        4. Compute:       for each selected expert i, E_i(z ; l)  in latent space,
                          weighted by the (renormalized) routing prob p'_i.
        5. Up-project:    y   = W_up @ (weighted sum)  (l -> d)
        6. Shared:        y  += sum_j E_j(x ; d)       (full-dim shared experts)

    Only the routed-expert path is compressed; the router and shared experts stay
    in dimension ``d`` because they are not the dominant memory / comms cost.

    Args:
        config: A :class:`LatentMoEConfig` instance.
    """

    def __init__(self, config: LatentMoEConfig) -> None:
        super().__init__()
        self.config = config
        d, l, m = config.d, config.latent_dim, config.m

        # Shared latent projections. W_down: (l, d), W_up: (d, l).
        self.w_down = nn.Linear(d, l, bias=config.bias)
        self.w_up = nn.Linear(l, d, bias=config.bias)

        # Router reads the *original* token x in R^d and scores all N' experts.
        self.router = nn.Linear(
            d, config.n_experts_scaled, bias=False
        )

        # Routed experts live in the latent space (in_dim = l).
        self.experts = nn.ModuleList(
            Expert(l, m, bias=config.bias)
            for _ in range(config.n_experts_scaled)
        )

        # Shared experts always run in the full dimension (in_dim = d).
        self.shared_experts = nn.ModuleList(
            Expert(d, m, bias=config.bias)
            for _ in range(config.n_shared)
        )

    # ------------------------------------------------------------------ #
    def _route(self, x_flat: Tensor) -> tuple[Tensor, Tensor]:
        """Compute top-k expert indices and renormalized gating weights.

        Args:
            x_flat: Tokens of shape ``(T, d)`` (T = tokens in the batch).

        Returns:
            A tuple ``(topk_idx, topk_weight)`` where ``topk_idx`` has shape
            ``(T, K')`` (long) and ``topk_weight`` has shape ``(T, K')`` (float),
            with weights renormalized to sum to 1 over the selected experts.
        """
        logits = self.router(x_flat)  # (T, N')
        probs = F.softmax(logits, dim=-1)  # (T, N')
        topk_weight, topk_idx = probs.topk(  # (T, K'), (T, K')
            self.config.top_k_scaled, dim=-1
        )
        # Renormalize over the selected experts (standard MoE practice).
        topk_weight = topk_weight / topk_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)
        return topk_idx, topk_weight

    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:
        """Run the LatentMoE layer.

        Args:
            x: Input activations of shape ``(B, S, d)`` or ``(T, d)``.

        Returns:
            Output activations of the same shape as ``x``.
        """
        orig_shape = x.shape
        d = self.config.d
        x_flat = x.reshape(-1, d)  # (T, d)
        T = x_flat.shape[0]

        # --- Routing (in full dimension d). ---
        topk_idx, topk_weight = self._route(
            x_flat
        )  # (T, K'), (T, K')

        # --- Down-project once; all routed experts share this latent token. ---
        z = self.w_down(x_flat)  # (T, l)

        # --- Dispatch to experts. Loop over experts (clear, not fused). ---
        latent_out = torch.zeros_like(z)  # (T, l)
        flat_idx = topk_idx.reshape(-1)  # (T * K',)
        flat_weight = topk_weight.reshape(-1)  # (T * K',)
        token_ids = torch.arange(
            T, device=x.device
        ).repeat_interleave(
            self.config.top_k_scaled
        )  # (T * K',)

        for e, expert in enumerate(self.experts):
            sel = flat_idx == e  # mask into the (T*K') list
            if not torch.any(sel):
                continue
            tok = token_ids[sel]  # tokens routed to expert e
            out = expert(z[tok])  # (n_e, l)
            latent_out.index_add_(
                0, tok, out * flat_weight[sel].unsqueeze(-1)
            )

        # --- Up-project back to d and add shared-expert contributions. ---
        y = self.w_up(latent_out)  # (T, d)
        for shared in self.shared_experts:
            y = y + shared(x_flat)  # full-dim path

        return y.reshape(orig_shape)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def cost_summary(
        self, t_exp: float = 256.0, ep: int = 1
    ) -> dict[str, float]:
        """Report the paper's asymptotic per-expert cost quantities.

        These mirror Table 1: communication volume ``~ (N/EP) * t_exp * l`` and
        per-expert weight-loading memory ``~ l * m`` for LatentMoE, versus
        ``* d`` and ``d * m`` respectively for a standard MoE.

        Args:
            t_exp: Average tokens routed to a single expert.
            ep: Expert-parallel degree (GPUs experts are sharded across).

        Returns:
            Dict of cost quantities (arbitrary units) for comparison.
        """
        c = self.config
        comm_latent = (c.n_experts_scaled / ep) * t_exp * c.latent_dim
        comm_standard = (c.n_experts / ep) * t_exp * c.d
        return {
            "comm_volume_latent": comm_latent,
            "comm_volume_standard": comm_standard,
            "comm_ratio_vs_standard": comm_latent / comm_standard,
            "weight_mem_per_expert_latent": c.latent_dim * c.m,
            "weight_mem_per_expert_standard": c.d * c.m,
            "u_eff_latent": c.top_k_scaled
            * c.m,  # K' * m (acc) or K * m (eff)
            "u_eff_standard": c.top_k * c.m,  # K * m
        }
