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
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Variant = Literal["acc", "eff"]
Impl = Literal["loop", "grouped"]


# ----------------------------------------------------------------------------- #
# Device detection / placement
# ----------------------------------------------------------------------------- #
def detect_devices(
    device: Optional[str] = None,
    multi_gpu: bool = True,
) -> tuple[torch.device, list[torch.device]]:
    """Auto-detect the best available compute device(s).

    Selection order when ``device is None``:

        1. CUDA  -> use every visible GPU (for expert sharding) if ``multi_gpu``,
           otherwise just ``cuda:0``.
        2. MPS   -> Apple-Silicon single-device.
        3. CPU   -> fallback.

    Args:
        device: Force a specific device string (e.g. ``"cpu"``, ``"cuda"``,
            ``"cuda:1"``, ``"mps"``). When given, no auto-detection happens and
            everything is placed on that single device.
        multi_gpu: When ``True`` and more than one CUDA GPU is visible, return the
            full GPU list so routed experts can be sharded across them.

    Returns:
        A tuple ``(primary_device, expert_devices)``. ``primary_device`` holds the
        router, projections and shared experts; ``expert_devices`` is the pool the
        routed experts are round-robin-distributed over (length 1 for single-device).
    """
    if device is not None:
        dev = torch.device(device)
        return dev, [dev]

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        gpus = [torch.device(f"cuda:{i}") for i in range(n)]
        primary = gpus[0]
        expert_pool = gpus if (multi_gpu and n > 1) else [primary]
        return primary, expert_pool

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        dev = torch.device("mps")
        return dev, [dev]

    dev = torch.device("cpu")
    return dev, [dev]


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
        device: Force placement on a specific device (e.g. ``"cpu"``, ``"cuda"``,
            ``"cuda:0"``, ``"mps"``). When ``None`` (default), the best available
            device is auto-detected.
        multi_gpu: When ``True`` (default) and more than one CUDA GPU is visible,
            the routed experts are round-robin-sharded across every GPU so the
            layer is ready for single-process multi-GPU (expert-parallel) training.
        aux_loss_coef: Weight of the load-balancing auxiliary loss. Without it the
            router collapses onto a handful of experts; the standard Switch/GShard
            term pushes tokens to spread evenly. Set to ``0.0`` to disable.
        z_loss_coef: Weight of the router z-loss, which keeps the router logits
            small and numerically stable. Set to ``0.0`` to disable.
        impl: Expert-dispatch strategy. ``"grouped"`` (default) sorts tokens by
            expert once and processes contiguous per-expert segments — faster,
            fewer kernel launches, numerically equivalent to the loop. ``"loop"``
            is the simplest reference. When experts are sharded across multiple
            devices, ``"grouped"`` automatically falls back to ``"loop"``.
    """

    d: int = 2048
    m: int = 1408
    n_experts: int = 64
    top_k: int = 6
    alpha: int = 4
    n_shared: int = 2
    variant: Variant = "acc"
    bias: bool = False
    device: Optional[str] = None
    multi_gpu: bool = True
    aux_loss_coef: float = 1e-2
    z_loss_coef: float = 1e-3
    impl: Impl = "grouped"

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
        if self.impl not in ("loop", "grouped"):
            raise ValueError(
                f"impl must be 'loop' or 'grouped', got {self.impl!r}"
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
        d, latent, m = config.d, config.latent_dim, config.m

        # Shared latent projections. W_down: (l, d), W_up: (d, l).
        self.w_down = nn.Linear(d, latent, bias=config.bias)
        self.w_up = nn.Linear(latent, d, bias=config.bias)

        # Router reads the *original* token x in R^d and scores all N' experts.
        self.router = nn.Linear(
            d, config.n_experts_scaled, bias=False
        )

        # Routed experts live in the latent space (in_dim = l).
        self.experts = nn.ModuleList(
            Expert(latent, m, bias=config.bias)
            for _ in range(config.n_experts_scaled)
        )

        # Shared experts always run in the full dimension (in_dim = d).
        self.shared_experts = nn.ModuleList(
            Expert(d, m, bias=config.bias)
            for _ in range(config.n_shared)
        )

        # Auto-detect devices and place / shard the sub-modules accordingly.
        self.primary_device: torch.device = torch.device("cpu")
        self.expert_devices: list[torch.device] = []
        self.distribute()

        # Router regularization losses, refreshed every forward pass. A trainer
        # should add ``layer.aux_loss + layer.z_loss`` to the main task loss.
        self.aux_loss: Tensor = torch.zeros(
            (), device=self.primary_device
        )
        self.z_loss: Tensor = torch.zeros(
            (), device=self.primary_device
        )

    # ------------------------------------------------------------------ #
    def distribute(
        self,
        device: Optional[str] = None,
        multi_gpu: Optional[bool] = None,
    ) -> "LatentMoE":
        """(Re)detect devices and place / shard the layer's parameters.

        Called automatically at construction. The router, projections and shared
        experts go on the primary device; the routed experts are round-robin
        distributed across every device in the detected pool (all visible CUDA
        GPUs when ``multi_gpu`` is set), giving single-process expert parallelism.

        Note:
            Calling ``.to(...)`` / ``.cuda()`` on the module afterwards collapses
            every expert onto one device. Re-run :meth:`distribute` to re-shard.

        Args:
            device: Override :attr:`config.device` for this call only.
            multi_gpu: Override :attr:`config.multi_gpu` for this call only.

        Returns:
            ``self`` (for chaining).
        """
        dev = device if device is not None else self.config.device
        mg = (
            multi_gpu
            if multi_gpu is not None
            else self.config.multi_gpu
        )

        primary, expert_pool = detect_devices(
            device=dev, multi_gpu=mg
        )
        self.primary_device = primary

        # Router, projections and shared experts live on the primary device.
        self.w_down.to(primary)
        self.w_up.to(primary)
        self.router.to(primary)
        self.shared_experts.to(primary)

        # Round-robin the routed experts across the device pool.
        self.expert_devices = []
        for i, expert in enumerate(self.experts):
            expert_dev = expert_pool[i % len(expert_pool)]
            expert.to(expert_dev)
            self.expert_devices.append(expert_dev)

        return self

    # ------------------------------------------------------------------ #
    def device_summary(self) -> dict[str, object]:
        """Report how the layer is currently placed across devices.

        Returns:
            Dict with the primary device and a per-device count of routed experts.
        """
        counts: dict[str, int] = {}
        for dev in self.expert_devices:
            counts[str(dev)] = counts.get(str(dev), 0) + 1
        return {
            "primary_device": str(self.primary_device),
            "num_expert_devices": len(set(self.expert_devices)),
            "experts_per_device": counts,
        }

    # ------------------------------------------------------------------ #
    def _route(self, logits: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute top-k expert indices, gating weights and the full soft probs.

        Args:
            logits: Router logits of shape ``(T, N')`` (T = tokens in the batch).

        Returns:
            A tuple ``(topk_idx, topk_weight, probs)`` where ``topk_idx`` has shape
            ``(T, K')`` (long), ``topk_weight`` has shape ``(T, K')`` (float,
            renormalized to sum to 1 over the selected experts), and ``probs`` is
            the full softmax distribution ``(T, N')`` used by the balancing loss.
        """
        probs = F.softmax(logits, dim=-1)  # (T, N')
        topk_weight, topk_idx = probs.topk(  # (T, K'), (T, K')
            self.config.top_k_scaled, dim=-1
        )
        # Renormalize over the selected experts (standard MoE practice).
        topk_weight = topk_weight / topk_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)
        return topk_idx, topk_weight, probs

    # ------------------------------------------------------------------ #
    def _balancing_losses(
        self, logits: Tensor, probs: Tensor, topk_idx: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute the load-balancing auxiliary loss and the router z-loss.

        The auxiliary loss is the Switch/GShard term
        ``N' * sum_e f_e * P_e``, where ``f_e`` is the fraction of routing slots
        that landed on expert ``e`` and ``P_e`` is the mean router probability for
        expert ``e``. It is minimized when load is spread uniformly, preventing the
        router from collapsing onto a few experts. The z-loss
        ``mean_t (logsumexp(logits_t))^2`` keeps the logits from drifting large.

        Args:
            logits: Router logits ``(T, N')``.
            probs: Full softmax probabilities ``(T, N')``.
            topk_idx: Selected expert indices ``(T, K')``.

        Returns:
            A tuple ``(aux_loss, z_loss)``, each already scaled by its coefficient.
        """
        c = self.config
        # Fraction of dispatch slots that landed on each expert, f_e (sums to 1).
        mask = torch.zeros_like(probs)  # (T, N')
        mask.scatter_(1, topk_idx, 1.0)
        f = mask.sum(dim=0) / mask.sum().clamp_min(1.0)  # (N',)
        # Mean router probability per expert, P_e (sums to 1).
        prob_mean = probs.mean(dim=0)  # (N',)
        # Switch/GShard aux loss: minimized (== 1.0) under uniform load.
        aux = c.n_experts_scaled * torch.sum(f * prob_mean)
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        return c.aux_loss_coef * aux, c.z_loss_coef * z

    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor) -> Tensor:
        """Run the LatentMoE layer.

        Side effect: refreshes :attr:`aux_loss` and :attr:`z_loss` with the
        router-regularization terms for this batch. Add them to your task loss.

        Args:
            x: Input activations of shape ``(B, S, d)`` or ``(T, d)``.

        Returns:
            Output activations of the same shape as ``x``.
        """
        orig_shape = x.shape
        d = self.config.d
        # Everything but the routed experts lives on the primary device; move the
        # input there so the output comes back on the same device the caller used.
        input_device = x.device
        x_flat = x.reshape(-1, d).to(self.primary_device)  # (T, d)
        T = x_flat.shape[0]

        # --- Routing (in full dimension d). ---
        logits = self.router(x_flat)  # (T, N')
        topk_idx, topk_weight, probs = self._route(
            logits
        )  # (T, K') ...
        self.aux_loss, self.z_loss = self._balancing_losses(
            logits, probs, topk_idx
        )

        # --- Down-project once; all routed experts share this latent token. ---
        z = self.w_down(x_flat)  # (T, l)

        # --- Flatten the (token, slot) dispatch lists. ---
        flat_idx = topk_idx.reshape(-1)  # (T * K',)
        flat_weight = topk_weight.reshape(-1)  # (T * K',)
        token_ids = torch.arange(
            T, device=self.primary_device
        ).repeat_interleave(
            self.config.top_k_scaled
        )  # (T * K',)

        # --- Dispatch to experts. Grouped is faster; loop handles sharding. ---
        sharded = len(set(self.expert_devices)) > 1
        if self.config.impl == "grouped" and not sharded:
            latent_out = self._dispatch_grouped(
                z, flat_idx, flat_weight, token_ids
            )
        else:
            latent_out = self._dispatch_loop(
                z, flat_idx, flat_weight, token_ids
            )

        # --- Up-project back to d and add shared-expert contributions. ---
        y = self.w_up(latent_out)  # (T, d)
        for shared in self.shared_experts:
            y = y + shared(x_flat)  # full-dim path

        # Return on the caller's original device.
        return y.reshape(orig_shape).to(input_device)

    # ------------------------------------------------------------------ #
    def _dispatch_loop(
        self,
        z: Tensor,
        flat_idx: Tensor,
        flat_weight: Tensor,
        token_ids: Tensor,
    ) -> Tensor:
        """Reference dispatch: iterate experts, gather-compute-scatter each.

        Handles experts sharded across multiple devices by shipping the selected
        latent tokens to each expert's device. Clear over fast.

        Args:
            z: Latent tokens ``(T, l)`` on the primary device.
            flat_idx: Expert id per dispatch slot ``(T * K',)``.
            flat_weight: Gating weight per dispatch slot ``(T * K',)``.
            token_ids: Source token id per dispatch slot ``(T * K',)``.

        Returns:
            Accumulated latent output ``(T, l)`` on the primary device.
        """
        latent_out = torch.zeros_like(z)  # (T, l), on primary device

        # Copy the latent tokens to each expert device *once* (not once per
        # expert): O(#devices) transfers instead of O(N'). No-op single-device.
        z_by_device: dict[torch.device, Tensor] = {}
        for dev in self.expert_devices:
            if dev not in z_by_device:
                z_by_device[dev] = z.to(dev)

        for e, expert in enumerate(self.experts):
            sel = flat_idx == e  # mask into the (T*K') list
            if not torch.any(sel):
                continue
            tok = token_ids[sel]  # tokens routed to expert e
            expert_dev = self.expert_devices[e]
            # Index the device-local latent copy, compute on the expert's device,
            # then bring the result back to the primary device for accumulation.
            z_local = z_by_device[expert_dev][tok.to(expert_dev)]
            out = expert(z_local).to(self.primary_device)  # (n_e, l)
            latent_out.index_add_(
                0, tok, out * flat_weight[sel].unsqueeze(-1)
            )
        return latent_out

    # ------------------------------------------------------------------ #
    def _dispatch_grouped(
        self,
        z: Tensor,
        flat_idx: Tensor,
        flat_weight: Tensor,
        token_ids: Tensor,
    ) -> Tensor:
        """Vectorized single-device dispatch: sort by expert, process segments.

        Sorting the dispatch slots by expert id turns per-expert selection into
        contiguous slicing — no ``O(N')`` boolean masks over the whole slot list,
        and a single scatter-add at the end instead of one per expert. Numerically
        equivalent to :meth:`_dispatch_loop` (up to float summation order).

        Args:
            z: Latent tokens ``(T, l)`` on the primary device.
            flat_idx: Expert id per dispatch slot ``(T * K',)``.
            flat_weight: Gating weight per dispatch slot ``(T * K',)``.
            token_ids: Source token id per dispatch slot ``(T * K',)``.

        Returns:
            Accumulated latent output ``(T, l)`` on the primary device.
        """
        # Sort slots so every expert owns a contiguous run.
        order = torch.argsort(flat_idx)
        sorted_tok = token_ids[order]  # (S,)
        sorted_w = flat_weight[order]  # (S,)
        z_sorted = z[sorted_tok]  # (S, l), gathered once

        # Per-expert segment boundaries via token counts.
        counts = torch.bincount(
            flat_idx, minlength=self.config.n_experts_scaled
        )  # (N',)
        offsets = torch.cumsum(counts, dim=0)  # segment ends
        counts_list = counts.tolist()  # host-side loop bounds
        ends = offsets.tolist()

        out_sorted = torch.empty_like(z_sorted)  # (S, l)
        start = 0
        for e, n_e in enumerate(counts_list):
            if n_e == 0:
                continue
            end = ends[e]
            out_sorted[start:end] = self.experts[e](
                z_sorted[start:end]
            )
            start = end

        # Weight and scatter-add every contribution back in one shot.
        latent_out = torch.zeros_like(z)  # (T, l)
        latent_out.index_add_(
            0, sorted_tok, out_sorted * sorted_w.unsqueeze(-1)
        )
        return latent_out

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
