## Latent-MoE

Implementation of <a href="https://arxiv.org/abs/2601.18089">LatentMoE</a> — *Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts* (Elango et al., NVIDIA 2026) — in Pytorch. A single-file, dependency-light layer you can drop in place of a standard MoE FFN.

The idea is simple. A standard MoE routes and computes its experts in the model hidden dimension `d`. LatentMoE first projects each token down into a smaller *latent* dimension `l = d / alpha` with a shared down-projection, runs all routed experts inside that latent space, then projects back up to `d`. Because dispatch traffic and expert weights now live in `l` rather than `d`, both all-to-all communication volume and per-expert weight-loading memory drop by a factor of `alpha`.

Those savings are reinvested by scaling the number of experts `N' = alpha * N`, exponentially expanding the space of expert combinations. Two flavors:

- `l-MoE_eff` — keep top-k `K` fixed → match baseline accuracy at lower inference cost.
- `l-MoE_acc` — scale top-k `K' = alpha * K` → match baseline cost while improving accuracy (recommended, Pareto-optimal).

The router and shared experts continue to operate in the original dimension `d`, since they are not the memory/communication bottleneck.

## Install

```bash
uv pip install latent-moe
```

## Usage

```python
import torch
from latent_moe import LatentMoE, LatentMoEConfig

config = LatentMoEConfig(
    d = 2048,          # model hidden dim
    m = 1408,          # expert intermediate width
    n_experts = 64,    # base routed experts (N)
    top_k = 6,         # base active experts per token (K)
    alpha = 4,         # latent compression factor (l = d / alpha)
    n_shared = 2,      # always-on shared experts
    variant = "acc",   # "acc" (iso-cost, higher accuracy) or "eff" (cheaper)
)

layer = LatentMoE(config)

x = torch.randn(2, 128, config.d)  # (batch, seq, d)
y = layer(x)                       # (batch, seq, d)

assert y.shape == x.shape
```

Inspect the asymptotic cost quantities from Table 1 of the paper:

```python
for k, v in layer.cost_summary().items():
    print(f"{k}: {v:,.2f}")
```

## Citations

```bibtex
@article{elango2026latentmoe,
    title   = {LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts},
    author  = {Elango and others},
    journal = {arXiv preprint arXiv:2601.18089},
    year    = {2026},
}
```
