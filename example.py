import torch

from latent_moe import LatentMoE, LatentMoEConfig

# Configure a LatentMoE layer.
config = LatentMoEConfig(
    d=2048,  # model hidden dim
    m=1408,  # expert intermediate width
    n_experts=64,  # base routed experts (N)
    top_k=6,  # base active experts per token (K)
    alpha=4,  # latent compression factor (l = d / alpha)
    n_shared=2,  # always-on shared experts
    variant="acc",  # "acc" (iso-cost, higher accuracy) or "eff" (cheaper)
)

layer = LatentMoE(config)

# Drop it in like any FFN: (batch, seq, d) -> (batch, seq, d).
x = torch.randn(2, 128, config.d)
y = layer(x)

print(y.shape)  # torch.Size([2, 128, 2048])
