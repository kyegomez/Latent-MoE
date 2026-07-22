"""The ``grouped`` and ``loop`` dispatch paths must be numerically equivalent.

``grouped`` (sort tokens by expert, process contiguous segments) is the default
fast path; ``loop`` is the reference. They should agree on both the forward
output and the gradients, up to floating-point summation order.
"""

import torch

from latent_moe import LatentMoE, LatentMoEConfig


def _paired_layers(**overrides):
    """Build a (loop, grouped) pair sharing identical weights on CPU."""
    kwargs = dict(
        d=256,
        m=128,
        n_experts=16,
        top_k=3,
        alpha=2,
        n_shared=2,
        device="cpu",
    )
    kwargs.update(overrides)

    loop = LatentMoE(LatentMoEConfig(impl="loop", **kwargs))
    grouped = LatentMoE(LatentMoEConfig(impl="grouped", **kwargs))
    grouped.load_state_dict(loop.state_dict())  # identical params
    return loop, grouped


def test_forward_equivalence():
    torch.manual_seed(0)
    loop, grouped = _paired_layers()
    x = torch.randn(4, 64, loop.config.d)

    y_loop = loop(x)
    y_grouped = grouped(x)

    assert y_loop.shape == y_grouped.shape
    assert torch.allclose(y_loop, y_grouped, atol=1e-5, rtol=1e-4)
    # Router regularization terms must match too.
    assert torch.allclose(loop.aux_loss, grouped.aux_loss, atol=1e-6)
    assert torch.allclose(loop.z_loss, grouped.z_loss, atol=1e-6)


def test_backward_equivalence():
    torch.manual_seed(0)
    loop, grouped = _paired_layers()
    x = torch.randn(4, 64, loop.config.d)

    for layer in (loop, grouped):
        loss = layer(x).pow(2).mean() + layer.aux_loss + layer.z_loss
        loss.backward()

    assert torch.allclose(
        loop.router.weight.grad,
        grouped.router.weight.grad,
        atol=1e-5,
        rtol=1e-4,
    )
    for e in range(loop.config.n_experts_scaled):
        gl = loop.experts[e].fc1.weight.grad
        gg = grouped.experts[e].fc1.weight.grad
        assert torch.allclose(
            gl, gg, atol=1e-5, rtol=1e-4
        ), f"expert {e}"


def test_eff_variant_equivalence():
    """Also cover the eff variant (K' = K), a different dispatch shape."""
    torch.manual_seed(1)
    loop, grouped = _paired_layers(variant="eff", top_k=4)
    x = torch.randn(2, 48, loop.config.d)
    assert torch.allclose(loop(x), grouped(x), atol=1e-5, rtol=1e-4)


def test_grouped_is_default():
    assert LatentMoEConfig().impl == "grouped"
