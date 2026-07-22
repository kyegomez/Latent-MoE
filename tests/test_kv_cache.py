"""The KV cache must not change model outputs, only make decoding faster."""

import torch

from latent_moe import MoETransformer, TransformerConfig


def _model():
    torch.manual_seed(0)
    cfg = TransformerConfig(
        vocab_size=256,
        d_model=128,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,  # GQA
        max_seq_len=128,
        d_ff=128,
        n_experts=8,
        top_k=2,
        alpha=2,
        n_shared=1,
        device="cpu",
    )
    return MoETransformer(cfg).eval()


def test_prefill_matches_full_forward():
    """Filling the cache with the prompt yields the same logits as a plain pass."""
    model = _model()
    idx = torch.randint(0, model.config.vocab_size, (2, 12))

    cache = model.init_kv_cache()
    logits_cached, _ = model(idx, cache=cache)
    logits_full, _ = model(idx)

    assert torch.allclose(logits_cached, logits_full, atol=1e-5)
    assert cache[0].seq_len == 12


def test_greedy_generation_equivalence():
    """Greedy decoding is identical with and without the cache."""
    model = _model()
    prompt = torch.randint(0, model.config.vocab_size, (2, 10))

    out_nocache = model.generate(
        prompt, max_new_tokens=25, temperature=0.0, use_cache=False
    )
    out_cache = model.generate(
        prompt, max_new_tokens=25, temperature=0.0, use_cache=True
    )
    assert torch.equal(out_nocache, out_cache)


def test_incremental_decode_matches_full():
    """Token-by-token cached decoding matches re-encoding the whole prefix."""
    model = _model()
    idx = torch.randint(0, model.config.vocab_size, (1, 6))

    # Reference: full forward over the whole sequence.
    ref_logits, _ = model(idx)

    # Cached: feed the first 4 tokens, then the last 2 one at a time.
    cache = model.init_kv_cache()
    model(idx[:, :4], cache=cache)
    model(idx[:, 4:5], cache=cache)
    step_logits, _ = model(idx[:, 5:6], cache=cache)

    assert cache[0].seq_len == 6
    assert torch.allclose(
        step_logits[:, -1, :], ref_logits[:, -1, :], atol=1e-5
    )


def test_cache_reset():
    model = _model()
    cache = model.init_kv_cache()
    model(
        torch.randint(0, model.config.vocab_size, (1, 5)), cache=cache
    )
    assert cache[0].seq_len == 5
    for layer in cache:
        layer.reset()
    assert cache[0].seq_len == 0
