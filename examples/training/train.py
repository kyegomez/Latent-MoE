from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from loguru import logger

from latent_moe.transformer import MoETransformer, TransformerConfig


# ----------------------------------------------------------------------------- #
# Configuration
# ----------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    # Data.
    dataset: str = "wikimedia/wikipedia"
    dataset_config: str = "20231101.en"
    num_articles: int = 2000
    val_fraction: float = 0.05

    # Model (kept small so it trains on a laptop).
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ff: int = 512
    n_experts: int = 16
    top_k: int = 4
    alpha: int = 2
    n_shared: int = 1

    # Optimization.
    steps: int = 2000
    batch_size: int = 8
    block_size: int = 256
    muon_lr: float = 0.02
    adam_lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Logistics.
    save_every: int = 100
    eval_every: int = 250
    eval_iters: int = 20
    log_every: int = 10
    ckpt_dir: str = "checkpoints"
    seed: int = 0
    device: Optional[str] = None  # None -> auto-detect (cuda/mps/cpu)
    resume: bool = False


# ----------------------------------------------------------------------------- #
# Data
# ----------------------------------------------------------------------------- #
def load_token_stream(cfg: TrainConfig) -> tuple[torch.Tensor, int]:
    """Stream ``num_articles`` Wikipedia articles and tokenize into one 1D tensor.

    Returns:
        ``(tokens, vocab_size)`` where ``tokens`` is a 1D ``long`` tensor of token
        ids and ``vocab_size`` is the tokenizer's vocabulary size.
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - guidance path
        raise SystemExit(
            "Missing training deps. Run: pip install -r requirements-train.txt"
        ) from exc

    logger.info(f"loading tokenizer (gpt2) and dataset {cfg.dataset}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(
        1e9
    )  # silence long-sequence warnings
    eos = tokenizer.eos_token_id

    stream = load_dataset(
        cfg.dataset, cfg.dataset_config, split="train", streaming=True
    )

    ids: list[int] = []
    for i, example in enumerate(stream):
        if i >= cfg.num_articles:
            break
        ids.extend(tokenizer.encode(example["text"]))
        ids.append(eos)
        if (i + 1) % 250 == 0:
            logger.info(
                f"tokenized {i + 1}/{cfg.num_articles} articles "
                f"({len(ids):,} tokens)"
            )

    tokens = torch.tensor(ids, dtype=torch.long)
    logger.info(
        f"token stream ready: {len(tokens):,} tokens, "
        f"vocab_size={tokenizer.vocab_size}"
    )
    return tokens, tokenizer.vocab_size


def make_batch_fn(
    tokens: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
):
    """Build a ``get_batch(split)`` closure over a train/val token split."""
    n_val = int(len(tokens) * cfg.val_fraction)
    train_data = tokens[:-n_val] if n_val > 0 else tokens
    val_data = tokens[-n_val:] if n_val > 0 else tokens
    logger.info(
        f"split: {len(train_data):,} train / {len(val_data):,} val tokens"
    )

    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = train_data if split == "train" else val_data
        high = len(data) - cfg.block_size - 1
        ix = torch.randint(high, (cfg.batch_size,))
        x = torch.stack([data[i : i + cfg.block_size] for i in ix])
        y = torch.stack(
            [data[i + 1 : i + 1 + cfg.block_size] for i in ix]
        )
        return x.to(device), y.to(device)

    return get_batch


# ----------------------------------------------------------------------------- #
# Optimizers (Muon for 2D hidden weights, AdamW for the rest)
# ----------------------------------------------------------------------------- #
def build_optimizers(
    model: MoETransformer, cfg: TrainConfig
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    """Split params: Muon for 2D hidden matrices, AdamW for embeddings / 1D.

    The token embedding (tied to the LM head) is 2D but must be optimized by
    AdamW, not Muon — Muon is for hidden weight matrices only.
    """
    embed_id = id(model.tok_emb.weight)
    muon_params, adam_params = [], []
    seen: set[int] = set()
    for _, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim == 2 and id(p) != embed_id:
            muon_params.append(p)
        else:
            adam_params.append(p)

    n_muon = sum(p.numel() for p in muon_params)
    n_adam = sum(p.numel() for p in adam_params)
    logger.info(
        f"muon params: {len(muon_params)} tensors / {n_muon:,} | "
        f"adamw params: {len(adam_params)} tensors / {n_adam:,}"
    )

    muon = torch.optim.Muon(
        muon_params,
        lr=cfg.muon_lr,
        momentum=0.95,
        weight_decay=cfg.weight_decay,
    )
    adamw = torch.optim.AdamW(
        adam_params,
        lr=cfg.adam_lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )
    return muon, adamw


def lr_scale(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to zero, returned as a multiplier."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ----------------------------------------------------------------------------- #
# Checkpointing
# ----------------------------------------------------------------------------- #
def save_checkpoint(
    cfg: TrainConfig,
    step: int,
    model: MoETransformer,
    muon: torch.optim.Optimizer,
    adamw: torch.optim.Optimizer,
    val_loss: Optional[float],
) -> None:
    """Save full training state to ``<ckpt_dir>/step_<step>.pt`` and ``latest.pt``."""
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "muon": muon.state_dict(),
        "adamw": adamw.state_dict(),
        "train_config": asdict(cfg),
        "val_loss": val_loss,
    }
    step_path = os.path.join(cfg.ckpt_dir, f"step_{step:06d}.pt")
    latest_path = os.path.join(cfg.ckpt_dir, "latest.pt")
    torch.save(payload, step_path)
    torch.save(payload, latest_path)
    logger.success(f"saved checkpoint -> {step_path}")


def maybe_resume(
    cfg: TrainConfig,
    model: MoETransformer,
    muon: torch.optim.Optimizer,
    adamw: torch.optim.Optimizer,
) -> int:
    """Load ``latest.pt`` if ``--resume`` and it exists. Returns the start step."""
    latest = os.path.join(cfg.ckpt_dir, "latest.pt")
    if not (cfg.resume and os.path.exists(latest)):
        return 0
    ckpt = torch.load(latest, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    muon.load_state_dict(ckpt["muon"])
    adamw.load_state_dict(ckpt["adamw"])
    start = int(ckpt["step"]) + 1
    logger.info(f"resumed from {latest} at step {start}")
    return start


# ----------------------------------------------------------------------------- #
# Evaluation
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def estimate_val_loss(
    model: MoETransformer, get_batch, eval_iters: int
) -> float:
    """Average loss over ``eval_iters`` validation batches."""
    model.eval()
    total = 0.0
    for _ in range(eval_iters):
        x, y = get_batch("val")
        _, loss = model(x, y)
        total += float(loss)
    model.train()
    return total / eval_iters


# ----------------------------------------------------------------------------- #
# Training loop
# ----------------------------------------------------------------------------- #
def train(cfg: TrainConfig) -> None:
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    logger.add(
        os.path.join(cfg.ckpt_dir, "train.log"),
        level="INFO",
        rotation="10 MB",
    )
    logger.info(f"config: {asdict(cfg)}")
    torch.manual_seed(cfg.seed)

    # -- Data. --
    tokens, vocab_size = load_token_stream(cfg)

    # -- Model. --
    model_cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        max_seq_len=cfg.block_size,
        d_ff=cfg.d_ff,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        alpha=cfg.alpha,
        n_shared=cfg.n_shared,
        device=cfg.device,
    )
    model = MoETransformer(model_cfg)
    device = model.primary_device
    logger.info(
        f"model: {model.num_params():,} params on {device} "
        f"(N'={model_cfg.n_experts * model_cfg.alpha} experts/layer)"
    )

    get_batch = make_batch_fn(tokens, cfg, device)
    muon, adamw = build_optimizers(model, cfg)
    start_step = maybe_resume(cfg, model, muon, adamw)

    model.train()
    tokens_per_step = cfg.batch_size * cfg.block_size
    t0 = time.perf_counter()

    for step in range(start_step, cfg.steps):
        # Learning-rate schedule (shared multiplier for both optimizers).
        scale = lr_scale(step, cfg.warmup_steps, cfg.steps)
        for group in muon.param_groups:
            group["lr"] = cfg.muon_lr * scale
        for group in adamw.param_groups:
            group["lr"] = cfg.adam_lr * scale

        x, y = get_batch("train")
        _, loss = model(x, y)

        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        )
        muon.step()
        adamw.step()

        if step % cfg.log_every == 0:
            dt = time.perf_counter() - t0
            tok_s = (
                tokens_per_step * cfg.log_every / dt if step else 0.0
            )
            logger.info(
                f"step {step:>6d}/{cfg.steps} | loss {float(loss):.4f} | "
                f"aux {float(model.aux_loss):.4f} | "
                f"lr {cfg.muon_lr * scale:.2e} | {tok_s:,.0f} tok/s"
            )
            t0 = time.perf_counter()

        if step > 0 and step % cfg.eval_every == 0:
            val = estimate_val_loss(model, get_batch, cfg.eval_iters)
            logger.info(f"step {step:>6d} | val loss {val:.4f}")

        if step > 0 and step % cfg.save_every == 0:
            val = estimate_val_loss(model, get_batch, cfg.eval_iters)
            save_checkpoint(cfg, step, model, muon, adamw, val)

    # Final checkpoint.
    val = estimate_val_loss(model, get_batch, cfg.eval_iters)
    save_checkpoint(cfg, cfg.steps, model, muon, adamw, val)
    logger.success("training complete")


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__)
    for name, value in asdict(cfg).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            p.add_argument(flag, action="store_true", default=value)
        elif value is None:
            p.add_argument(flag, type=str, default=None)
        else:
            p.add_argument(flag, type=type(value), default=value)
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
