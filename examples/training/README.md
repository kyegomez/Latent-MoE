# Training LatentMoE on Wikipedia

A minimal, self-contained training loop for the `MoETransformer` (Grouped-Query
Attention + LatentMoE) on a small slice of Wikipedia.

| Piece | What it uses |
| --- | --- |
| **Data** | A streamed subset of [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) via HuggingFace `datasets`, tokenized with the GPT-2 BPE tokenizer |
| **Model** | `latent_moe.transformer.MoETransformer` — GQA attention + LatentMoE FFNs |
| **Optimizer** | [`torch.optim.Muon`](https://pytorch.org/docs/stable/generated/torch.optim.Muon.html) for the 2D hidden weight matrices, `AdamW` for the embedding / 1D params (the recipe Muon expects) |
| **Logging** | [loguru](https://github.com/Delgan/loguru), to stderr **and** `<ckpt_dir>/train.log` |
| **Checkpoints** | Full training state saved every `--save-every` steps |

## Install

From the repository root, install the package and the training extras:

```bash
pip install -e .
pip install -r examples/training/requirements-train.txt
```

`requirements-train.txt` pulls in `torch>=2.9` (Muon landed in PyTorch 2.9),
`datasets`, `transformers`, and `loguru`.

## Quickstart

```bash
python examples/training/train.py
```

That trains the default ~30M-parameter model for 2000 steps on 2000 streamed
Wikipedia articles, auto-selecting CUDA → MPS → CPU. A shorter smoke run:

```bash
python examples/training/train.py \
  --steps 200 --num-articles 200 --batch-size 8 --block-size 256 \
  --save-every 50
```

## How it works

1. **Tokenize** – stream `--num-articles` articles, encode each with GPT-2, join
   with `<eos>` into one long token tensor, and hold out `--val-fraction` for
   validation.
2. **Batch** – sample random `--block-size` windows (`x`) and their next-token
   targets (`y`), nanoGPT-style.
3. **Step** – forward returns `(logits, loss)` where the loss already includes
   the MoE load-balancing `aux_loss` + `z_loss`. Backprop, clip grads, then step
   **both** optimizers (Muon + AdamW).
4. **Schedule** – a shared linear-warmup → cosine-decay multiplier scales both
   optimizers' learning rates.
5. **Checkpoint** – every `--save-every` steps, write `step_NNNNNN.pt` and
   `latest.pt` containing model + both optimizer states + config + val loss.

## Key options

Every field of `TrainConfig` is exposed as a `--flag` (underscores become
dashes). The most useful:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--steps` | `2000` | Training steps |
| `--batch-size` | `8` | Sequences per step |
| `--block-size` | `256` | Sequence length (also the model's `max_seq_len`) |
| `--num-articles` | `2000` | Wikipedia articles to stream |
| `--muon-lr` | `0.02` | Muon learning rate (2D hidden weights) |
| `--adam-lr` | `3e-4` | AdamW learning rate (embedding / 1D params) |
| `--save-every` | `100` | Checkpoint interval (steps) |
| `--eval-every` | `250` | Validation-loss logging interval |
| `--ckpt-dir` | `checkpoints` | Output directory for checkpoints + `train.log` |
| `--device` | auto | Force `cpu`, `cuda`, `cuda:0`, or `mps` |
| `--resume` | off | Resume from `<ckpt-dir>/latest.pt` |

Model-shape flags (`--d-model`, `--n-layers`, `--n-heads`, `--n-kv-heads`,
`--d-ff`, `--n-experts`, `--top-k`, `--alpha`, `--n-shared`) let you resize the
network. Constraints: `d_model` divisible by `n_heads`, `n_heads` divisible by
`n_kv_heads`, and `d_model / n_heads` even (RoPE).

## Resuming

```bash
python examples/training/train.py --ckpt-dir checkpoints --resume
```

Loads `latest.pt` and continues from the next step, restoring model weights and
both optimizer states.

## Notes

- **`wikimedia/wikipedia` is heavy** — even in streaming mode it fetches whole
  parquet shards, so the first batch can take a while to download. For a run
  that starts in seconds, point at a lighter Wikipedia-derived corpus:

  ```bash
  python examples/training/train.py \
    --dataset wikitext --dataset-config wikitext-103-raw-v1
  ```

- **Optimizer split** — the token embedding is 2D but is deliberately given to
  AdamW, not Muon; Muon orthogonalizes hidden weight matrices only. The script
  logs the exact tensor / parameter counts in each group at startup.

- **Checkpoints accumulate** — one `step_NNNNNN.pt` is written per save interval
  plus a rolling `latest.pt`. Prune old ones yourself if disk is a concern.
