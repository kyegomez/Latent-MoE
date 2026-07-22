import torch
from latent_moe.transformer import TransformerConfig, MoETransformer

torch.manual_seed(0)

cfg = TransformerConfig(
    vocab_size=1000,
    d_model=256,
    n_layers=4,
    n_heads=8,
    n_kv_heads=2,  # GQA: 4 query heads share each KV head
    max_seq_len=128,
    d_ff=256,
    n_experts=8,
    top_k=2,
    alpha=2,
    n_shared=1,
)
model = MoETransformer(cfg)
print(f"params        : {model.num_params():,}")
print(f"primary device: {model.primary_device}")

idx = torch.randint(0, cfg.vocab_size, (2, 64))
targets = torch.randint(0, cfg.vocab_size, (2, 64))

logits, loss = model(idx, targets)
print(f"logits shape  : {tuple(logits.shape)}")
print(f"loss          : {loss.item():.4f}")
print(f"moe aux+z loss: {model.aux_loss.item():.4f}")

loss.backward()
print("backward OK")

out = model.generate(idx[:, :8], max_new_tokens=16, top_k=20)
print(f"generated     : {tuple(out.shape)}")
