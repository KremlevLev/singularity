# Singularity Roadmap / Instruction

This file is the practical roadmap for [`Singularity 32B`](README.en.md:1). It exists to prevent jumping straight into 32B MoE + MLA + Ring Attention + GRPO. The correct path is sequential: tiny working model → train loop → checkpoint → data → sharding → MoE/MLA/DoRA → alignment → serving.

The main rule: **every new block must first pass on a tiny config**. Only after shape tests, forward, backward, and checkpointing should you increase model size.

---

## Target Architecture

The project targets an LLM-style decoder with:

- Llama 3.1 tokenizer/embedding adapter.
- SwiGLU FFN.
- Decoupled RoPE / YaRN.
- Multi-Head Latent Attention.
- Shared Experts + Routed Experts.
- Expert-Choice Routing.
- Router Z-Loss + Load Balancing Loss.
- DoRA adapters.
- Fake INT4 QAT.
- SFT → GRPO alignment.
- Orbax checkpointing.
- FastAPI serving.

The current implementation is a scaffold. Blocks should be implemented gradually.

---

## Stage 0. Environment Setup

### Check

```bash
python --version
python -m pip --version
python - <<'PY'
import jax
print(jax.default_backend())
print(len(jax.devices()))
PY
```

If `jax` is missing:

```bash
pip install -r requirements.txt
```

Minimal JAX stack:

```bash
pip install "jax[tpu]" flax optax orbax-checkpoint
```

### Expected state

- Kaggle TPU is visible through `jax.devices()`.
- Backend is `tpu`.
- Python >= 3.10.
- No large files are tracked in Git.
- `artifacts/`, `checkpoints/`, `logs/` are ignored by [`.gitignore`](.gitignore:1).

---

## Stage 1. Tiny Config

Create:

```text
configs/debug_small.yaml
```

Example:

```yaml
model:
  vocab_size: 1024
  hidden_size: 128
  intermediate_size: 256
  num_layers: 2
  num_heads: 4
  num_kv_heads: 2
  max_seq_len: 256
  rope_theta: 10000.0
  tie_embeddings: false

  mla:
    latent_kv_dim: 32
    q_lora_rank: 8
    kv_lora_rank: 32

  moe:
    num_experts: 4
    shared_experts: 1
    routed_experts: 3
    top_k: 1
    expert_capacity_factor: 1.25

  dora:
    enabled: false
    rank: 8

  quantization:
    enabled: false

training:
  phase: sft
  batch_size: 1
  seq_len: 128
  max_steps: 5
  gradient_accumulation_steps: 1
  dtype: float32
  remat: false
  optimizer:
    lr: 1.0e-3
    weight_decay: 0.0
    warmup_steps: 0
    max_grad_norm: 1.0

sharding:
  mesh_shape: [1]
  mesh_axes: ["data"]
```

### Why this matters

The 32B config is the target, but development must happen on a tiny model. If the tiny model cannot run forward/backward, the 32B model definitely will not.

---

## Stage 2. Synthetic Data and Tokenizer

Goal: produce a batch like:

```python
{
    "input_ids": jax.Array,        # [batch, seq_len]
    "labels": jax.Array,           # [batch, seq_len]
    "attention_mask": jax.Array,   # [batch, seq_len], bool
}
```

### Files

- [`singularity/data/tokenization.py`](singularity/data/tokenization.py:1)
- [`singularity/data/batch.py`](singularity/data/batch.py:1)
- [`singularity/data/datasets.py`](singularity/data/datasets.py:1)

### What to implement

1. Add `make_synthetic_batch(batch_size, seq_len, vocab_size, dtype)`.
2. Check dtypes:
   - `input_ids` / `labels`: `int32` or `int64`.
   - `attention_mask`: `bool`.
3. Check shapes:
   - `input_ids.shape == (batch_size, seq_len)`
   - `labels.shape == (batch_size, seq_len)`
4. Add `tests/test_batch.py`.

### Do not do yet

- Do not connect real datasets.
- Do not write a complex dataloader.
- Do not touch Polars until the first train step works.

---

## Stage 3. Working Forward Pass

Goal:

```python
import jax
import jax.numpy as jnp
from singularity.model import SingularityConfig, SingularityTransformer

config = SingularityConfig(
    vocab_size=1024,
    hidden_size=128,
    intermediate_size=256,
    num_layers: 2,
    num_heads: 4,
    num_kv_heads: 2,
    max_seq_len: 256,
    moe=None,
)

model = SingularityTransformer(config)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 128), dtype=jnp.int32))
logits = model.apply(params, jnp.ones((1, 128), dtype=jnp.int32))
assert logits.shape == (1, 128, 1024)
```

### Files

- [`singularity/model/config.py`](singularity/model/config.py:1)
- [`singularity/model/embeddings.py`](singularity/model/embeddings.py:1)
- [`singularity/model/rope.py`](singularity/model/rope.py:1)
- [`singularity/model/ffn.py`](singularity/model/ffn.py:1)
- [`singularity/model/attention.py`](singularity/model/attention.py:1)
- [`singularity/model/transformer.py`](singularity/model/transformer.py:1)

### Implementation order

1. Start with standard causal attention or even a placeholder attention returning `hidden_states`.
2. Add SwiGLU.
3. Add RoPE.
4. Add normalization.
5. Add `lm_head`.

### Do not start with

- MLA.
- MoE.
- DoRA.
- Ring Attention.

---

## Stage 4. Loss and Backward Pass

Goal: get scalar loss and gradients.

### Files

- [`singularity/training/losses.py`](singularity/training/losses.py:1)
- [`singularity/training/state.py`](singularity/training/state.py:1)
- [`singularity/training/train_sft.py`](singularity/training/train_sft.py:1)

### Minimal train step

```python
@jax.jit
def train_step(params, state, batch, key):
    def loss_fn(current_params):
        logits = model.apply(current_params, batch["input_ids"])
        loss = language_modeling_loss(logits, batch["labels"], batch.get("attention_mask"))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_state = state.replace(step=state.step + 1, opt_state=new_opt_state)
    return new_params, new_state, {"loss": loss}
```

### Checks

- Loss is finite.
- Gradients are finite.
- Parameter shapes are stable.
- `state.step` increases.
- First step works without OOM.

---

## Stage 5. JIT and Remat

Goal: wrap the train step in `jax.jit`.

### File

- [`singularity/training/train_sft.py`](singularity/training/train_sft.py:1)

### Add

```python
compiled_train_step = jax.jit(train_step)
```

Later, when the model becomes heavier:

```python
from jax import remat

remat_block = remat(TransformerBlock)
```

### Advice

- Do not add `remat` before ordinary backward works.
- Test tiny config first.
- `remat` can increase compile time but reduce HBM usage.

---

## Stage 6. Orbax Checkpointing

Goal: save and restore:

```python
{
    "step": int,
    "params": pytree,
    "opt_state": pytree,
    "config": dict,
}
```

### Files

- [`singularity/training/checkpoint.py`](singularity/training/checkpoint.py:1)
- [`scripts/export_checkpoint.py`](scripts/export_checkpoint.py:1)

### Minimal workflow

1. Save checkpoint every N steps.
2. On training start, search for the latest checkpoint.
3. If found, restore.
4. If not found, initialize from scratch.

### Important

Do not commit checkpoints to Git. Use:

```text
artifacts/checkpoints/
```

---

## Stage 7. Data Pipeline

Only after a working train step.

### Files

- [`singularity/data/preprocessing.py`](singularity/data/preprocessing.py:1)
- [`singularity/data/datasets.py`](singularity/data/datasets.py:1)
- [`singularity/data/tokenization.py`](singularity/data/tokenization.py:1)

### Plan

1. Take a small parquet file.
2. Process it with Polars.
3. Save tokenized shards.
4. Read shards batch-by-batch.
5. Feed them into the train loop.

### Recommended tokenized parquet format

Simple format:

```text
input_ids: list<int32>
labels: list<int32>
attention_mask: list<bool>
source: string
```

Faster flat format:

```text
input_ids_flat: int32
labels_flat: int32
attention_mask_flat: bool
```

---

## Stage 8. TPU Sharding

Goal: understand how the model maps to 8 TPU cores.

### Files

- [`singularity/training/sharding.py`](singularity/training/sharding.py:1)
- [`singularity/utils/tpu.py`](singularity/utils/tpu.py:1)

### Initial mesh

```yaml
sharding:
  mesh_shape: [1, 8, 1]
  mesh_axes: ["data", "tensor", "expert"]
```

But start simpler:

```yaml
sharding:
  mesh_shape: [8]
  mesh_axes: ["data"]
```

Then move to:

```yaml
sharding:
  mesh_shape: [1, 8, 1]
  mesh_axes: ["data", "tensor", "expert"]
```

### Check

```python
import jax
print(jax.devices())
```

### Advice

- Do not shard MoE immediately.
- Start with data parallel.
- Then tensor parallel.
- Then expert parallel.
- XLA prefers static shapes.

---

## Stage 9. MoE

Only after a normal transformer train loop works.

### Files

- [`singularity/model/moe.py`](singularity/model/moe.py:1)
- [`singularity/training/losses.py`](singularity/training/losses.py:1)

### Order

1. Token-choice routing.
2. Load balancing loss.
3. Router z-loss.
4. Shared experts.
5. Routed experts.
6. Expert capacity.
7. Expert-choice routing.

### Why this order

Expert-choice routing is harder because it requires grouping/padding/top-N tokens per expert. Starting there can easily cause dynamic shapes and XLA problems.

---

## Stage 10. MLA

Only after standard attention works.

### File

- [`singularity/model/attention.py`](singularity/model/attention.py:1)

### Order

1. Causal attention.
2. GQA/MQA.
3. Latent KV compression.
4. Decoupled RoPE.
5. Ring Attention.

### Important

Ring Attention is a separate large block. Do not mix it with the first MLA implementation.

---

## Stage 11. DoRA

Only after working full fine-tuning or at least a working LoRA-like adapter.

### Files

- [`singularity/model/dora.py`](singularity/model/dora.py:1)
- [`singularity/serving/merge.py`](singularity/serving/merge.py:1)

### Order

1. LoRA linear.
2. DoRA magnitude vector.
3. Freeze base weights.
4. Train adapters only.
5. Merge adapters.
6. Export merged checkpoint.

---

## Stage 12. Fake INT4 QAT

Only after stable BF16 training.

### File

- [`singularity/model/quantization.py`](singularity/model/quantization.py:1)

### Order

1. Fake-quantize weights before forward.
2. Verify gradients still flow.
3. Keep embeddings/router/MLA-sensitive parts in BF16.
4. Only then connect AQT/Qwix.

---

## Stage 13. GRPO Alignment

Only after an SFT checkpoint exists.

### Files

- [`singularity/alignment/sampler.py`](singularity/alignment/sampler.py:1)
- [`singularity/alignment/rewards.py`](singularity/alignment/rewards.py:1)
- [`singularity/alignment/grpo.py`](singularity/alignment/grpo.py:1)
- [`configs/grpo.yaml`](configs/grpo.yaml:1)

### Order

1. Sample 2 answers for a small prompt.
2. Compute reward.
3. Compute group advantage.
4. Update policy.
5. Add KL penalty.
6. Increase group size.
7. Connect code/math verifiers.

---

## Stage 14. Serving

Only after a restorable checkpoint exists.

### Files

- [`singularity/serving/api.py`](singularity/serving/api.py:1)
- [`singularity/serving/merge.py`](singularity/serving/merge.py:1)
- [`singularity/serving/search.py`](singularity/serving/search.py:1)
- [`configs/serving.yaml`](configs/serving.yaml:1)

### Order

1. Restore checkpoint.
2. Add `/generate`.
3. Add streaming response.
4. Add DoRA merge.
5. Add speculative decoding.
6. Add MCTS/PRM hooks.

---

## Practical Advice

### 1. Keep tiny config next to the big config

The big config is the dream. The tiny config is the working tool.

```text
configs/model.yaml        # 32B target
configs/debug_small.yaml  # working debug config
```

### 2. Start every new module with a shape test

Example:

```python
def test_module_output_shape():
    y = module(x)
    assert y.shape == expected
```

### 3. Avoid Python loops inside `jit`

Bad:

```python
for expert in experts:
    ...
```

Better:

```python
lax.scan(...)
vmap(...)
```

### 4. Do not increase sequence length too early

Use this order:

```text
128 → 512 → 2048 → 8192 → 16384 → 65536+
```

### 5. Loss first, speed later

Do not optimize speed before backward works.

### 6. Checkpoint often

Kaggle sessions can die. Minimum:

```yaml
training:
  checkpoint:
    every_steps: 500
    keep_last: 3
```

For long runs:

```yaml
training:
  checkpoint:
    every_steps: 1000
    keep_last: 5
    async_upload: true
```

### 7. Log more than loss

Minimum metrics:

```text
step
loss
grad_norm
learning_rate
tokens_per_second
memory_estimate_if_available
router_load_balance_if_moe
```

---

## Milestones

### Milestone 1 — Tiny Forward

- [ ] `SingularityTransformer` runs forward.
- [ ] `logits.shape == (batch, seq_len, vocab_size)`.
- [ ] No `NotImplementedError`.

### Milestone 2 — Tiny Train Step

- [ ] Synthetic batch.
- [ ] Language modeling loss.
- [ ] Optax state.
- [ ] One train step.
- [ ] Finite loss.

### Milestone 3 — JIT + Checkpoint

- [ ] `jax.jit(train_step)`.
- [ ] Orbax save.
- [ ] Orbax restore.
- [ ] Resume training.

### Milestone 4 — Data

- [ ] Synthetic parquet.
- [ ] Polars preprocessing.
- [ ] Tokenized batch.
- [ ] Real training loop on a small dataset.

### Milestone 5 — TPU Sharding

- [ ] `jax.devices()` sees 8 cores.
- [ ] Mesh is created.
- [ ] Data parallel works.
- [ ] Tensor parallel experiment.

### Milestone 6 — MoE

- [ ] Token-choice routing.
- [ ] Load balance loss.
- [ ] Router z-loss.
- [ ] Shared experts.
- [ ] Expert capacity.
- [ ] Expert-choice routing.

### Milestone 7 — MLA

- [ ] Latent KV.
- [ ] Decoupled RoPE.
- [ ] Long-context smoke test.
- [ ] Ring attention prototype.

### Milestone 8 — DoRA + QAT

- [ ] DoRA adapter train.
- [ ] Freeze base.
- [ ] Merge adapters.
- [ ] Fake INT4 forward.

### Milestone 9 — GRPO

- [ ] Sample group.
- [ ] Reward functions.
- [ ] Advantage.
- [ ] Policy update.
- [ ] KL penalty.

### Milestone 10 — Serving

- [ ] Restore checkpoint.
- [ ] `/generate`.
- [ ] Merge checkpoint.
- [ ] FastAPI deployment.
- [ ] Optional MCTS/PRM.

---

## What To Write First Right Now

1. [`configs/debug_small.yaml`](configs/debug_small.yaml:1)
2. `singularity/data/batch.py::make_synthetic_batch`
3. `tests/test_transformer_forward.py`
4. `singularity/model/attention.py` — simple causal attention
5. `singularity/training/train_sft.py::train_step`
6. `tests/test_train_step.py`
7. `singularity/training/checkpoint.py` — save/restore
8. Only then MoE/MLA/DoRA

---

## Red Flags

If you see these, stop and simplify:

- `CompilerError: dynamic shape`
- `RuntimeError: Ran out of memory`
- `NaN loss on first step`
- `XLA compilation > 20 minutes on tiny config`
- `MoE routing breaks shapes`
- `checkpoint restore does not match saved state`

In these cases, return to tiny config, disable MoE/MLA/DoRA/QAT, and fix the base loop.

---

## Summary

The correct strategy is:

```text
tiny dense transformer
→ tiny train loop
→ checkpoint
→ data
→ TPU sharding
→ MoE
→ MLA
→ DoRA
→ QAT
→ GRPO
→ serving
```

Not the other way around.
