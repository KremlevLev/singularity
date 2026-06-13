# Singularity 32B

**Singularity 32B** is a research scaffold for a JAX / Flax / Optax / Orbax LLM project targeting Kaggle TPU v5e-8 and later GPU inference. The project is designed as a base for a 32B-parameter MoE transformer with MLA, Decoupled RoPE, Shared/Routed Experts, DoRA adapters, fake INT4 QAT, SFT, GRPO alignment, checkpointing, and serving API.

The current repository is an **engineering scaffold**: directory structure, configs, imports, utility modules, skeleton modules, tests, and a development roadmap. Most heavy blocks are intentionally left as scaffolds so that the project can first establish a stable pipeline, and only then replace `NotImplementedError` placeholders with real JAX implementations.

---

## Quick Start

```bash
cd .lightning_studio/singularity

pip install -r requirements.txt

python main.py \
  --phase sft \
  --config configs/base.yaml \
  --config configs/model.yaml \
  --config configs/sft.yaml
```

Alternative script entrypoints:

```bash
python scripts/run_sft.py \
  --config configs/base.yaml \
  --config configs/model.yaml \
  --config configs/sft.yaml

python scripts/run_grpo.py \
  --config configs/base.yaml \
  --config configs/model.yaml \
  --config configs/grpo.yaml
```

Serving:

```bash
python main.py \
  --phase serve \
  --config configs/base.yaml \
  --config configs/serving.yaml
```

> Important: `jax` may not be installed in the current environment. If JAX imports fail with `ModuleNotFoundError: No module named 'jax'`, install dependencies from [`requirements.txt`](requirements.txt:1) first.

---

## Architecture Overview

The project is split into layers:

1. [`configs/`](configs/base.yaml:1) — YAML configs merged into the final runtime config.
2. [`singularity/utils/`](singularity/utils/__init__.py:1) — config loading, logging, seeding, TPU bootstrap, path helpers.
3. [`singularity/data/`](singularity/data/__init__.py:1) — Polars preprocessing, tokenization, batching.
4. [`singularity/model/`](singularity/model/__init__.py:1) — Flax modules: embeddings, RoPE, MLA attention, SwiGLU, MoE, DoRA, quantization, transformer.
5. [`singularity/training/`](singularity/training/__init__.py:1) — sharding, losses, state, checkpointing, SFT training loop.
6. [`singularity/alignment/`](singularity/alignment/__init__.py:1) — sampler, rewards, GRPO update.
7. [`singularity/serving/`](singularity/serving/__init__.py:1) — DoRA merge, search/MCTS, FastAPI.
8. [`scripts/`](scripts/run_sft.py:1) — convenience CLI entrypoints.
9. [`tests/`](tests/test_rope.py:1) — shape tests and smoke tests.

---

## Repository Map

### Root files

| File | Purpose |
|---|---|
| [`main.py`](main.py:1) | Thin wrapper that calls [`singularity.main`](singularity/main.py:1). |
| [`pyproject.toml`](pyproject.toml:1) | Project metadata, dependencies, ruff, pytest config. |
| [`requirements.txt`](requirements.txt:1) | Fast install file for Kaggle. |
| [`.gitignore`](.gitignore:1) | Ignores `artifacts/`, `checkpoints/`, `__pycache__/`, logs, envs. |
| [`README.md`](README.md:1) | Russian documentation. |
| [`README.en.md`](README.en.md:1) | English documentation. |
| [`instruction.md`](instruction.md:1) | Russian roadmap. |
| [`instruction.en.md`](instruction.en.md:1) | English roadmap. |

### Configs

| File | Purpose |
|---|---|
| [`configs/base.yaml`](configs/base.yaml:1) | Project name, seed, precision, paths. |
| [`configs/model.yaml`](configs/model.yaml:1) | Vocab size, hidden size, layers, heads, MLA, MoE, DoRA, quantization. |
| [`configs/sft.yaml`](configs/sft.yaml:1) | SFT batch size, sequence length, max steps, optimizer, checkpointing, sharding mesh. |
| [`configs/grpo.yaml`](configs/grpo.yaml:1) | Group size, max new tokens, temperature, KL coefficient, checkpointing. |
| [`configs/serving.yaml`](configs/serving.yaml:1) | FastAPI host/port, checkpoint dir, max sequence length. |

### Package `singularity`

| Module | Purpose |
|---|---|
| [`singularity/main.py`](singularity/main.py:1) | CLI runner: selects `sft`, `grpo`, or `serve`, initializes TPU, seed, and logger. |
| [`singularity/utils/config.py`](singularity/utils/config.py:1) | YAML loading and deep merge. |
| [`singularity/utils/logging.py`](singularity/utils/logging.py:1) | Rich logger with standard logging fallback. |
| [`singularity/utils/seed.py`](singularity/utils/seed.py:1) | Deterministic seeding for Python, NumPy, and JAX. |
| [`singularity/utils/tpu.py`](singularity/utils/tpu.py:1) | Kaggle/Colab TPU bootstrap via `jax.tools.colab_tpu`. |
| [`singularity/utils/paths.py`](singularity/utils/paths.py:1) | Project root and path resolver. |

### Data layer

| File | Purpose |
|---|---|
| [`singularity/data/tokenization.py`](singularity/data/tokenization.py:1) | `TokenizerAdapter` based on `tiktoken`. |
| [`singularity/data/preprocessing.py`](singularity/data/preprocessing.py:1) | Polars `scan_parquet`, mixing, shuffling, `sink_parquet`. |
| [`singularity/data/datasets.py`](singularity/data/datasets.py:1) | `load_parquet_dataset`, NumPy batch iterator. |
| [`singularity/data/batch.py`](singularity/data/batch.py:1) | Dict-to-JAX-array conversion, mask dtype handling, batch splitting. |

### Model layer

| File | Purpose |
|---|---|
| [`singularity/model/config.py`](singularity/model/config.py:1) | Dataclasses: `SingularityConfig`, `MLAConfig`, `MoEConfig`, `QuantizationConfig`. |
| [`singularity/model/embeddings.py`](singularity/model/embeddings.py:1) | `LlamaEmbeddingAdapter` using `nn.Embed`. |
| [`singularity/model/rope.py`](singularity/model/rope.py:1) | Scaffold for Decoupled RoPE. |
| [`singularity/model/attention.py`](singularity/model/attention.py:1) | Scaffold for Multi-Head Latent Attention. |
| [`singularity/model/ffn.py`](singularity/model/ffn.py:1) | SwiGLU block. |
| [`singularity/model/moe.py`](singularity/model/moe.py:1) | ExpertChoiceRouter and SharedRoutedMoE scaffold. |
| [`singularity/model/dora.py`](singularity/model/dora.py:1) | DoRA linear adapter. |
| [`singularity/model/quantization.py`](singularity/model/quantization.py:1) | Fake INT4 weight and optional AQT/Qwix hooks. |
| [`singularity/model/transformer.py`](singularity/model/transformer.py:1) | Full `SingularityTransformer` assembly. |

### Training layer

| File | Purpose |
|---|---|
| [`singularity/training/sharding.py`](singularity/training/sharding.py:1) | `Mesh`, `NamedSharding`, `PartitionSpec`. |
| [`singularity/training/state.py`](singularity/training/state.py:1) | `TrainState` with `optax`. |
| [`singularity/training/losses.py`](singularity/training/losses.py:1) | Language modeling loss, router z-loss, load balancing loss. |
| [`singularity/training/checkpoint.py`](singularity/training/checkpoint.py:1) | Orbax checkpoint manager. |
| [`singularity/training/train_sft.py`](singularity/training/train_sft.py:1) | SFT runner scaffold, optimizer builder, placeholder train step. |

### Alignment layer

| File | Purpose |
|---|---|
| [`singularity/alignment/sampler.py`](singularity/alignment/sampler.py:1) | Autoregressive token sampler scaffold. |
| [`singularity/alignment/rewards.py`](singularity/alignment/rewards.py:1) | Math answer verifier and Python compile verifier. |
| [`singularity/alignment/grpo.py`](singularity/alignment/grpo.py:1) | GRPO runner scaffold. |

### Serving layer

| File | Purpose |
|---|---|
| [`singularity/serving/api.py`](singularity/serving/api.py:1) | FastAPI app and `/generate` endpoint. |
| [`singularity/serving/merge.py`](singularity/serving/merge.py:1) | DoRA matrix absorption helper. |
| [`singularity/serving/search.py`](singularity/serving/search.py:1) | MCTS node scaffold and branch selection. |

### Scripts

| Script | Purpose |
|---|---|
| [`scripts/run_sft.py`](scripts/run_sft.py:1) | Run SFT with config forwarding. |
| [`scripts/run_grpo.py`](scripts/run_grpo.py:1) | Run GRPO with config forwarding. |
| [`scripts/export_checkpoint.py`](scripts/export_checkpoint.py:1) | Upload checkpoint folder to Hugging Face Hub. |
| [`scripts/merge_dora.py`](scripts/merge_dora.py:1) | Placeholder for DoRA checkpoint merge. |

### Tests

| Test | Purpose |
|---|---|
| [`tests/test_rope.py`](tests/test_rope.py:1) | RoPE scaffold shape check. |
| [`tests/test_moe_routing.py`](tests/test_moe_routing.py:1) | Router logits/probs shape check. |
| [`tests/test_loss.py`](tests/test_loss.py:1) | Scalar loss shape check. |

---

## Current Implementation Status

### Done

- Full project structure.
- YAML configs for base/model/SFT/GRPO/serving.
- JAX/Flax-oriented imports.
- Kaggle TPU bootstrap.
- Config loader with deep merge.
- Logging, seeding, paths.
- Tokenizer/data/batch scaffolds.
- Flax module scaffolds.
- Loss functions.
- Orbax checkpoint manager.
- FastAPI serving scaffold.
- Test placeholders.

### Still scaffolded

- Real MLA forward.
- Full Decoupled RoPE/YaRN.
- Expert-choice routing with padding/expert capacity.
- Ring Attention.
- `jax.remat` integration.
- `jax.lax.scan` over layers.
- JIT-compiled SFT train step.
- Real parquet/tokenized data loader.
- Full GRPO update.
- Real PRM/MCTS integration.
- Production checkpoint matrix absorption.

---

## CLI Usage

Main runner: [`singularity/main.py`](singularity/main.py:1).

SFT:

```bash
python main.py \
  --phase sft \
  --config configs/base.yaml \
  --config configs/model.yaml \
  --config configs/sft.yaml
```

GRPO:

```bash
python main.py \
  --phase grpo \
  --config configs/base.yaml \
  --config configs/model.yaml \
  --config configs/grpo.yaml
```

Serving:

```bash
python main.py \
  --phase serve \
  --config configs/base.yaml \
  --config configs/serving.yaml
```

Help:

```bash
python main.py --help
```

Disable TPU bootstrap for local CPU/GPU debugging:

```bash
python main.py --no-tpu --phase serve --config configs/base.yaml --config configs/serving.yaml
```

---

## Config Loading

Configs are merged through [`singularity/utils/config.py`](singularity/utils/config.py:1):

```python
from singularity.utils.config import load_config

config = load_config(
    "configs/base.yaml",
    "configs/model.yaml",
    "configs/sft.yaml",
)
```

Order matters: later configs override or deep-merge earlier configs.

Example future debug config:

```bash
python main.py \
  --config configs/base.yaml \
  --config configs/model.yaml \
  --config configs/sft.yaml \
  --config configs/debug_small.yaml
```

Suggested `configs/debug_small.yaml`:

```yaml
model:
  vocab_size: 1024
  hidden_size: 256
  intermediate_size: 512
  num_layers: 2
  num_heads: 4
  num_kv_heads: 2
  max_seq_len: 512

training:
  batch_size: 1
  seq_len: 512
  max_steps: 10
```

---

## Recommended Development Order

Do **not** start with 32B. First make a tiny model that can run forward, loss, backward, checkpointing, and one train step on CPU/TPU.

Correct order:

1. Tiny config.
2. Tokenizer and synthetic batch.
3. Embedding + SwiGLU + RoPE.
4. Standard causal attention.
5. One transformer block.
6. Full transformer forward.
7. Language modeling loss.
8. Optax state.
9. JIT train step.
10. Orbax checkpoint.
11. Polars data loader.
12. Sharding mesh.
13. Remat and `lax.scan`.
14. MoE routing.
15. MLA.
16. DoRA.
17. GRPO.
18. Serving.

Avoid implementing MLA + MoE + DoRA + Ring Attention in one giant step.

---

## Kaggle TPU Checklist

Check devices:

```python
import jax
print(jax.default_backend())
print(len(jax.devices()))
print(jax.devices())
```

TPU bootstrap is implemented in [`singularity/utils/tpu.py`](singularity/utils/tpu.py:1).

TPU tips:

- Keep large Python lists out of `jit`.
- Avoid dynamic shapes when XLA requires static shapes.
- For MoE, plan expert capacity and padding from the beginning.
- For long context, test gradually: 512 → 2k → 8k → 16k.
- Prefer `bfloat16` on TPU.
- Keep `jax_enable_x64(False)` by default to avoid unnecessary memory usage.

---

## Testing

Syntax check:

```bash
python - <<'PY'
from pathlib import Path
import ast

for path in Path('.').rglob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
PY
```

Pytest:

```bash
pip install -r requirements.txt
pytest -q
```

If JAX is not installed, JAX tests will not run. After installing dependencies, start with:

```bash
pytest tests/test_loss.py -q
pytest tests/test_rope.py -q
pytest tests/test_moe_routing.py -q
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'jax'`

Install dependencies:

```bash
pip install -r requirements.txt
```

Or minimally:

```bash
pip install "jax[tpu]" flax optax orbax-checkpoint
```

### `No module named 'singularity'`

Run from the project root:

```bash
cd .lightning_studio/singularity
python main.py --help
```

Or set:

```bash
export PYTHONPATH=.
```

### `NotImplementedError`

This is expected for scaffolded blocks. Find them with:

```bash
grep -R "NotImplementedError" -n .
```

### TPU OOM

Reduce:

```yaml
training:
  batch_size: 1
  seq_len: 512

model:
  hidden_size: 256
  num_layers: 2
  num_heads: 4
  intermediate_size: 512
```

Then add remat, gradient accumulation, and sharding.

---

## What To Implement Next

1. Create [`configs/debug_small.yaml`](configs/debug_small.yaml:1).
2. Implement synthetic batching in [`singularity/data/batch.py`](singularity/data/batch.py:1).
3. Add `tests/test_transformer_forward.py`.
4. Implement simple causal attention in [`singularity/model/attention.py`](singularity/model/attention.py:1).
5. Implement real [`train_step`](singularity/training/train_sft.py:1) with `jax.jit` and `jax.value_and_grad`.
6. Add `tests/test_train_step.py`.
7. Add Orbax save/load for `TrainState`.
8. Only then move to MoE, MLA, and DoRA.

---

## Final Strategy

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

Do not reverse this order.
