# Singularity Roadmap / Instruction

Этот файл — практический roadmap разработки проекта [`Singularity 32B`](README.md:1). Он нужен, чтобы не прыгать сразу в 32B MoE + MLA + Ring Attention + GRPO, а идти последовательно: tiny working model → train loop → checkpoint → data → sharding → MoE/MLA/DoRA → alignment → serving.

Главное правило: **каждый новый блок сначала проходит на tiny config**. Только после shape-тестов, forward, backward и checkpoint можно увеличивать размер.

---

## Целевая архитектура

Проект ориентирован на модель с такими идеями:

- Base LLM-style decoder.
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

Но текущая реализация — scaffold. Реализовывать нужно постепенно.

---

## Этап 0. Зафиксировать окружение

### Что проверить

```bash
python --version
python -m pip --version
python - <<'PY'
import jax
print(jax.default_backend())
print(len(jax.devices()))
PY
```

Если `jax` не установлен:

```bash
pip install -r requirements.txt
```

Минимальный JAX-стек:

```bash
pip install "jax[tpu]" flax optax orbax-checkpoint
```

### Что должно быть

- Kaggle TPU виден через `jax.devices()`.
- Backend: `tpu`.
- Python >= 3.10.
- Нет больших файлов в Git.
- `artifacts/`, `checkpoints/`, `logs/` игнорируются через [`.gitignore`](.gitignore:1).

---

## Этап 1. Tiny config

Создай файл:

```text
configs/debug_small.yaml
```

Пример:

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

### Почему это важно

32B-конфиг нужен для документации, но разработка должна идти на маленькой модели. Если tiny-модель не проходит forward/backward, 32B точно не пройдет.

---

## Этап 2. Synthetic data и tokenizer

Цель: получить batch вида:

```python
{
    "input_ids": jax.Array,        # [batch, seq_len]
    "labels": jax.Array,           # [batch, seq_len]
    "attention_mask": jax.Array,   # [batch, seq_len], bool
}
```

### Файлы

- [`singularity/data/tokenization.py`](singularity/data/tokenization.py:1)
- [`singularity/data/batch.py`](singularity/data/batch.py:1)
- [`singularity/data/datasets.py`](singularity/data/datasets.py:1)

### Что сделать

1. Добавить `make_synthetic_batch(batch_size, seq_len, vocab_size, dtype)`.
2. Проверить dtype:
   - `input_ids` / `labels`: `int32` или `int64`.
   - `attention_mask`: `bool`.
3. Проверить shape:
   - `input_ids.shape == (batch_size, seq_len)`
   - `labels.shape == (batch_size, seq_len)`
4. Добавить тест `tests/test_batch.py`.

### Не делать пока

- Не подключать реальные датасеты.
- Не писать сложный dataloader.
- Не трогать Polars до появления первого train step.

---

## Этап 3. Рабочий forward маленькой модели

Цель: вызвать:

```python
import jax
import jax.numpy as jnp
from singularity.model import SingularityConfig, SingularityTransformer

config = SingularityConfig(
    vocab_size=1024,
    hidden_size=128,
    intermediate_size=256,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    max_seq_len=256,
    moe=None,
)

model = SingularityTransformer(config)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 128), dtype=jnp.int32))
logits = model.apply(params, jnp.ones((1, 128), dtype=jnp.int32))
assert logits.shape == (1, 128, 1024)
```

### Файлы

- [`singularity/model/config.py`](singularity/model/config.py:1)
- [`singularity/model/embeddings.py`](singularity/model/embeddings.py:1)
- [`singularity/model/rope.py`](singularity/model/rope.py:1)
- [`singularity/model/ffn.py`](singularity/model/ffn.py:1)
- [`singularity/model/attention.py`](singularity/model/attention.py:1)
- [`singularity/model/transformer.py`](singularity/model/transformer.py:1)

### Что делать первым

1. Сначала сделай обычный causal attention или даже placeholder attention, который возвращает `hidden_states`.
2. Потом подключи SwiGLU.
3. Потом RoPE.
4. Потом нормализации.
5. Потом lm_head.

### Что не делать

- Не начинай с MLA.
- Не начинай с MoE.
- Не начинай с DoRA.
- Не начинай с Ring Attention.

---

## Этап 4. Loss и backward

Цель: получить скалярный loss и градиенты.

### Файлы

- [`singularity/training/losses.py`](singularity/training/losses.py:1)
- [`singularity/training/state.py`](singularity/training/state.py:1)
- [`singularity/training/train_sft.py`](singularity/training/train_sft.py:1)

### Минимальный train step

Псевдокод:

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

### Проверки

- Loss finite.
- Grad finite.
- Params shape не ломаются.
- `state.step` увеличивается.
- Первый шаг работает без OOM.

---

## Этап 5. JIT и remat

Цель: обернуть train step в `jax.jit`.

### Файлы

- [`singularity/training/train_sft.py`](singularity/training/train_sft.py:1)

### Что добавить

```python
compiled_train_step = jax.jit(train_step)
```

Потом, когда модель станет тяжелее:

```python
from jax import remat

remat_block = remat(TransformerBlock)
```

### Советы

- Не добавляй `remat`, пока обычный backward не работает.
- Сначала проверяй tiny config.
- `remat` может увеличить compile time, но уменьшить HBM usage.

---

## Этап 6. Orbax checkpoint

Цель: сохранять и восстанавливать:

```python
{
    "step": int,
    "params": pytree,
    "opt_state": pytree,
    "config": dict,
}
```

### Файлы

- [`singularity/training/checkpoint.py`](singularity/training/checkpoint.py:1)
- [`scripts/export_checkpoint.py`](scripts/export_checkpoint.py:1)

### Минимальный сценарий

1. После каждого N steps сохранять checkpoint.
2. При старте training искать latest checkpoint.
3. Если найден — restore.
4. Если нет — init from scratch.

### Важно

Не сохраняй checkpoint в Git. Используй:

```text
artifacts/checkpoints/
```

---

## Этап 7. Данные

Только после рабочего train step.

### Файлы

- [`singularity/data/preprocessing.py`](singularity/data/preprocessing.py:1)
- [`singularity/data/datasets.py`](singularity/data/datasets.py:1)
- [`singularity/data/tokenization.py`](singularity/data/tokenization.py:1)

### План

1. Взять маленький parquet.
2. Прогнать через Polars.
3. Сохранить tokenized shards.
4. Читать shards batch-by-batch.
5. Подать в train loop.

### Формат tokenized parquet

Рекомендуемый формат:

```text
input_ids: list<int32>
labels: list<int32>
attention_mask: list<bool>
source: string
```

Или лучше для скорости:

```text
input_ids_flat: int32
labels_flat: int32
attention_mask_flat: bool
```

---

## Этап 8. Sharding на TPU

Цель: понять, как модель ложится на 8 TPU cores.

### Файлы

- [`singularity/training/sharding.py`](singularity/training/sharding.py:1)
- [`singularity/utils/tpu.py`](singularity/utils/tpu.py:1)

### Начальный mesh

```yaml
sharding:
  mesh_shape: [1, 8, 1]
  mesh_axes: ["data", "tensor", "expert"]
```

Но для начала лучше:

```yaml
sharding:
  mesh_shape: [8]
  mesh_axes: ["data"]
```

Потом:

```yaml
sharding:
  mesh_shape: [1, 8, 1]
  mesh_axes: ["data", "tensor", "expert"]
```

### Что проверить

```python
import jax
print(jax.devices())
```

### Советы

- Не пытайся сразу shard'ить MoE.
- Сначала data parallel.
- Потом tensor parallel.
- Потом expert parallel.
- XLA любит static shapes.

---

## Этап 9. MoE

Только после обычного transformer train loop.

### Файлы

- [`singularity/model/moe.py`](singularity/model/moe.py:1)
- [`singularity/training/losses.py`](singularity/training/losses.py:1)

### Очередность

1. Token-choice routing.
2. Load balancing loss.
3. Router z-loss.
4. Shared experts.
5. Routed experts.
6. Expert capacity.
7. Expert-choice routing.

### Почему так

Expert-choice routing сложнее, потому что требует grouping/padding/top-N токенов на эксперт. Если начать с него, легко получить dynamic shapes и XLA-проблемы.

---

## Этап 10. MLA

Только после обычного attention.

### Файлы

- [`singularity/model/attention.py`](singularity/model/attention.py:1)

### Очередность

1. Causal attention.
2. GQA/MQA.
3. Latent KV compression.
4. Decoupled RoPE.
5. Ring Attention.

### Важно

Ring Attention — это отдельный большой блок. Не смешивай его с первой MLA-реализацией.

---

## Этап 11. DoRA

Только после working full fine-tuning или хотя бы working LoRA-like adapter.

### Файлы

- [`singularity/model/dora.py`](singularity/model/dora.py:1)
- [`singularity/serving/merge.py`](singularity/serving/merge.py:1)

### Очередность

1. LoRA linear.
2. DoRA magnitude vector.
3. Freeze base weights.
4. Train adapters only.
5. Merge adapters.
6. Export merged checkpoint.

---

## Этап 12. Fake INT4 QAT

Только после stable BF16 training.

### Файлы

- [`singularity/model/quantization.py`](singularity/model/quantization.py:1)

### Очередность

1. Fake quantize weight перед forward.
2. Проверить, что gradients проходят.
3. Оставить embedding/router/MLA-sensitive parts в BF16.
4. Только потом подключать AQT/Qwix.

---

## Этап 13. GRPO alignment

Только после SFT checkpoint.

### Файлы

- [`singularity/alignment/sampler.py`](singularity/alignment/sampler.py:1)
- [`singularity/alignment/rewards.py`](singularity/alignment/rewards.py:1)
- [`singularity/alignment/grpo.py`](singularity/alignment/grpo.py:1)
- [`configs/grpo.yaml`](configs/grpo.yaml:1)

### Очередность

1. Sample 2 ответа на маленький prompt.
2. Посчитать reward.
3. Посчитать advantage внутри group.
4. Обновить policy.
5. Добавить KL penalty.
6. Увеличить group size.
7. Подключить code/math verifiers.

---

## Этап 14. Serving

Только после checkpoint, который можно restore.

### Файлы

- [`singularity/serving/api.py`](singularity/serving/api.py:1)
- [`singularity/serving/merge.py`](singularity/serving/merge.py:1)
- [`singularity/serving/search.py`](singularity/serving/search.py:1)
- [`configs/serving.yaml`](configs/serving.yaml:1)

### Очередность

1. Restore checkpoint.
2. Сделать `/generate`.
3. Добавить streaming response.
4. Добавить merge DoRA.
5. Добавить speculative decoding.
6. Добавить MCTS/PRM hooks.

---

## Практические советы

### 1. Держи tiny config рядом с большим

Большой конфиг — мечта. Tiny config — рабочий инструмент.

```text
configs/model.yaml        # 32B target
configs/debug_small.yaml  # рабочая отладка
```

### 2. Каждый новый модуль начинай с shape-теста

Пример:

```python
def test_module_output_shape():
    y = module(x)
    assert y.shape == expected
```

### 3. Не пиши Python-loop внутри `jit`

Плохо:

```python
for expert in experts:
    ...
```

Лучше:

```python
lax.scan(...)
vmap(...)
```

### 4. Не увеличивай seq_len слишком рано

Порядок:

```text
128 → 512 → 2048 → 8192 → 16384 → 65536+
```

### 5. Сначала loss, потом speed

Не оптимизируй скорость, пока нет working backward.

### 6. Checkpoint часто

На Kaggle сессии могут падать. Минимально:

```yaml
training:
  checkpoint:
    every_steps: 500
    keep_last: 3
```

Для долгих запусков:

```yaml
training:
  checkpoint:
    every_steps: 1000
    keep_last: 5
    async_upload: true
```

### 7. Логируй не только loss

Минимальные метрики:

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

## Минимальные milestone'ы

### Milestone 1 — Tiny forward

- [ ] `SingularityTransformer` проходит forward.
- [ ] `logits.shape == (batch, seq_len, vocab_size)`.
- [ ] Нет `NotImplementedError`.

### Milestone 2 — Tiny train step

- [ ] Synthetic batch.
- [ ] Language modeling loss.
- [ ] Optax state.
- [ ] One train step.
- [ ] Loss finite.

### Milestone 3 — JIT + checkpoint

- [ ] `jax.jit(train_step)`.
- [ ] Orbax save.
- [ ] Orbax restore.
- [ ] Resume training.

### Milestone 4 — Data

- [ ] Synthetic parquet.
- [ ] Polars preprocessing.
- [ ] Tokenized batch.
- [ ] Real training loop на маленьком датасете.

### Milestone 5 — TPU sharding

- [ ] `jax.devices()` видит 8 cores.
- [ ] Mesh создан.
- [ ] Data parallel работает.
- [ ] Tensor parallel эксперимент.

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
- [ ] Long context smoke test.
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

## Что писать первым прямо сейчас

1. [`configs/debug_small.yaml`](configs/debug_small.yaml:1)
2. `singularity/data/batch.py::make_synthetic_batch`
3. `tests/test_transformer_forward.py`
4. `singularity/model/attention.py` — простая causal attention
5. `singularity/training/train_sft.py::train_step`
6. `tests/test_train_step.py`
7. `singularity/training/checkpoint.py` — save/restore
8. Только потом MoE/MLA/DoRA

---

## Красные флаги

Если видишь это — остановись и упрощай:

- `CompilerError: dynamic shape`
- `RuntimeError: Ran out of memory`
- `NaN loss на первом шаге`
- `XLA compilation > 20 минут на tiny config`
- `MoE routing ломает shape`
- `checkpoint restore не совпадает с saved state`

В таких случаях возвращайся к tiny config, отключай MoE/MLA/DoRA/QAT и чини базовый loop.

---

## Итог

Правильная стратегия:

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

Не наоборот.
