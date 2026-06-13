# Singularity 32B

**Singularity 32B** — это scaffold для исследовательского LLM-проекта на **JAX / Flax / Optax / Orbax** под Kaggle TPU v5e-8 и последующий инференс на GPU. Проект задуман как база для 32B MoE-модели с MLA, Decoupled RoPE, Shared/Routed Experts, DoRA-адаптерами, fake INT4 QAT, SFT, GRPO alignment, checkpointing и serving API.

Текущий репозиторий — это **инженерный каркас**: структура, конфиги, импорты, базовые utility-модули, skeleton-модули и roadmap. Большинство тяжелых блоков пока намеренно оставлены как scaffold, чтобы сначала стабильно собрать pipeline, а потом постепенно заменять `NotImplementedError` на реальные JAX-реализации.

## Быстрый старт

```bash
cd .lightning_studio/singularity

pip install -r requirements.txt

python main.py \
  --phase sft \
  --config configs/base.yaml \
  --config configs/model.yaml \
  --config configs/sft.yaml
```

Альтернативно через скрипты:

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

Для serving:

```bash
python main.py \
  --phase serve \
  --config configs/base.yaml \
  --config configs/serving.yaml
```

> Важно: в текущем окружении `jax` может быть не установлен. Если импорт JAX-модулей падает с `ModuleNotFoundError: No module named 'jax'`, сначала установи зависимости из [`requirements.txt`](requirements.txt:1).

---

## Главная идея архитектуры

Проект разделен на слои:

1. [`configs/`](configs/base.yaml:1) — YAML-конфиги, из которых собирается финальный dictionary.
2. [`singularity/utils/`](singularity/utils/__init__.py:1) — загрузка конфигов, logging, seed, TPU bootstrap, paths.
3. [`singularity/data/`](singularity/data/__init__.py:1) — Polars preprocessing, tokenization, batching.
4. [`singularity/model/`](singularity/model/__init__.py:1) — Flax-модули: embeddings, RoPE, MLA attention, SwiGLU, MoE, DoRA, quantization, transformer.
5. [`singularity/training/`](singularity/training/__init__.py:1) — sharding, losses, state, checkpointing, SFT train loop.
6. [`singularity/alignment/`](singularity/alignment/__init__.py:1) — sampler, rewards, GRPO update.
7. [`singularity/serving/`](singularity/serving/__init__.py:1) — DoRA merge, MCTS/search, FastAPI.
8. [`scripts/`](scripts/run_sft.py:1) — удобные CLI entrypoints.
9. [`tests/`](tests/test_rope.py:1) — shape-тесты и smoke-тесты.

---

## Что уже создано

### Корневые файлы

| Файл | Назначение |
|---|---|
| [`main.py`](main.py:1) | Тонкий wrapper, который вызывает [`singularity.main`](singularity/main.py:1). |
| [`pyproject.toml`](pyproject.toml:1) | project metadata, dependencies, ruff, pytest config. |
| [`requirements.txt`](requirements.txt:1) | быстрый install-файл для Kaggle. |
| [`.gitignore`](.gitignore:1) | игнорирует `artifacts/`, `checkpoints/`, `__pycache__/`, logs, envs. |
| [`README.md`](README.md:1) | подробная документация проекта. |
| [`instruction.md`](instruction.md:1) | roadmap и порядок разработки. |

### Конфиги

| Файл | Что внутри |
|---|---|
| [`configs/base.yaml`](configs/base.yaml:1) | имя проекта, seed, precision, paths. |
| [`configs/model.yaml`](configs/model.yaml:1) | vocab size, hidden size, layers, heads, MLA, MoE, DoRA, quantization. |
| [`configs/sft.yaml`](configs/sft.yaml:1) | SFT batch size, seq len, max steps, optimizer, checkpointing, sharding mesh. |
| [`configs/grpo.yaml`](configs/grpo.yaml:1) | group size, max tokens, temperature, KL coef, checkpointing. |
| [`configs/serving.yaml`](configs/serving.yaml:1) | FastAPI host/port, checkpoint dir, max seq len. |

### Package `singularity`

| Модуль | Назначение |
|---|---|
| [`singularity/main.py`](singularity/main.py:1) | CLI runner: выбирает phase `sft`, `grpo` или `serve`, поднимает TPU, seed, logger. |
| [`singularity/utils/config.py`](singularity/utils/config.py:1) | YAML loading и deep merge конфигов. |
| [`singularity/utils/logging.py`](singularity/utils/logging.py:1) | Rich logger или fallback logging. |
| [`singularity/utils/seed.py`](singularity/utils/seed.py:1) | deterministic seed для Python, NumPy, JAX. |
| [`singularity/utils/tpu.py`](singularity/utils/tpu.py:1) | Kaggle/Colab TPU bootstrap через `jax.tools.colab_tpu`. |
| [`singularity/utils/paths.py`](singularity/utils/paths.py:1) | project root и path resolver. |

### Data layer

| Файл | Назначение |
|---|---|
| [`singularity/data/tokenization.py`](singularity/data/tokenization.py:1) | `TokenizerAdapter` на `tiktoken`. |
| [`singularity/data/preprocessing.py`](singularity/data/preprocessing.py:1) | Polars `scan_parquet`, mix, shuffle, `sink_parquet`. |
| [`singularity/data/datasets.py`](singularity/data/datasets.py:1) | `load_parquet_dataset`, numpy batch iterator. |
| [`singularity/data/batch.py`](singularity/data/batch.py:1) | conversion dict → JAX arrays, mask dtype handling, batch split. |

### Model layer

| Файл | Назначение |
|---|---|
| [`singularity/model/config.py`](singularity/model/config.py:1) | dataclasses: `SingularityConfig`, `MLAConfig`, `MoEConfig`, `QuantizationConfig`. |
| [`singularity/model/embeddings.py`](singularity/model/embeddings.py:1) | `LlamaEmbeddingAdapter` на `nn.Embed`. |
| [`singularity/model/rope.py`](singularity/model/rope.py:1) | scaffold для Decoupled RoPE. |
| [`singularity/model/attention.py`](singularity/model/attention.py:1) | scaffold для Multi-Head Latent Attention. |
| [`singularity/model/ffn.py`](singularity/model/ffn.py:1) | SwiGLU block. |
| [`singularity/model/moe.py`](singularity/model/moe.py:1) | ExpertChoiceRouter и SharedRoutedMoE scaffold. |
| [`singularity/model/dora.py`](singularity/model/dora.py:1) | DoRA linear adapter. |
| [`singularity/model/quantization.py`](singularity/model/quantization.py:1) | fake INT4 weight и optional AQT/Qwix hooks. |
| [`singularity/model/transformer.py`](singularity/model/transformer.py:1) | сборка полного `SingularityTransformer`. |

### Training layer

| Файл | Назначение |
|---|---|
| [`singularity/training/sharding.py`](singularity/training/sharding.py:1) | `Mesh`, `NamedSharding`, `PartitionSpec`. |
| [`singularity/training/state.py`](singularity/training/state.py:1) | `TrainState` с `optax`. |
| [`singularity/training/losses.py`](singularity/training/losses.py:1) | language modeling loss, router z-loss, load balancing loss. |
| [`singularity/training/checkpoint.py`](singularity/training/checkpoint.py:1) | Orbax checkpoint manager. |
| [`singularity/training/train_sft.py`](singularity/training/train_sft.py:1) | SFT runner scaffold, optimizer builder, placeholder train step. |

### Alignment layer

| Файл | Назначение |
|---|---|
| [`singularity/alignment/sampler.py`](singularity/alignment/sampler.py:1) | autoregressive token sampler scaffold. |
| [`singularity/alignment/rewards.py`](singularity/alignment/rewards.py:1) | math answer verifier и Python compile verifier. |
| [`singularity/alignment/grpo.py`](singularity/alignment/grpo.py:1) | GRPO runner scaffold. |

### Serving layer

| Файл | Назначение |
|---|---|
| [`singularity/serving/api.py`](singularity/serving/api.py:1) | FastAPI app и `/generate` endpoint. |
| [`singularity/serving/merge.py`](singularity/serving/merge.py:1) | DoRA matrix absorption helper. |
| [`singularity/serving/search.py`](singularity/serving/search.py:1) | MCTS node scaffold и branch selection. |

### Scripts

| Скрипт | Назначение |
|---|---|
| [`scripts/run_sft.py`](scripts/run_sft.py:1) | запуск SFT с передачей конфигов. |
| [`scripts/run_grpo.py`](scripts/run_grpo.py:1) | запуск GRPO с передачей конфигов. |
| [`scripts/export_checkpoint.py`](scripts/export_checkpoint.py:1) | upload checkpoint folder в Hugging Face Hub. |
| [`scripts/merge_dora.py`](scripts/merge_dora.py:1) | placeholder для merge DoRA checkpoint. |

### Tests

| Тест | Что проверяет |
|---|---|
| [`tests/test_rope.py`](tests/test_rope.py:1) | shape для RoPE scaffold. |
| [`tests/test_moe_routing.py`](tests/test_moe_routing.py:1) | shape router logits/probs. |
| [`tests/test_loss.py`](tests/test_loss.py:1) | scalar loss shape. |

---

## Текущий статус реализации

### Готово

- Полная структура проекта.
- YAML-конфиги для base/model/SFT/GRPO/serving.
- JAX/Flax-oriented imports.
- Kaggle TPU bootstrap.
- Config loader с deep merge.
- Logging, seed, paths.
- Tokenizer/data/batch scaffolds.
- Flax module scaffolds.
- Loss functions.
- Orbax checkpoint manager.
- FastAPI serving scaffold.
- Tests placeholders.

### Пока scaffold

- Настоящий MLA forward.
- Полноценный Decoupled RoPE/YaRN.
- Expert-Choice routing с padding/expert capacity.
- Ring Attention.
- `jax.remat` integration.
- `jax.lax.scan` over layers.
- JIT-compiled SFT train step.
- Real data loader для parquet/tokenized shards.
- Full GRPO update.
- Real PRM/MCTS integration.
- Matrix absorption для production checkpoint.

---

## Как устроен CLI

Главный runner — [`singularity/main.py`](singularity/main.py:1).

Основные аргументы:

```bash
python main.py --phase sft --config configs/base.yaml --config configs/model.yaml --config configs/sft.yaml
python main.py --phase grpo --config configs/base.yaml --config configs/model.yaml --config configs/grpo.yaml
python main.py --phase serve --config configs/base.yaml --config configs/serving.yaml
```

Также можно:

```bash
python main.py --help
python main.py --no-tpu --phase serve --config configs/base.yaml --config configs/serving.yaml
```

`--no-tpu` нужен для локальной отладки на CPU/GPU, если Kaggle TPU bootstrap не нужен.

---

## Как читать конфиги

Конфиги собираются через [`singularity/utils/config.py`](singularity/utils/config.py:1):

```python
from singularity.utils.config import load_config

config = load_config(
    "configs/base.yaml",
    "configs/model.yaml",
    "configs/sft.yaml",
)
```

Порядок важен: поздние конфиги перезаписывают или deep-merge-ят ранние. Например:

```bash
python main.py \
  --config configs/base.yaml \
  --config configs/model.yaml \
  --config configs/sft.yaml \
  --config configs/debug_small.yaml
```

В будущем удобно добавить `configs/debug_small.yaml`, где уменьшить:

```yaml
model:
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

## Рекомендуемый порядок разработки

Самое важное: **не начинай сразу с 32B**. Сначала сделай tiny-модель, которая проходит forward, loss, backward, checkpoint и маленький train step на CPU/TPU.

Правильная последовательность:

1. Tiny config.
2. Tokenizer + synthetic batch.
3. Embedding + SwiGLU + RoPE.
4. Обычная attention.
5. Один transformer block.
6. Full transformer forward.
7. Language modeling loss.
8. Optax state.
9. JIT train step.
10. Orbax checkpoint.
11. Polars dataloader.
12. Sharding mesh.
13. Remat/scan.
14. MoE routing.
15. MLA.
16. DoRA.
17. GRPO.
18. Serving.

Не пытайся сразу писать MLA + MoE + DoRA + Ring Attention в одном блоке. Сначала собери работающий training loop на маленькой Dense/GQA-модели.

---

## TPU checklist для Kaggle

- Проверь устройства:

```python
import jax
print(jax.default_backend())
print(len(jax.devices()))
print(jax.devices())
```

- Для Kaggle TPU bootstrap используется [`singularity/utils/tpu.py`](singularity/utils/tpu.py:1).
- Не держи большие Python-списки внутри `jit`.
- Не делай dynamic shapes там, где XLA требует static shapes.
- Для MoE заранее думай про expert capacity и padding.
- Для long context сначала проверяй 512 → 2k → 8k → 16k.
- `bfloat16` предпочтителен на TPU.
- `jax.enable_x64(False)` оставлен по умолчанию, чтобы не раздувать память.

---

## Тестирование

Синтаксис Python-файлов:

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

Пока JAX не установлен, JAX-тесты не запустятся. После установки зависимостей начни с:

```bash
pytest tests/test_loss.py -q
pytest tests/test_rope.py -q
pytest tests/test_moe_routing.py -q
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'jax'`

Установи зависимости:

```bash
pip install -r requirements.txt
```

Или минимально:

```bash
pip install "jax[tpu]" flax optax orbax-checkpoint
```

### `No module named 'singularity'`

Запускай из корня проекта:

```bash
cd .lightning_studio/singularity
python main.py --help
```

Или добавь root в `PYTHONPATH`:

```bash
export PYTHONPATH=.
```

### `NotImplementedError`

Это нормально для scaffold-блоков. Ищи места:

```bash
grep -R "NotImplementedError" -n .
```

### OOM на TPU

Сначала уменьши:

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

Потом включай remat, gradient accumulation и sharding.

---

## Что делать следующим шагом

1. Создай [`configs/debug_small.yaml`](configs/debug_small.yaml:1).
2. Реализуй synthetic batch в [`singularity/data/batch.py`](singularity/data/batch.py:1).
3. Доведи [`singularity/model/transformer.py`](singularity/model/transformer.py:1) до working forward на tiny config.
4. Реализуй настоящий [`train_step`](singularity/training/train_sft.py:1) с `jax.jit` и `jax.value_and_grad`.
5. Добавь Orbax save/load для `TrainState`.
6. Только после этого переходи к MoE, MLA и DoRA.
