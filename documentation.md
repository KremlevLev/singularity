✅ config/model_config.py — параметры модели (48 слоёв, 36 Mamba, 12 Attention)
✅ config/training_config.py — гиперпараметры обучения (LR, batch size, шаги)
✅ config/data_config.py — пути к датасетам, пропорции смешивания
✅ data/tokenizer.py — обёртка над токенизатором Qwen
✅ data/dataset.py — загрузчик датасетов (streaming из .jsonl)
✅ data/collator.py — батчирование и padding
✅ data/data_mixer.py — смешивание датасетов по пропорциям

