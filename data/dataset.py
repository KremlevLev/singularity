from typing import List, Dict, Optional
import json
import random
from torch.utils.data import Dataset, IterableDataset
import torch.utils
from config import DataConfig
from .tokenizer import QwenTokenizer


class PretrainDataset(IterableDataset):
    """
    Итеративный датасет для continue pre-training.
    Поддерживает streaming из .jsonl файлов.
    """

    def __init__(
        self,
        data_paths: List[str],
        tokenizer: QwenTokenizer,
        config: DataConfig,
    ):
        self.data_paths = data_paths
        self.tokenizer = tokenizer
        self.config = config
        self.text_key = config.text_key

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        paths = self.data_paths

        # Если используется несколько воркеров, разбиваем файлы между ними
        if worker_info is not None:
            per_worker = len(paths) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker
            paths = paths[start:end]

        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        text = data.get(self.text_key, "")
                        if len(text) < self.config.min_length:
                            continue
                        if len(text) > self.config.max_length:
                            text = text[: self.config.max_length]

                        tokens = self.tokenizer.encode(
                            text,
                            max_length=self.config.max_seq_length,
                            truncation=self.config.truncation,
                            padding=False,
                        )
                        yield {
                            "input_ids": tokens["input_ids"],
                            "attention_mask": tokens["attention_mask"],
                        }
                    except json.JSONDecodeError:
                        continue
