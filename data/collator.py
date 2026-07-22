from typing import List, Dict, Optional
import torch
from torch.nn.utils.rnn import pad_sequence

from config import DataConfig
from .tokenizer import QwenTokenizer


class DataCollator:
    """
    Коллатор для батчирования и padding.
    Принимает список примеров из Dataset, возвращает батч тензоров.
    """

    def __init__(self, tokenizer: QwenTokenizer, config: DataConfig):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.config = config

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        attention_mask = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]

        # Padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        # Labels: копия input_ids, но с -100 для padding токенов
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
