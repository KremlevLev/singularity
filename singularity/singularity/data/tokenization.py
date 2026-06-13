from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tiktoken


class TokenizerAdapter:
    def __init__(self, name_or_path: str | Path = "cl100k_base") -> None:
        self.name_or_path = str(name_or_path)
        try:
            self.tokenizer = tiktoken.get_encoding(self.name_or_path)
        except ValueError:
            self.tokenizer = tiktoken.encoding_from_model(self.name_or_path)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, tokens: Sequence[int], **kwargs: Any) -> str:
        return self.tokenizer.decode(list(tokens), **kwargs)

    def encode_batch(self, texts: Sequence[str], **kwargs: Any) -> list[list[int]]:
        return [self.encode(text, **kwargs) for text in texts]

    def pad_batch(self, encoded: Sequence[Sequence[int]], pad_id: int = 0) -> np.ndarray:
        max_len = max(len(item) for item in encoded)
        array = np.full((len(encoded), max_len), pad_id, dtype=np.int64)
        for row, tokens in enumerate(encoded):
            array[row, : len(tokens)] = np.asarray(tokens, dtype=np.int64)
        return array
