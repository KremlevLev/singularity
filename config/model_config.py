from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """
    Архитектурные параметры гибридной модели Qwen3-14B (36 Mamba-2 + 12 GQA).
    """
    # --- Основные параметры (Qwen3-14B) ---
    vocab_size: int = 152064
    hidden_size: int = 5120
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    intermediate_size: int = 13824
    max_position_embeddings: int = 8192

    # --- Параметры гибридной архитектуры ---
    mamba_layers: int = 36
    attention_layers: int = 12
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand_factor: int = 2

    # --- Параметры инициализации ---
    use_weight_subcloning: bool = True
    init_from_qwen: bool = True
    qwen_model_id: str = "Qwen/Qwen2.5-14B"

    # --- Параметры нормализации ---
    rms_norm_eps: float = 1e-6
    use_parallel_residual: bool = True

    # --- Прочее ---
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    # --- Распределение слоёв (Attention слои) ---
    attention_layer_indices: List[int] = field(default_factory=lambda: [0][4][8][12][16][20][24][28][32][36][40][44])

    def __post_init__(self):
        assert self.mamba_layers + self.attention_layers == self.num_hidden_layers, \
            f"Сумма Mamba ({self.mamba_layers}) и Attention ({self.attention_layers}) слоёв должна быть равна {self.num_hidden_layers}"
        assert len(self.attention_layer_indices) == self.attention_layers, \
            f"Количество индексов Attention слоёв ({len(self.attention_layer_indices)}) должно быть равно {self.attention_layers}"
