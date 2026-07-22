from dataclasses import dataclass
#from typing import Optional, List

@dataclass
class ModelConfig:
    # --- Основные параметры ---
    vocab_size: int = 152064          # Размер словаря Qwen2.5-14B
    hidden_size: int = 5120           # d_model
    num_hidden_layers: int = 48       # Всего слоёв
    num_attention_heads: int = 40     # Для GQA
    num_key_value_heads: int = 8      # Для GQA (групп)
    intermediate_size: int = 13824    # Размер FFN
    max_position_embeddings: int = 8192  # Максимальная длина контекста

    # --- Параметры гибридной архитектуры ---
    mamba_layers: int = 36            # Количество Mamba-слоёв
    attention_layers: int = 12        # Количество Attention-слоёв
    mamba_d_state: int = 128          # Размер состояния Mamba-2 (SSM)
    mamba_d_conv: int = 4             # Размер свёртки Mamba-2
    mamba_expand_factor: int = 2      # Коэффициент расширения Mamba-2

    # --- Параметры инициализации ---
    use_weight_subcloning: bool = True  # Использовать weight sub-cloning из Qwen
    init_from_qwen: bool = True         # Загружать веса из Qwen2.5

    # --- Параметры нормализации ---
    rms_norm_eps: float = 1e-6
    use_parallel_residual: bool = True  # Параллельный residual (как в Mamba-2)

    # --- Прочее ---
    rope_theta: float = 1000000.0       # Для RoPE (если нужно)
    tie_word_embeddings: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
