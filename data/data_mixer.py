from typing import List, Optional, Iterator
import random
import numpy as np

from .dataset import PretrainDataset


class DataMixer:
    """
    Смешивает несколько датасетов с заданными пропорциями.
    Возвращает бесконечный итератор батчей.
    """

    def __init__(
        self,
        datasets: List[PretrainDataset],
        ratios: List[float],
        seed: int = 42,
    ):
        assert len(datasets) == len(ratios), \
            f"Количество датасетов ({len(datasets)}) должно совпадать с количеством пропорций ({len(ratios)})"
        assert abs(sum(ratios) - 1.0) < 1e-6, "Сумма пропорций должна быть 1.0"

        self.datasets = datasets
        self.ratios = ratios
        self.rng = random.Random(seed)

        # Создаём итераторы для каждого датасета
        self.iterators = [iter(ds) for ds in datasets]

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        """Возвращает следующий батч из случайного датасета (согласно пропорциям)."""
        # Выбираем датасет согласно пропорциям
        dataset_idx = self.rng.choices(range(len(self.datasets)), weights=self.ratios, k=1)[0]

        # Берём следующий пример из выбранного датасета
        try:
            return next(self.iterators[dataset_idx])
        except StopIteration:
            # Если датасет закончился, пересоздаём итератор
            self.iterators[dataset_idx] = iter(self.datasets[dataset_idx])
            return next(self.iterators[dataset_idx])

    def get_batch(self) -> dict:
        """Синоним для __next__."""
        return self.__next__()
