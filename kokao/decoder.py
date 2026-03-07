# kokao/decoder.py
"""
Декодер для генерации входного вектора по целевому сигналу.

Обёртка над InverseProblem с удобными параметрами по умолчанию.
"""
import torch
from typing import Optional
from .inverse import InverseProblem


class Decoder:
    """
    Декодер для генерации входного вектора по целевому сигналу.
    
    Автоматически адаптиет max_steps в зависимости от S_target:
    - |S| < 10: 200 шагов
    - |S| >= 10: 500 шагов (экстремальные значения)
    """

    def __init__(
        self,
        core,
        lr: float = 0.05,
        max_steps: Optional[int] = None,  # None = автовыбор
    ):
        """
        Инициализация декодера.

        Args:
            core: KokaoCore ядро
            lr: Learning rate для оптимизации
            max_steps: Максимальное число шагов оптимизации (None = автовыбор)
        """
        self.core = core
        self.lr = lr
        self.max_steps = max_steps  # Может быть None для автовыбора

    def generate(
        self,
        S_target: float,
        x_init: Optional[torch.Tensor] = None,
        verbose: Optional[bool] = None,
        max_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Генерация вектора для достижения целевого сигнала.

        Args:
            S_target: Целевой сигнал
            x_init: Начальное приближение (опционально)
            verbose: Выводить ли прогресс (None = использовать GLOBAL_DEBUG)
            max_steps: Переопределение max_steps (None = использовать self.max_steps или автовыбор)

        Returns:
            Сгенерированный вектор
        """
        # Импортируем DEBUG динамически
        from . import DEBUG as GLOBAL_DEBUG

        if verbose is None:
            verbose = GLOBAL_DEBUG

        # Используем переданный max_steps или self.max_steps (может быть None)
        steps_to_use = max_steps if max_steps is not None else self.max_steps

        inverse = self.core.to_inverse_problem()
        return inverse.solve(
            S_target,
            x_init=x_init,
            lr=self.lr,
            max_steps=steps_to_use,  # Передаём None для автовыбора
            verbose=verbose
        )
