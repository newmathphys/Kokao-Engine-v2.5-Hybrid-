"""
Модуль, реализующий режимы сна и гипноза.
Основан на книге Косякова Ю.Б., п.2.3.6.
"""

import torch
import random
from typing import List, Optional
from .core import KokaoCore


class MentalStateManager:
    """Управляет состояниями сна и гипноза для консолидации памяти и внушения."""

    def __init__(
        self,
        core: KokaoCore,
        memory_pool: Optional[List[torch.Tensor]] = None
    ):
        """
        Инициализация менеджера состояний.

        Args:
            core: Экземпляр KokaoCore
            memory_pool: Пул памяти (список векторов)
        """
        self.core = core
        self.memory_pool = memory_pool if memory_pool is not None else []

    def add_to_memory(self, x: torch.Tensor):
        """
        Добавление вектора в пул памяти.

        Args:
            x: Вектор для сохранения
        """
        self.memory_pool.append(x.clone().detach())

    def sleep_cycle(
        self,
        epochs: int = 5,
        lr: float = 0.001,
        noise_level: float = 0.05
    ):
        """
        Консолидация памяти во сне (обучение на зашумленных эталонах).

        Args:
            epochs: Количество эпох обучения
            lr: Скорость обучения
            noise_level: Уровень шума для зашумления
        """
        if not self.memory_pool:
            return
        for _ in range(epochs):
            for x in self.memory_pool:
                x_noisy = x + torch.randn_like(x) * noise_level
                s = self.core.signal(x_noisy)
                self.core.train(x_noisy, target=s, lr=lr, mode="gradient")

    def hypnosis_imprint(
        self,
        x_pattern: torch.Tensor,
        target_signal: float,
        strength: float = 1.0,
        skip_normalize: bool = False
    ):
        """
        Прямая запись образа в веса (гипноз).

        Args:
            x_pattern: Вектор образа для внушения
            target_signal: Целевой сигнал
            strength: Сила внушения
            skip_normalize: Пропустить нормализацию весов
        """
        with torch.no_grad():
            s_current = self.core.signal(x_pattern)
            error = target_signal - s_current
            delta_w = error * x_pattern * strength
            self.core.w_plus += delta_w * 0.5
            self.core.w_minus -= delta_w * 0.5
            if not skip_normalize:
                self.core._normalize()
