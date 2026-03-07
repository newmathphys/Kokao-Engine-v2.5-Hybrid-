"""
Модуль систем наведения с нелинейной обратной связью.
Основан на книге Косякова Ю.Б., п.1.6.2.
"""

import torch
from .core import KokaoCore


class GuidanceSystem:
    """Управляющая система на основе интуитивного ядра."""

    def __init__(self, core: KokaoCore):
        """
        Инициализация системы наведения.

        Args:
            core: Экземпляр KokaoCore
        """
        self.core = core
        self.target_signal = 1.0

    def set_target(self, target: float):
        """
        Установка целевого сигнала.

        Args:
            target: Целевое значение сигнала
        """
        self.target_signal = target

    def compute_control_vector(
        self,
        current_x: torch.Tensor,
        max_steps: int = 20
    ) -> torch.Tensor:
        """
        Вычисляет вектор коррекции для достижения целевого сигнала.

        Args:
            current_x: Текущий вектор состояния
            max_steps: Количество шагов оптимизации

        Returns:
            Вектор коррекции
        """
        solver = self.core.to_inverse_problem()
        x_ideal = solver.solve(S_target=self.target_signal, max_steps=max_steps)
        error_vector = x_ideal - current_x
        error_magnitude = torch.norm(error_vector)
        gain = 1.0 + torch.log1p(error_magnitude)
        return error_vector * gain

    def step(
        self,
        current_x: torch.Tensor,
        dt: float = 0.1,
        max_steps: int = 20
    ) -> torch.Tensor:
        """
        Выполняет один шаг симуляции наведения.

        Args:
            current_x: Текущий вектор состояния
            dt: Шаг по времени
            max_steps: Количество шагов оптимизации

        Returns:
            Новый вектор состояния
        """
        delta = self.compute_control_vector(current_x, max_steps)
        return current_x + delta * dt

    def simulate(
        self,
        initial_x: torch.Tensor,
        steps: int = 10,
        dt: float = 0.1
    ) -> list:
        """
        Запускает симуляцию наведения на несколько шагов.

        Args:
            initial_x: Начальный вектор состояния
            steps: Количество шагов симуляции
            dt: Шаг по времени

        Returns:
            Список векторов состояния (траектория)
        """
        trajectory = [initial_x]
        x = initial_x.clone()
        for _ in range(steps):
            x = self.step(x, dt)
            trajectory.append(x)
        return trajectory
