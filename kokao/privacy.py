"""Модуль дифференциальной приватности для KokaoCore."""
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from dataclasses import dataclass
import math

from .core import KokaoCore
from .core_base import CoreConfig

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudget:
    """
    Бюджет приватности (epsilon, delta).
    """
    epsilon: float  # Параметр эпсилон-дифференциальной приватности
    delta: float    # Параметр дельта
    spent_epsilon: float = 0.0

    def remaining(self) -> float:
        """Оставшийся бюджет приватности."""
        return max(0.0, self.epsilon - self.spent_epsilon)

    def can_spend(self, amount: float) -> bool:
        """Можно ли потратить указанное количество бюджета."""
        return self.remaining() >= amount

    def spend(self, amount: float) -> None:
        """Трата бюджета."""
        self.spent_epsilon += amount
        logger.info(f"Spent {amount:.4f} privacy budget. Remaining: {self.remaining():.4f}")


class GaussianMechanism:
    """
    Гауссовский механизм для дифференциальной приватности.
    """

    def __init__(self, sensitivity: float, epsilon: float, delta: float = 1e-5):
        """
        Инициализация механизма.

        Args:
            sensitivity: Чувствительность функции (L2 норма)
            epsilon: Параметр эпсилон
            delta: Параметр дельта
        """
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta

        # Вычисление стандартного отклонения шума
        self.sigma = self._compute_sigma()

    def _compute_sigma(self) -> float:
        """Вычисление необходимого стандартного отклонения."""
        # Формула для (epsilon, delta)-DP
        if self.epsilon <= 0:
            return float('inf')

        c = np.sqrt(2 * np.log(1.25 / self.delta))
        sigma = max(self.sensitivity * c / self.epsilon, self.sensitivity / np.sqrt(2 * self.epsilon))

        return sigma

    def add_noise(self, value: torch.Tensor) -> torch.Tensor:
        """
        Добавление гауссовского шума.

        Args:
            value: Тензор для защиты

        Returns:
            Тензор с шумом
        """
        noise = torch.randn_like(value) * self.sigma
        return value + noise

    def add_noise_to_gradient(self, gradient: torch.Tensor, 
                               clip_norm: float = 1.0) -> torch.Tensor:
        """
        Добавление шума к градиенту с клиппированием.

        Args:
            gradient: Градиент
            clip_norm: Норма для клиппирования

        Returns:
            Защищенный градиент
        """
        # Клиппирование градиента
        grad_norm = torch.norm(gradient)
        if grad_norm > clip_norm:
            gradient = gradient * (clip_norm / grad_norm)

        # Добавление шума
        noise_scale = self.sigma * clip_norm
        noise = torch.randn_like(gradient) * noise_scale

        return gradient + noise


class MomentsAccountant:
    """
    Moments Accountant для точного отслеживания бюджета приватности.
    """

    def __init__(self, noise_multiplier: float, sensitivity: float = 1.0):
        """
        Инициализация accountant.

        Args:
            noise_multiplier: Множитель шума (sigma / sensitivity)
            sensitivity: Чувствительность
        """
        self.noise_multiplier = noise_multiplier
        self.sensitivity = sensitivity

        # Моменты
        self.moments: List[float] = []
        self.num_steps = 0

    def step(self, sampling_rate: float = 1.0) -> None:
        """
        Один шаг обучения с приватностью.

        Args:
            sampling_rate: Вероятность включения записи в батч
        """
        # Вычисление момента для этого шага
        moment = self._compute_moment(sampling_rate)
        self.moments.append(moment)
        self.num_steps += 1

    def _compute_moment(self, sampling_rate: float, 
                        order: int = 2) -> float:
        """
        Вычисление момента заданного порядка.

        Args:
            sampling_rate: Вероятность сэмплирования
            order: Порядок момента

        Returns:
            Значение момента
        """
        # Упрощенная формула для гауссовского механизма
        if order == 2:
            return (sampling_rate ** 2) / (2 * self.noise_multiplier ** 2)
        else:
            # Для высших порядков
            return (sampling_rate ** order) * order / (self.noise_multiplier ** 2)

    def get_epsilon(self, delta: float = 1e-5) -> float:
        """
        Получение общего epsilon для данного delta.

        Args:
            delta: Параметр дельта

        Returns:
            Значение epsilon
        """
        if not self.moments:
            return 0.0

        # Сумма моментов
        total_moment = sum(self.moments)

        # Вычисление epsilon
        orders = [2]  # Можно добавить больше порядков для точности
        epsilon = float('inf')

        for order in orders:
            # Граница через моменты
            eps_bound = (total_moment * order + np.log(1/delta)) / (order - 1)
            epsilon = min(epsilon, eps_bound)

        return epsilon

    def reset(self) -> None:
        """Сброс accountant."""
        self.moments.clear()
        self.num_steps = 0


class DPSGD:
    """
    Стохастический градиентный спуск с дифференциальной приватностью (DP-SGD).
    """

    def __init__(self, core: KokaoCore, noise_multiplier: float = 1.0,
                 clip_norm: float = 1.0, learning_rate: float = 0.01,
                 privacy_budget: Optional[PrivacyBudget] = None):
        """
        Инициализация DP-SGD оптимизатора.

        Args:
            core: KokaoCore для оптимизации
            noise_multiplier: Множитель шума
            clip_norm: Норма для клиппирования градиентов
            learning_rate: Скорость обучения
            privacy_budget: Бюджет приватности
        """
        self.core = core
        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm
        self.learning_rate = learning_rate
        self.privacy_budget = privacy_budget or PrivacyBudget(epsilon=10.0, delta=1e-5)

        # Moments accountant
        self.accountant = MomentsAccountant(noise_multiplier)

        # Статистика
        self.training_history: List[Dict[str, float]] = []

    def compute_per_sample_gradients(self, X: torch.Tensor, 
                                      targets: torch.Tensor) -> List[torch.Tensor]:
        """
        Вычисление градиентов для каждого примера.

        Args:
            X: Входные данные
            targets: Целевые значения

        Returns:
            Список градиентов для каждого примера
        """
        gradients = []

        for x, target in zip(X, targets):
            # Вычисление градиента для одного примера
            self.core.optimizer.zero_grad()

            prediction = self.core.forward(x.unsqueeze(0))
            loss = (prediction - target) ** 2
            loss.backward()

            # Сбор градиентов
            grad = torch.cat([
                self.core.w_plus.grad.flatten(),
                self.core.w_minus.grad.flatten()
            ])

            gradients.append(grad.clone())

        return gradients

    def clip_and_noise(self, gradients: List[torch.Tensor],
                       sampling_rate: float) -> torch.Tensor:
        """
        Клиппирование и добавление шума к градиентам.

        Args:
            gradients: Градиенты для каждого примера
            sampling_rate: Вероятность сэмплирования

        Returns:
            Агрегированный защищенный градиент
        """
        # Клиппирование каждого градиента
        clipped_grads = []
        for grad in gradients:
            grad_norm = torch.norm(grad)
            if grad_norm > self.clip_norm:
                grad = grad * (self.clip_norm / grad_norm)
            clipped_grads.append(grad)

        # Средний градиент
        avg_grad = torch.stack(clipped_grads).mean(dim=0)

        # Добавление шума
        noise_scale = self.noise_multiplier * self.clip_norm
        noise = torch.randn_like(avg_grad) * noise_scale
        noisy_grad = avg_grad + noise

        # Обновление accountant
        self.accountant.step(sampling_rate)

        return noisy_grad

    def step(self, X: torch.Tensor, targets: torch.Tensor,
             sampling_rate: float = 0.1) -> Dict[str, float]:
        """
        Один шаг DP-SGD.

        Args:
            X: Входные данные
            targets: Целевые значения
            sampling_rate: Вероятность сэмплирования

        Returns:
            Статистика шага
        """
        # Вычисление градиентов
        per_sample_grads = self.compute_per_sample_gradients(X, targets)

        # Клиппирование и шум
        noisy_grad = self.clip_and_noise(per_sample_grads, sampling_rate)

        # Разделение градиента на w_plus и w_minus
        dim = self.core.config.input_dim
        grad_plus = noisy_grad[:dim].reshape_as(self.core.w_plus)
        grad_minus = noisy_grad[dim:].reshape_as(self.core.w_minus)

        # Обновление весов
        with torch.no_grad():
            self.core.w_plus -= self.learning_rate * grad_plus
            self.core.w_minus -= self.learning_rate * grad_minus
            self.core._normalize()

        # Вычисление потерь
        with torch.no_grad():
            predictions = self.core.forward(X)
            loss = ((predictions - targets) ** 2).mean().item()

        # Вычисление текущего epsilon
        current_epsilon = self.accountant.get_epsilon(self.privacy_budget.delta)

        stats = {
            'loss': loss,
            'epsilon': current_epsilon,
            'noise_scale': self.noise_multiplier * self.clip_norm
        }

        self.training_history.append(stats)
        self.privacy_budget.spend(current_epsilon - sum(h['epsilon'] for h in self.training_history[:-1]))

        return stats

    def train(self, X: torch.Tensor, targets: torch.Tensor,
              num_epochs: int = 100, batch_size: int = 32,
              verbose: bool = True) -> List[Dict[str, float]]:
        """
        Обучение с дифференциальной приватностью.

        Args:
            X: Входные данные
            targets: Целевые значения
            num_epochs: Количество эпох
            batch_size: Размер батча
            verbose: Выводить ли прогресс

        Returns:
            История обучения
        """
        num_samples = len(X)
        sampling_rate = batch_size / num_samples

        for epoch in range(num_epochs):
            # Перемешивание данных
            indices = torch.randperm(num_samples)
            X_shuffled = X[indices]
            targets_shuffled = targets[indices]

            epoch_stats = []

            for i in range(0, num_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_targets = targets_shuffled[i:i+batch_size]

                stats = self.step(batch_X, batch_targets, sampling_rate)
                epoch_stats.append(stats)

            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = np.mean([s['loss'] for s in epoch_stats])
                current_epsilon = epoch_stats[-1]['epsilon']
                logger.info(f"Epoch {epoch + 1}/{num_epochs}: "
                           f"Loss = {avg_loss:.4f}, Epsilon = {current_epsilon:.2f}")

        return self.training_history

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Получение потраченного бюджета приватности.

        Returns:
            (epsilon, delta)
        """
        epsilon = self.accountant.get_epsilon(self.privacy_budget.delta)
        return epsilon, self.privacy_budget.delta


def add_dp_noise_to_weights(weights: torch.Tensor, 
                            epsilon: float, delta: float = 1e-5,
                            sensitivity: float = 1.0) -> torch.Tensor:
    """
    Добавление шума для дифференциальной приватности к весам.

    Args:
        weights: Веса модели
        epsilon: Параметр эпсилон
        delta: Параметр дельта
        sensitivity: Чувствительность

    Returns:
        Защищенные веса
    """
    mechanism = GaussianMechanism(sensitivity, epsilon, delta)
    return mechanism.add_noise(weights)


def compute_privacy_budget(noise_multiplier: float, num_steps: int,
                           sampling_rate: float, delta: float = 1e-5) -> float:
    """
    Вычисление бюджета приватности для заданных параметров.

    Args:
        noise_multiplier: Множитель шума
        num_steps: Количество шагов
        sampling_rate: Вероятность сэмплирования
        delta: Параметр дельта

    Returns:
        Значение epsilon
    """
    accountant = MomentsAccountant(noise_multiplier)

    for _ in range(num_steps):
        accountant.step(sampling_rate)

    return accountant.get_epsilon(delta)
