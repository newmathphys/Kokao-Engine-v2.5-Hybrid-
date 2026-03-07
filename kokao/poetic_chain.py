"""
Модуль, реализующий принцип стихотворения – цепочки действий.
Основан на книге Косякова Ю.Б., п.1.6.4.
"""

import torch
from typing import Callable, List, Dict
from .core import KokaoCore


class PoeticChain:
    """Цепочка превращений, где переход определяется сигналом."""

    def __init__(self, core: KokaoCore):
        """
        Инициализация цепочки.

        Args:
            core: Экземпляр KokaoCore
        """
        self.core = core

    def run_sequence(
        self,
        initial_x: torch.Tensor,
        transition_rule: Callable[[float], torch.Tensor],
        steps: int = 10
    ) -> List[Dict]:
        """
        Запускает цепочку превращений.

        Args:
            initial_x: Начальное состояние
            transition_rule: Функция перехода (сигнал → изменение)
            steps: Количество шагов

        Returns:
            История состояний
        """
        history = []
        current_x = initial_x.clone()
        for step in range(steps):
            s = self.core.signal(current_x)
            delta_x = transition_rule(s)
            current_x = current_x + delta_x
            history.append({
                'step': step,
                'signal': s,
                'state': current_x.tolist(),
                'delta': delta_x.tolist()
            })
        return history

    def learn_transition(
        self,
        dataset: List[tuple],
        lr: float = 0.01,
        epochs: int = 10
    ) -> Callable:
        """
        Обучает нейросеть, аппроксимирующую transition_rule.

        Args:
            dataset: Список кортежей (сигнал, изменение)
            lr: Скорость обучения
            epochs: Количество эпох

        Returns:
            Обученная функция перехода
        """
        X = torch.tensor([[s] for s, _ in dataset], dtype=torch.float32)
        y = torch.stack([delta for _, delta in dataset])

        W = torch.randn(1, y.shape[1], requires_grad=True)
        b = torch.randn(y.shape[1], requires_grad=True)

        optimizer = torch.optim.Adam([W, b], lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            pred = X @ W + b
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()

        def learned_transition(s: float) -> torch.Tensor:
            s_t = torch.tensor([[s]], dtype=torch.float32)
            with torch.no_grad():
                return (s_t @ W + b).squeeze(0)

        return learned_transition
