"""
Модуль автоматического смешанного точности (AMP) для train_batch.
Использует torch.amp API (PyTorch >= 2.1).
"""

import torch
from typing import Optional
from .core import KokaoCore


class AMPTrainer:
    """Тренер с автоматическим смешанным точностью (AMP)."""

    def __init__(self, core: KokaoCore, use_amp: bool = True):
        """
        Инициализация AMP тренера.

        Args:
            core: Экземпляр KokaoCore
            use_amp: Использовать ли AMP (только для CUDA)
        """
        self.core = core
        self.use_amp = use_amp and torch.cuda.is_available()
        # Используем новый API torch.amp.GradScaler (PyTorch >= 2.1)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

    def train_batch_amp(
        self,
        X: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.01
    ) -> float:
        """
        Обучение на батче с использованием AMP.

        Args:
            X: Входные данные (Batch, input_dim)
            targets: Целевые значения (Batch,)
            lr: Скорость обучения

        Returns:
            Значение функции потерь
        """
        if not self.use_amp:
            # Fallback на обычное обучение
            return self.core.train_batch(X, targets, lr)

        # Перемещаем данные на GPU
        X = X.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # Используем новый API torch.amp.autocast с указанием устройства
        with torch.amp.autocast('cuda'):
            # Forward pass
            S = self.core.forward(X)
            loss = ((S - targets) ** 2).mean()

        # Backward pass с масштабированием градиентов
        self.scaler.scale(loss).backward()

        with torch.no_grad():
            # Шаг оптимизатора с масштабированием
            self.scaler.step(self.core.optimizer)
            self.scaler.update()

            # Нормализация весов
            self.core._normalize()

        return loss.item()

    def train_epoch_amp(
        self,
        dataloader: torch.utils.data.DataLoader,
        lr: float = 0.01
    ) -> float:
        """
        Обучение на эпохе с использованием AMP.

        Args:
            dataloader: Загрузчик данных
            lr: Скорость обучения

        Returns:
            Средняя функция потерь за эпоху
        """
        total_loss = 0.0
        num_batches = 0

        for X_batch, targets_batch in dataloader:
            loss = self.train_batch_amp(X_batch, targets_batch, lr)
            total_loss += loss
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def is_using_amp(self) -> bool:
        """Проверка, используется ли AMP."""
        return self.use_amp
