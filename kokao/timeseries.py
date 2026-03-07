"""Модуль прогнозирования временных рядов на основе KokaoCore."""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from collections import deque
import numpy as np

from .core import KokaoCore
from .core_base import CoreConfig


class TimeSeriesPredictor(nn.Module):
    """
    Модель для прогнозирования временных рядов с использованием KokaoCore.
    Использует скользящее окно для предсказания следующего значения.
    """

    def __init__(self, window_size: int = 10, input_dim: int = 1, 
                 hidden_dim: int = 32, num_cores: int = 4):
        """
        Инициализация предсказателя временных рядов.

        Args:
            window_size: Размер скользящего окна
            input_dim: Размерность входных данных
            hidden_dim: Размерность скрытого представления
            num_cores: Количество ядер KokaoCore для ансамбля
        """
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_cores = num_cores

        # Входное преобразование
        self.input_projection = nn.Linear(window_size * input_dim, hidden_dim)
        
        # Ансамбль ядер KokaoCore
        self.cores = nn.ModuleList([
            KokaoCore(CoreConfig(input_dim=hidden_dim)) 
            for _ in range(num_cores)
        ])

        # Выходное преобразование
        self.output_projection = nn.Linear(num_cores, 1)

        # История для онлайн-прогнозирования
        self.history: deque = deque(maxlen=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход.

        Args:
            x: Входные данные формы (batch, window_size, input_dim) 
               или (batch, window_size * input_dim)

        Returns:
            Прогноз следующего значения формы (batch, 1)
        """
        if x.ndim == 2:
            # (batch, window_size * input_dim)
            batch_size = x.shape[0]
        else:
            # (batch, window_size, input_dim) -> (batch, window_size * input_dim)
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)

        # Входное преобразование
        hidden = torch.relu(self.input_projection(x))  # (batch, hidden_dim)

        # Прогнозы от каждого ядра
        core_outputs = []
        for core in self.cores:
            # Применяем ядро к каждому элементу батча
            signals = []
            for i in range(batch_size):
                signal = core.signal(hidden[i])
                signals.append(torch.tensor(signal, dtype=hidden.dtype))
            core_outputs.append(torch.stack(signals))

        # Стек прогнозов: (num_cores, batch)
        core_outputs = torch.stack(core_outputs)  # (num_cores, batch)
        core_outputs = core_outputs.t()  # (batch, num_cores)

        # Выходное преобразование
        prediction = self.output_projection(core_outputs)  # (batch, 1)

        return prediction

    def predict_next(self, new_value: float) -> float:
        """
        Онлайн-прогноз следующего значения.

        Args:
            new_value: Новое наблюдаемое значение

        Returns:
            Прогноз следующего значения
        """
        # Добавляем в историю
        self.history.append(new_value)

        if len(self.history) < self.window_size:
            # Недостаточно данных для прогноза
            return np.mean(list(self.history))

        # Формируем входной вектор
        x = torch.tensor(list(self.history), dtype=torch.float32).unsqueeze(0)
        
        # Делаем прогноз
        with torch.no_grad():
            prediction = self.forward(x)

        return prediction.item()

    def reset_history(self) -> None:
        """Сброс истории."""
        self.history.clear()

    def train_step(self, x: torch.Tensor, y: torch.Tensor, 
                   lr: float = 0.01) -> float:
        """
        Один шаг обучения.

        Args:
            x: Входные данные (batch, window_size * input_dim)
            y: Целевые значения (batch, 1)
            lr: Скорость обучения

        Returns:
            Значение функции потерь
        """
        self.train()
        
        # Прямой проход
        prediction = self.forward(x)
        
        # Вычисление потерь
        loss = nn.functional.mse_loss(prediction, y)
        
        # Обратный проход
        loss.backward()
        
        # Градиентный шаг
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param -= lr * param.grad
                    param.grad.zero_()

        return loss.item()


class TimeSeriesDataset:
    """
    Датасет для временных рядов со скользящим окном.
    """

    def __init__(self, data: np.ndarray, window_size: int, 
                 prediction_horizon: int = 1):
        """
        Инициализация датасета.

        Args:
            data: Данные временного ряда (N,) или (N, input_dim)
            window_size: Размер скользящего окна
            prediction_horizon: Горизонт прогнозирования
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.data = data
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon

        # Создаем пары (окно, целевое значение)
        self.X = []
        self.y = []

        for i in range(len(data) - window_size - prediction_horizon + 1):
            window = data[i:i + window_size].flatten()
            target = data[i + window_size + prediction_horizon - 1].flatten()
            self.X.append(window)
            self.y.append(target)

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

    def get_dataloader(self, batch_size: int = 32, 
                       shuffle: bool = True) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Получение данных в формате для обучения.

        Args:
            batch_size: Размер батча
            shuffle: Перемешивать ли данные

        Returns:
            Список батчей (X, y)
        """
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)

        batches = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            X_batch = torch.stack([self.X[i] for i in batch_indices])
            y_batch = torch.stack([self.y[i] for i in batch_indices])
            batches.append((X_batch, y_batch))

        return batches


def create_seasonal_features(time_index: np.ndarray, 
                             periods: List[int]) -> np.ndarray:
    """
    Создание сезонных признаков для временного ряда.

    Args:
        time_index: Индексы времени
        periods: Список периодов сезонности (например, [24, 168] для часов и недель)

    Returns:
        Массив сезонных признаков
    """
    features = []
    for period in periods:
        # Синус и косинус для кодирования периода
        features.append(np.sin(2 * np.pi * time_index / period))
        features.append(np.cos(2 * np.pi * time_index / period))

    return np.column_stack(features)
