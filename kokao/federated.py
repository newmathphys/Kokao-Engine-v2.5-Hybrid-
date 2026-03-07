"""Модуль федеративного обучения для KokaoCore."""
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import logging
from collections import defaultdict
import json
from pathlib import Path

from .core import KokaoCore
from .core_base import CoreConfig

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Конфигурация клиента федеративного обучения."""
    client_id: str
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    data_size: int = 0


class FederatedClient:
    """
    Клиент для федеративного обучения.
    Обучает локальную модель на своих данных.
    """

    def __init__(self, config: ClientConfig, input_dim: int):
        """
        Инициализация клиента.

        Args:
            config: Конфигурация клиента
            input_dim: Размерность входа модели
        """
        self.config = config
        self.input_dim = input_dim

        # Локальная модель
        self.model = KokaoCore(CoreConfig(input_dim=input_dim))

        # Локальные данные
        self.local_data: Optional[List[Tuple[torch.Tensor, float]]] = None

        # Статистика обучения
        self.training_history: List[float] = []

    def set_local_data(self, data: List[Tuple[torch.Tensor, float]]) -> None:
        """
        Установка локальных данных.

        Args:
            data: Список пар (вход, целевой сигнал)
        """
        self.local_data = data
        self.config.data_size = len(data)
        logger.info(f"Client {self.config.client_id}: loaded {len(data)} samples")

    def train_local(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Локальное обучение на основе глобальных весов.

        Args:
            global_weights: Глобальные веса модели

        Returns:
            Обновленные веса модели
        """
        if self.local_data is None:
            raise ValueError("Local data not set")

        # Загрузка глобальных весов
        self._load_weights(global_weights)

        # Локальное обучение
        losses = []
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            np.random.shuffle(self.local_data)

            for x, target in self.local_data:
                loss = self.model.train(x, target, lr=self.config.learning_rate)
                epoch_loss += loss

            avg_loss = epoch_loss / len(self.local_data)
            losses.append(avg_loss)

            if epoch % 10 == 0:
                logger.debug(f"Client {self.config.client_id}, epoch {epoch}: loss={avg_loss:.4f}")

        self.training_history.extend(losses)

        # Возврат обновленных весов
        return self._get_weights()

    def _get_weights(self) -> Dict[str, torch.Tensor]:
        """Получение текущих весов модели."""
        return {
            'w_plus': self.model.w_plus.clone().detach().cpu(),
            'w_minus': self.model.w_minus.clone().detach().cpu()
        }

    def _load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Загрузка весов в модель."""
        with torch.no_grad():
            self.model.w_plus.copy_(weights['w_plus'].to(self.model.device))
            self.model.w_minus.copy_(weights['w_minus'].to(self.model.device))


class FederatedServer:
    """
    Сервер для федеративного обучения.
    Координирует обучение между клиентами.
    """

    def __init__(self, input_dim: int, num_clients: int = 10):
        """
        Инициализация сервера.

        Args:
            input_dim: Размерность входа модели
            num_clients: Количество клиентов
        """
        self.input_dim = input_dim
        self.num_clients = num_clients

        # Глобальная модель
        self.global_model = KokaoCore(CoreConfig(input_dim=input_dim))

        # Клиенты
        self.clients: Dict[str, FederatedClient] = {}

        # Статистика раундов
        self.round_history: List[Dict[str, float]] = []

    def register_client(self, client: FederatedClient) -> None:
        """Регистрация клиента."""
        self.clients[client.config.client_id] = client
        logger.info(f"Registered client: {client.config.client_id}")

    def aggregate_weights(self, client_weights: List[Tuple[str, Dict[str, torch.Tensor], int]]
                         ) -> Dict[str, torch.Tensor]:
        """
        Агрегация весов от клиентов (FedAvg).

        Args:
            client_weights: Список (client_id, weights, data_size)

        Returns:
            Агрегированные веса
        """
        total_samples = sum(size for _, _, size in client_weights)

        aggregated = {
            'w_plus': torch.zeros_like(self.global_model.w_plus),
            'w_minus': torch.zeros_like(self.global_model.w_minus)
        }

        for client_id, weights, data_size in client_weights:
            weight_factor = data_size / total_samples

            aggregated['w_plus'] += weights['w_plus'] * weight_factor
            aggregated['w_minus'] += weights['w_minus'] * weight_factor

        logger.info(f"Aggregated weights from {len(client_weights)} clients")
        return aggregated

    def run_round(self, selected_clients: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Выполнение одного раунда федеративного обучения.

        Args:
            selected_clients: Список ID клиентов для участия (None = все)

        Returns:
            Статистика раунда
        """
        if selected_clients is None:
            selected_clients = list(self.clients.keys())

        logger.info(f"Starting round with {len(selected_clients)} clients")

        # Получение обновлений от клиентов
        client_updates = []
        for client_id in selected_clients:
            client = self.clients[client_id]

            # Отправка глобальных весов
            global_weights = self._get_global_weights()

            # Локальное обучение клиента
            local_weights = client.train_local(global_weights)

            client_updates.append((
                client_id,
                local_weights,
                client.config.data_size
            ))

        # Агрегация весов
        aggregated_weights = self.aggregate_weights(client_updates)

        # Обновление глобальной модели
        self._set_global_weights(aggregated_weights)

        # Вычисление статистики
        round_stats = self._compute_round_stats(client_updates)
        self.round_history.append(round_stats)

        logger.info(f"Round completed. Avg loss: {round_stats['avg_loss']:.4f}")
        return round_stats

    def _get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Получение весов глобальной модели."""
        return {
            'w_plus': self.global_model.w_plus.clone().detach().cpu(),
            'w_minus': self.global_model.w_minus.clone().detach().cpu()
        }

    def _set_global_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Установка весов глобальной модели."""
        with torch.no_grad():
            self.global_model.w_plus.copy_(weights['w_plus'])
            self.global_model.w_minus.copy_(weights['w_minus'])
            self.global_model._normalize()

    def _compute_round_stats(self, client_updates: List[Tuple[str, Dict, int]]) -> Dict[str, float]:
        """Вычисление статистики раунда."""
        if not client_updates:
            return {'avg_loss': 0.0}

        # Получение последних потерь от клиентов
        losses = []
        for client_id, _, _ in client_updates:
            client = self.clients[client_id]
            if client.training_history:
                losses.append(client.training_history[-1])

        return {
            'avg_loss': np.mean(losses) if losses else 0.0,
            'std_loss': np.std(losses) if len(losses) > 1 else 0.0,
            'num_clients': len(client_updates)
        }

    def evaluate(self, test_data: List[Tuple[torch.Tensor, float]]) -> Dict[str, float]:
        """
        Оценка глобальной модели на тестовых данных.

        Args:
            test_data: Тестовые данные

        Returns:
            Метрики качества
        """
        if not test_data:
            return {'mse': 0.0, 'mae': 0.0}

        errors = []
        for x, target in test_data:
            prediction = self.global_model.signal(x)
            errors.append((prediction - target) ** 2)

        mse = np.mean(errors)
        return {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(np.mean([abs(e) for e in errors]))
        }


class FederatedLearning:
    """
    Высокоуровневый интерфейс для федеративного обучения.
    """

    def __init__(self, input_dim: int, num_clients: int = 10):
        """
        Инициализация федеративного обучения.

        Args:
            input_dim: Размерность входа
            num_clients: Количество клиентов
        """
        self.server = FederatedServer(input_dim, num_clients)
        self.input_dim = input_dim

    def create_client(self, client_id: str, 
                     local_epochs: int = 5,
                     learning_rate: float = 0.01) -> FederatedClient:
        """
        Создание и регистрация клиента.

        Args:
            client_id: ID клиента
            local_epochs: Количество локальных эпох
            learning_rate: Скорость обучения

        Returns:
            Созданный клиент
        """
        config = ClientConfig(
            client_id=client_id,
            local_epochs=local_epochs,
            learning_rate=learning_rate
        )
        client = FederatedClient(config, self.input_dim)
        self.server.register_client(client)
        return client

    def train(self, num_rounds: int, 
              clients_per_round: Optional[int] = None,
              test_data: Optional[List[Tuple[torch.Tensor, float]]] = None,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Обучение с федеративным обучением.

        Args:
            num_rounds: Количество раундов
            clients_per_round: Количество клиентов на раунд (None = все)
            test_data: Тестовые данные для оценки
            verbose: Выводить ли прогресс

        Returns:
            История обучения
        """
        history = {
            'round_loss': [],
            'test_mse': []
        }

        for round_num in range(num_rounds):
            # Выбор клиентов
            if clients_per_round is not None:
                selected = list(self.server.clients.keys())
                if len(selected) > clients_per_round:
                    selected = list(np.random.choice(selected, clients_per_round, replace=False))
            else:
                selected = None

            # Выполнение раунда
            round_stats = self.server.run_round(selected)
            history['round_loss'].append(round_stats['avg_loss'])

            # Оценка на тестовых данных
            if test_data:
                test_metrics = self.server.evaluate(test_data)
                history['test_mse'].append(test_metrics['mse'])

            if verbose and (round_num + 1) % 10 == 0:
                logger.info(f"Round {round_num + 1}/{num_rounds}: "
                           f"loss={round_stats['avg_loss']:.4f}")
                if test_data:
                    logger.info(f"  Test MSE: {test_metrics['mse']:.4f}")

        return history

    def get_global_model(self) -> KokaoCore:
        """Получение глобальной модели."""
        return self.server.global_model

    def save(self, path: str) -> None:
        """Сохранение состояния."""
        state = {
            'global_model': self.server._get_global_weights(),
            'round_history': self.server.round_history,
            'client_configs': {
                cid: {
                    'client_id': c.config.client_id,
                    'local_epochs': c.config.local_epochs,
                    'learning_rate': c.config.learning_rate
                }
                for cid, c in self.server.clients.items()
            }
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load(self, path: str) -> None:
        """Загрузка состояния."""
        with open(path) as f:
            state = json.load(f)

        self.server._set_global_weights({
            'w_plus': torch.tensor(state['global_model']['w_plus']),
            'w_minus': torch.tensor(state['global_model']['w_minus'])
        })
        self.server.round_history = state['round_history']
