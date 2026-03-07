"""Модуль распределенного обучения для KokaoCore."""
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from typing import Optional, Dict, Any, List, Tuple, Callable
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime

from .core import KokaoCore
from .core_base import CoreConfig

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Конфигурация распределенного обучения."""
    world_size: int = 1
    rank: int = 0
    backend: str = 'nccl'  # 'nccl', 'gloo', 'mpi'
    init_method: str = 'env://'
    gpu_ids: List[int] = None

    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = list(range(self.world_size))


class DistributedKokaoTrainer:
    """
    Тренер для распределенного обучения KokaoCore.
    """

    def __init__(self, core: KokaoCore, config: DistributedConfig):
        """
        Инициализация тренера.

        Args:
            core: Модель для обучения
            config: Конфигурация распределенного обучения
        """
        self.core = core
        self.config = config
        self.is_distributed = False

    def setup(self) -> None:
        """Настройка распределенного обучения."""
        if self.config.world_size > 1:
            torch.distributed.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            self.is_distributed = True

            # Настройка GPU
            if self.config.gpu_ids:
                gpu_id = self.config.gpu_ids[self.config.rank % len(self.config.gpu_ids)]
                torch.cuda.set_device(gpu_id)
                self.core = self.core.to(f'cuda:{gpu_id}')

            logger.info(f"Initialized distributed training on rank {self.config.rank}")

    def cleanup(self) -> None:
        """Очистка распределенного обучения."""
        if self.is_distributed:
            torch.distributed.destroy_process_group()
            self.is_distributed = False
            logger.info("Cleaned up distributed training")

    def synchronize(self) -> None:
        """Синхронизация между процессами."""
        if self.is_distributed:
            torch.distributed.barrier()

    def average_gradients(self) -> None:
        """Усреднение градиентов между процессами."""
        if not self.is_distributed:
            return

        for param in [self.core.w_plus, self.core.w_minus]:
            if param.grad is not None:
                torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                param.grad /= self.config.world_size

    def train_step(self, x: torch.Tensor, target: float, 
                   lr: float = 0.01) -> float:
        """
        Один шаг распределенного обучения.

        Args:
            x: Входные данные
            target: Целевое значение
            lr: Скорость обучения

        Returns:
            Потери
        """
        # Перемещение на GPU если нужно
        if self.is_distributed and torch.cuda.is_available():
            x = x.to(self.core.device)

        # Прямой проход
        loss = self.core.train(x, target, lr=lr)

        # Усреднение градиентов
        self.average_gradients()

        return loss

    def train_batch(self, X: torch.Tensor, targets: torch.Tensor,
                    lr: float = 0.01) -> float:
        """
        Распределенное обучение на батче.

        Args:
            X: Входные данные
            targets: Целевые значения
            lr: Скорость обучения

        Returns:
            Потери
        """
        if self.is_distributed and torch.cuda.is_available():
            X = X.to(self.core.device)
            targets = targets.to(self.core.device)

        loss = self.core.train_batch(X, targets, lr=lr)
        self.average_gradients()

        return loss

    def broadcast_weights(self, src_rank: int = 0) -> None:
        """
        Трансляция весов от главного процесса.

        Args:
            src_rank: Ранк источника
        """
        if not self.is_distributed:
            return

        with torch.no_grad():
            torch.distributed.broadcast(
                self.core.w_plus, src=src_rank
            )
            torch.distributed.broadcast(
                self.core.w_minus, src=src_rank
            )

    def gather_losses(self, loss: float) -> List[float]:
        """
        Сбор потерь со всех процессов.

        Args:
            loss: Локальные потери

        Returns:
            Потери со всех процессов
        """
        if not self.is_distributed:
            return [loss]

        loss_tensor = torch.tensor([loss], device=self.core.device)
        gathered = [torch.zeros_like(loss_tensor) for _ in range(self.config.world_size)]
        torch.distributed.all_gather(gathered, loss_tensor)

        return [t.item() for t in gathered]


def distributed_train_worker(rank: int, world_size: int,
                             data: List[Tuple[torch.Tensor, float]],
                             config: CoreConfig,
                             num_epochs: int = 10,
                             lr: float = 0.01) -> Dict[str, Any]:
    """
    Worker функция для распределенного обучения.

    Args:
        rank: Ранк процесса
        world_size: Количество процессов
        data: Данные для обучения
        config: Конфигурация ядра
        num_epochs: Количество эпох
        lr: Скорость обучения

    Returns:
        Результаты обучения
    """
    # Настройка распределенного обучения
    dist_config = DistributedConfig(
        world_size=world_size,
        rank=rank
    )

    # Создание модели
    core = KokaoCore(config)
    trainer = DistributedKokaoTrainer(core, dist_config)

    try:
        trainer.setup()

        # Разделение данных между процессами
        data_per_rank = len(data) // world_size
        local_data = data[rank * data_per_rank:(rank + 1) * data_per_rank]

        # Обучение
        history = []
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for x, target in local_data:
                loss = trainer.train_step(x, target, lr=lr)
                epoch_loss += loss

            # Усреднение потерь
            avg_loss = epoch_loss / len(local_data)
            gathered_losses = trainer.gather_losses(avg_loss)
            global_avg_loss = sum(gathered_losses) / len(gathered_losses)

            history.append({'epoch': epoch, 'loss': global_avg_loss})

            if rank == 0 and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {global_avg_loss:.4f}")

        # Синхронизация финальных весов
        trainer.broadcast_weights(0)

        results = {
            'rank': rank,
            'history': history,
            'final_weights': {
                'w_plus': core.w_plus.cpu().tolist(),
                'w_minus': core.w_minus.cpu().tolist()
            }
        }

        return results

    finally:
        trainer.cleanup()


def train_distributed(data: List[Tuple[torch.Tensor, float]],
                      input_dim: int,
                      num_epochs: int = 10,
                      num_gpus: Optional[int] = None,
                      lr: float = 0.01) -> KokaoCore:
    """
    Распределенное обучение модели.

    Args:
        data: Данные для обучения
        input_dim: Размерность входа
        num_epochs: Количество эпох
        num_gpus: Количество GPU (None = все доступные)
        lr: Скорость обучения

    Returns:
        Обученная модель
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        # Одн GPU обучение
        config = CoreConfig(input_dim=input_dim)
        core = KokaoCore(config)

        for epoch in range(num_epochs):
            for x, target in data:
                core.train(x, target, lr=lr)

        return core

    # Распределенное обучение
    config = CoreConfig(input_dim=input_dim)

    mp.spawn(
        distributed_train_worker,
        args=(num_gpus, data, config, num_epochs, lr),
        nprocs=num_gpus,
        join=True
    )

    # Загрузка обученной модели (с rank 0)
    trained_core = KokaoCore(config)
    return trained_core


class DataParallelTrainer:
    """
    Тренер с использованием DataParallel.
    """

    def __init__(self, core: KokaoCore, device_ids: Optional[List[int]] = None):
        """
        Инициализация тренера.

        Args:
            core: Модель для обучения
            device_ids: IDs устройств
        """
        self.core = core
        self.device_ids = device_ids

        if device_ids and len(device_ids) > 1:
            self.model = nn.DataParallel(core, device_ids=device_ids)
        else:
            self.model = core

    def train_batch(self, X: torch.Tensor, targets: torch.Tensor,
                    lr: float = 0.01) -> float:
        """
        Обучение на батче.

        Args:
            X: Входные данные
            targets: Целевые значения
            lr: Скорость обучения

        Returns:
            Потери
        """
        # DataParallel автоматически распределяет по устройствам
        loss = self.core.train_batch(X, targets, lr=lr)
        return loss


class RayDistributedTrainer:
    """
    Тренер с использованием Ray для распределенного обучения.
    """

    def __init__(self, num_workers: int = 4):
        """
        Инициализация тренера.

        Args:
            num_workers: Количество worker'ов
        """
        self.num_workers = num_workers
        self.workers = []

        try:
            import ray
            self.ray = ray
        except ImportError:
            raise ImportError("Ray not installed. Install with: pip install ray")

    def setup(self) -> None:
        """Настройка Ray."""
        if not self.ray.is_initialized():
            self.ray.init()

    def create_worker(self, config: CoreConfig) -> Any:
        """
        Создание worker'а.

        Args:
            config: Конфигурация ядра

        Returns:
            Remote worker
        """
        @self.ray.remote
        class Worker:
            def __init__(self, core_config):
                self.core = KokaoCore(core_config)

            def train(self, data, lr):
                for x, target in data:
                    self.core.train(x, target, lr=lr)

            def get_weights(self):
                return {
                    'w_plus': self.core.w_plus.cpu().tolist(),
                    'w_minus': self.core.w_minus.cpu().tolist()
                }

            def set_weights(self, weights):
                self.core.w_plus = torch.nn.Parameter(
                    torch.tensor(weights['w_plus'])
                )
                self.core.w_minus = torch.nn.Parameter(
                    torch.tensor(weights['w_minus'])
                )

        return Worker.remote(config)

    def train(self, data: List[Tuple[torch.Tensor, float]],
              input_dim: int, num_epochs: int = 10,
              lr: float = 0.01) -> KokaoCore:
        """
        Распределенное обучение с Ray.

        Args:
            data: Данные для обучения
            input_dim: Размерность входа
            num_epochs: Количество эпох
            lr: Скорость обучения

        Returns:
            Обученная модель
        """
        self.setup()

        config = CoreConfig(input_dim=input_dim)

        # Создание worker'ов
        workers = [self.create_worker(config) for _ in range(self.num_workers)]

        # Разделение данных
        data_per_worker = len(data) // self.num_workers

        # Обучение
        for epoch in range(num_epochs):
            # Обучение на каждом worker'е
            futures = []
            for i, worker in enumerate(workers):
                worker_data = data[i * data_per_worker:(i + 1) * data_per_worker]
                futures.append(worker.train.remote(worker_data, lr))

            # Ожидание завершения
            self.ray.get(futures)

            # Сбор и усреднение весов
            weight_futures = [w.get_weights.remote() for w in workers]
            all_weights = self.ray.get(weight_futures)

            avg_weights = {
                'w_plus': torch.mean(torch.tensor([
                    torch.tensor(w['w_plus']) for w in all_weights
                ]), dim=0).tolist(),
                'w_minus': torch.mean(torch.tensor([
                    torch.tensor(w['w_minus']) for w in all_weights
                ]), dim=0).tolist()
            }

            # Установка усредненных весов
            for worker in workers:
                worker.set_weights.remote(avg_weights)

        # Получение финальной модели
        final_weights = self.ray.get(workers[0].get_weights.remote())

        # Создание финальной модели
        final_core = KokaoCore(config)
        final_core.w_plus = torch.nn.Parameter(torch.tensor(final_weights['w_plus']))
        final_core.w_minus = torch.nn.Parameter(torch.tensor(final_weights['w_minus']))

        return final_core

    def cleanup(self) -> None:
        """Очистка Ray."""
        if self.ray.is_initialized():
            self.ray.shutdown()


def create_distributed_trainer(core: KokaoCore, 
                               distributed: bool = True,
                               use_ray: bool = False,
                               num_workers: int = 4) -> Any:
    """
    Создание распределенного тренера.

    Args:
        core: Модель для обучения
        distributed: Использовать ли distributed training
        use_ray: Использовать ли Ray
        num_workers: Количество worker'ов

    Returns:
        Тренер
    """
    if use_ray:
        return RayDistributedTrainer(num_workers)
    elif distributed:
        config = DistributedConfig(world_size=num_workers)
        return DistributedKokaoTrainer(core, config)
    else:
        return DataParallelTrainer(core)
