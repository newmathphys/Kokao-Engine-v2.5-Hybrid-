"""Модуль CUDA Graphs для оптимизации повторяющихся вызовов."""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Callable
import logging
from pathlib import Path
import time

from .core import KokaoCore
from .core_base import CoreConfig

logger = logging.getLogger(__name__)


class CUDAGraphWrapper:
    """
    Обертка для использования CUDA Graphs с KokaoCore.
    """

    def __init__(self, core: KokaoCore, capture_inputs: Optional[torch.Tensor] = None):
        """
        Инициализация обертки.

        Args:
            core: Модель для оптимизации
            capture_inputs: Пример входа для захвата графа
        """
        self.core = core
        self.graph = None
        self.static_inputs = None
        self.static_outputs = None
        self.is_captured = False

        # Буфер для результатов
        self.output_buffer = None

        if capture_inputs is not None:
            self.capture(capture_inputs)

    def capture(self, example_inputs: torch.Tensor, 
                num_warmup_iters: int = 3,
                num_capture_iters: int = 1) -> None:
        """
        Захват CUDA графа.

        Args:
            example_inputs: Пример входа
            num_warmup_iters: Количество прогревочных итераций
            num_capture_iters: Количество итераций для захвата
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping graph capture")
            return

        device = self.core.device
        if device.type != 'cuda':
            logger.warning(f"Device is {device}, CUDA Graphs require CUDA device")
            return

        self.core.eval()

        # Прогрев
        with torch.cuda.stream(torch.cuda.Stream()):
            for _ in range(num_warmup_iters):
                _ = self.core.forward(example_inputs)

        # Создание статических буферов
        self.static_inputs = example_inputs.clone().detach()
        self.static_inputs.requires_grad_(False)

        # Захват графа
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            # Первый проход для выделения памяти
            self.static_outputs = self.core.forward(self.static_inputs)

            # Захват
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_outputs = self.core.forward(self.static_inputs)

        stream.synchronize()
        torch.cuda.current_stream().wait_stream(stream)

        self.is_captured = True
        logger.info("CUDA Graph captured successfully")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Выполнение с использованием захваченного графа.

        Args:
            inputs: Входные данные

        Returns:
            Выход модели
        """
        if not self.is_captured or self.graph is None:
            # Обычное выполнение
            return self.core.forward(inputs)

        # Копирование входа в статический буфер
        self.static_inputs.copy_(inputs)

        # Выполнение графа
        self.graph.replay()

        # Возврат результата
        return self.static_outputs

    def get_speedup(self, inputs: torch.Tensor, 
                    num_runs: int = 100) -> Dict[str, float]:
        """
        Измерение ускорения от CUDA Graphs.

        Args:
            inputs: Входные данные
            num_runs: Количество запусков

        Returns:
            Статистика ускорения
        """
        if not self.is_captured:
            return {'speedup': 1.0, 'error': 'Graph not captured'}

        # Обычное выполнение
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            _ = self.core.forward(inputs)
        torch.cuda.synchronize()
        regular_time = time.time() - start

        # Выполнение с графом
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            _ = self.forward(inputs)
        torch.cuda.synchronize()
        graph_time = time.time() - start

        speedup = regular_time / graph_time if graph_time > 0 else 1.0

        return {
            'regular_time_ms': regular_time / num_runs * 1000,
            'graph_time_ms': graph_time / num_runs * 1000,
            'speedup': speedup,
            'num_runs': num_runs
        }


class TrainingCUDAGraph:
    """
    CUDA Graphs для обучения KokaoCore.
    """

    def __init__(self, core: KokaoCore, example_x: torch.Tensor,
                 example_target: float):
        """
        Инициализация графа обучения.

        Args:
            core: Модель для обучения
            example_x: Пример входа
            example_target: Пример цели
        """
        self.core = core
        self.example_x = example_x
        self.example_target = example_target

        self.graph = None
        self.static_x = None
        self.static_target = None
        self.static_loss = None
        self.optimizer = None

        self.is_captured = False

    def capture(self, lr: float = 0.01) -> None:
        """
        Захват графа обучения.

        Args:
            lr: Скорость обучения
        """
        if not torch.cuda.is_available():
            return

        device = self.core.device
        if device.type != 'cuda':
            return

        # Создание оптимизатора
        self.optimizer = torch.optim.Adam(
            [self.core.w_plus, self.core.w_minus], lr=lr
        )

        # Статические буферы
        self.static_x = self.example_x.clone().detach()
        self.static_target = torch.tensor([self.example_target], device=device)

        # Прогрев
        for _ in range(3):
            self._training_step()

        # Захват
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            self.static_loss = self._training_step()

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_loss = self._training_step()

        stream.synchronize()
        torch.cuda.current_stream().wait_stream(stream)

        self.is_captured = True

    def _training_step(self) -> torch.Tensor:
        """Один шаг обучения."""
        self.optimizer.zero_grad()

        output = self.core.forward(self.static_x)
        loss = (output - self.static_target) ** 2

        loss.backward()
        self.optimizer.step()

        # Нормализация
        with torch.no_grad():
            self.core._normalize()

        return loss

    def step(self, x: torch.Tensor, target: float) -> float:
        """
        Выполнение шага обучения.

        Args:
            x: Входные данные
            target: Целевое значение

        Returns:
            Потери
        """
        if not self.is_captured or self.graph is None:
            # Обычное обучение
            self.core.optimizer.zero_grad()
            output = self.core.forward(x)
            loss = (output - target) ** 2
            loss.backward()
            self.core.optimizer.step()
            with torch.no_grad():
                self.core._normalize()
            return loss.item()

        # Копирование в статические буферы
        self.static_x.copy_(x)
        self.static_target.fill_(target)

        # Выполнение графа
        self.graph.replay()

        return self.static_loss.item()


class BatchedCUDAGraph:
    """
    CUDA Graphs для батчевой обработки.
    """

    def __init__(self, core: KokaoCore, max_batch_size: int = 32):
        """
        Инициализация батчевого графа.

        Args:
            core: Модель
            max_batch_size: Максимальный размер батча
        """
        self.core = core
        self.max_batch_size = max_batch_size
        self.input_dim = core.config.input_dim

        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[int, torch.Tensor] = {}
        self.static_outputs: Dict[int, torch.Tensor] = {}

    def capture_for_batch_size(self, batch_size: int) -> None:
        """
        Захват графа для конкретного размера батча.

        Args:
            batch_size: Размер батча
        """
        if not torch.cuda.is_available():
            return

        example = torch.randn(batch_size, self.input_dim, device=self.core.device)

        # Прогрев
        for _ in range(3):
            _ = self.core.forward(example)

        # Статические буферы
        self.static_inputs[batch_size] = example.clone().detach()

        # Захват
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            self.static_outputs[batch_size] = self.core.forward(
                self.static_inputs[batch_size]
            )

            self.graphs[batch_size] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graphs[batch_size]):
                self.static_outputs[batch_size] = self.core.forward(
                    self.static_inputs[batch_size]
                )

        stream.synchronize()
        torch.cuda.current_stream().wait_stream(stream)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Выполнение для батча.

        Args:
            x: Входные данные

        Returns:
            Выход модели
        """
        batch_size = x.shape[0]

        if batch_size not in self.graphs:
            self.capture_for_batch_size(batch_size)

        # Копирование входа
        self.static_inputs[batch_size].copy_(x)

        # Выполнение графа
        self.graphs[batch_size].replay()

        return self.static_outputs[batch_size]


class CUDAGraphManager:
    """
    Менеджер CUDA Graphs для управления несколькими графами.
    """

    def __init__(self, core: KokaoCore):
        """
        Инициализация менеджера.

        Args:
            core: Модель
        """
        self.core = core
        self.wrappers: Dict[str, CUDAGraphWrapper] = {}
        self.training_graphs: Dict[str, TrainingCUDAGraph] = {}

    def create_inference_graph(self, name: str, 
                                example_inputs: torch.Tensor) -> CUDAGraphWrapper:
        """
        Создание графа для инференса.

        Args:
            name: Имя графа
            example_inputs: Пример входа

        Returns:
            Обертка графа
        """
        wrapper = CUDAGraphWrapper(self.core, example_inputs)
        self.wrappers[name] = wrapper
        return wrapper

    def create_training_graph(self, name: str,
                               example_x: torch.Tensor,
                               example_target: float) -> TrainingCUDAGraph:
        """
        Создание графа для обучения.

        Args:
            name: Имя графа
            example_x: Пример входа
            example_target: Пример цели

        Returns:
            Граф обучения
        """
        graph = TrainingCUDAGraph(self.core, example_x, example_target)
        self.training_graphs[name] = graph
        return graph

    def get_wrapper(self, name: str) -> Optional[CUDAGraphWrapper]:
        """Получение обертки по имени."""
        return self.wrappers.get(name)

    def get_training_graph(self, name: str) -> Optional[TrainingCUDAGraph]:
        """Получение графа обучения по имени."""
        return self.training_graphs.get(name)

    def list_graphs(self) -> Dict[str, Any]:
        """
        Список всех графов.

        Returns:
            Информация о графах
        """
        return {
            'inference_graphs': {
                name: {'captured': w.is_captured}
                for name, w in self.wrappers.items()
            },
            'training_graphs': {
                name: {'captured': g.is_captured}
                for name, g in self.training_graphs.items()
            }
        }

    def cleanup(self) -> None:
        """Очистка всех графов."""
        self.wrappers.clear()
        self.training_graphs.clear()
        torch.cuda.empty_cache()


def enable_cuda_graphs(core: KokaoCore, 
                       example_inputs: torch.Tensor) -> CUDAGraphWrapper:
    """
    Быстрое включение CUDA Graphs.

    Args:
        core: Модель
        example_inputs: Пример входа

    Returns:
        Обертка с захваченным графом
    """
    wrapper = CUDAGraphWrapper(core, example_inputs)
    return wrapper


def benchmark_cuda_graphs(core: KokaoCore,
                          batch_sizes: List[int] = [1, 8, 16, 32],
                          num_runs: int = 100) -> Dict[str, Any]:
    """
    Бенчмарк CUDA Graphs для разных размеров батча.

    Args:
        core: Модель
        batch_sizes: Размеры батчей для тестирования
        num_runs: Количество запусков

    Returns:
        Результаты бенчмарка
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}

    results = {}

    for batch_size in batch_sizes:
        example = torch.randn(batch_size, core.config.input_dim, device=core.device)

        # Создание и захват графа
        wrapper = CUDAGraphWrapper(core, example)

        # Бенчмарк
        speedup_data = wrapper.get_speedup(example, num_runs)

        results[f'batch_{batch_size}'] = speedup_data

    return results
