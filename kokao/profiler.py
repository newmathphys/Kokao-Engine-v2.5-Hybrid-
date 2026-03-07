"""
Модуль профилирования с torch.profiler.
Основан на встроенном профайлере PyTorch.
"""

import torch
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from .core import KokaoCore


class KokaoProfiler:
    """Профилировщик для KokaoCore."""

    def __init__(self, core: KokaoCore, output_dir: str = "./profiler_output"):
        """
        Инициализация профайлера.

        Args:
            core: Экземпляр KokaoCore
            output_dir: Директория для вывода результатов
        """
        self.core = core
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profiler = None
        self.results: Dict[str, Any] = {}

    @contextmanager
    def profile(
        self,
        activities: Optional[List] = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True
    ):
        """
        Контекстный менеджер для профилирования.

        Args:
            activities: Список активностей для профилирования
            record_shapes: Записывать ли формы тензоров
            profile_memory: Профилировать ли память
            with_stack: Записывать ли стек вызовов
        """
        if activities is None:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU]

        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=self._on_trace,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack
        )

        try:
            self.profiler.__enter__()
            yield self
        finally:
            self.profiler.__exit__(None, None, None)

    def _on_trace(self, prof: torch.profiler.profiler.profile):
        """Обработчик трассировки."""
        output_path = self.output_dir / f"trace_{int(time.time())}.json"
        prof.export_chrome_trace(str(output_path))
        self.results['trace_path'] = str(output_path)

    def profile_signal(self, x: torch.Tensor, num_runs: int = 10) -> Dict[str, float]:
        """
        Профилирование метода signal.

        Args:
            x: Входной вектор
            num_runs: Количество запусков

        Returns:
            Статистика времени выполнения
        """
        times = []

        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.core.signal(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        self.results['signal'] = {
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'std': torch.tensor(times).std().item()
        }

        return self.results['signal']

    def profile_train(self, x: torch.Tensor, target: float,
                      num_runs: int = 10) -> Dict[str, float]:
        """
        Профилирование метода train.

        Args:
            x: Входной вектор
            target: Целевое значение
            num_runs: Количество запусков

        Returns:
            Статистика времени выполнения
        """
        times = []

        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.core.train(x, target=target, lr=0.01, mode="gradient")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        self.results['train'] = {
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'std': torch.tensor(times).std().item()
        }

        return self.results['train']

    def profile_train_batch(self, X: torch.Tensor, targets: torch.Tensor,
                            num_runs: int = 10) -> Dict[str, float]:
        """
        Профилирование метода train_batch.

        Args:
            X: Входные данные (Batch, input_dim)
            targets: Целевые значения (Batch,)
            num_runs: Количество запусков

        Returns:
            Статистика времени выполнения
        """
        times = []

        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.core.train_batch(X, targets, lr=0.01)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        self.results['train_batch'] = {
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'std': torch.tensor(times).std().item()
        }

        return self.results['train_batch']

    def get_summary(self) -> str:
        """Получить текстовую сводку результатов."""
        summary = ["Profiler Summary", "=" * 40]

        for method, stats in self.results.items():
            if isinstance(stats, dict) and 'mean' in stats:
                summary.append(
                    f"{method}: mean={stats['mean']*1000:.3f}ms, "
                    f"min={stats['min']*1000:.3f}ms, max={stats['max']*1000:.3f}ms"
                )

        if 'trace_path' in self.results:
            summary.append(f"Trace saved to: {self.results['trace_path']}")

        return "\n".join(summary)

    def save_results(self, filename: str = "profiler_results.json"):
        """Сохранить результаты в JSON."""
        output_path = self.output_dir / filename

        # Конвертируем тензоры в списки для JSON
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: (v.tolist() if hasattr(v, 'tolist') else v)
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        return str(output_path)


def quick_profile(core: KokaoCore, input_dim: int = 10,
                  batch_size: int = 32) -> str:
    """
    Быстрое профилирование основных методов.

    Args:
        core: Экземпляр KokaoCore
        input_dim: Размерность входа
        batch_size: Размер батча

    Returns:
        Текстовая сводка
    """
    profiler = KokaoProfiler(core)

    x = torch.randn(input_dim)
    X_batch = torch.randn(batch_size, input_dim)
    targets = torch.rand(batch_size)

    profiler.profile_signal(x)
    profiler.profile_train(x, target=0.8)
    profiler.profile_train_batch(X_batch, targets)

    return profiler.get_summary()
