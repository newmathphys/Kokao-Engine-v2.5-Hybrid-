"""
Модуль сбора и визуализации метрик через Prometheus + Grafana.
Основан на клиенте Prometheus для Python.
"""

import time
import logging
import socket
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Опциональный импорт
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Gauge = None
    Histogram = None
    start_http_server = None

from .core import KokaoCore

logger = logging.getLogger(__name__)


def _find_free_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """
    Поиск свободного порта начиная с start_port.
    
    Args:
        start_port: Начальный порт для проверки
        max_attempts: Максимальное количество попыток
        
    Returns:
        Свободный порт
    """
    for port_offset in range(max_attempts):
        port = start_port + port_offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    
    # Если не нашли свободный, вернём -1 (ошибка)
    return -1


@dataclass
class TrainingMetrics:
    """Метрики обучения."""
    total_iterations: int = 0
    total_samples: int = 0
    last_loss: float = 0.0
    avg_loss: float = 0.0
    total_time: float = 0.0
    samples_per_second: float = 0.0


class KokaoMetrics:
    """Сборщик метрик для Kokao."""

    def __init__(self, prefix: str = "kokao", registry=None):
        """
        Инициализация метрик.

        Args:
            prefix: Префикс для имён метрик
            registry: Реестр Prometheus (по умолчанию глобальный)
        """
        self.prefix = prefix
        self.registry = registry
        self.metrics: Dict[str, Any] = {}

        if PROMETHEUS_AVAILABLE:
            self._register_metrics()

    def _register_metrics(self):
        """Регистрация метрик Prometheus."""
        from prometheus_client import CollectorRegistry, REGISTRY

        # Используем переданный реестр или глобальный
        registry = self.registry if self.registry else REGISTRY

        # Счётчики
        self.metrics['train_iterations_total'] = Counter(
            f'{self.prefix}_train_iterations_total',
            'Total number of training iterations',
            registry=registry
        )

        self.metrics['train_samples_total'] = Counter(
            f'{self.prefix}_train_samples_total',
            'Total number of training samples processed',
            registry=registry
        )

        # Гauges
        self.metrics['current_loss'] = Gauge(
            f'{self.prefix}_current_loss',
            'Current training loss',
            registry=registry
        )

        self.metrics['signal_value'] = Gauge(
            f'{self.prefix}_signal_value',
            'Current signal value',
            registry=registry
        )

        self.metrics['weights_norm'] = Gauge(
            f'{self.prefix}_weights_norm',
            'Norm of weights',
            ['channel'],  # plus или minus
            registry=registry
        )

        self.metrics['samples_per_second'] = Gauge(
            f'{self.prefix}_samples_per_second',
            'Training throughput (samples/sec)',
            registry=registry
        )

        # Histograms
        self.metrics['loss_distribution'] = Histogram(
            f'{self.prefix}_loss_distribution',
            'Distribution of loss values',
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=registry
        )

        self.metrics['train_duration'] = Histogram(
            f'{self.prefix}_train_duration_seconds',
            'Training duration in seconds',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=registry
        )

    def record_train_iteration(self, loss: float, duration: float,
                               num_samples: int = 1) -> None:
        """
        Запись итерации обучения.

        Args:
            loss: Значение функции потерь
            duration: Длительность итерации
            num_samples: Количество обработанных образцов
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.metrics['train_iterations_total'].inc()
        self.metrics['train_samples_total'].inc(num_samples)
        self.metrics['current_loss'].set(loss)
        self.metrics['loss_distribution'].observe(loss)
        self.metrics['train_duration'].observe(duration)

        if duration > 0:
            self.metrics['samples_per_second'].set(num_samples / duration)

    def record_signal(self, signal: float) -> None:
        """
        Запись значения сигнала.

        Args:
            signal: Значение сигнала
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.metrics['signal_value'].set(signal)

    def record_weights(self, w_plus_norm: float, w_minus_norm: float) -> None:
        """
        Запись норм весов.

        Args:
            w_plus_norm: Норма весов S⁺ канала
            w_minus_norm: Норма весов S⁻ канала
        """
        if not PROMETHEUS_AVAILABLE:
            return

        self.metrics['weights_norm'].labels(channel='plus').set(w_plus_norm)
        self.metrics['weights_norm'].labels(channel='minus').set(w_minus_norm)


class MetricsCollector:
    """Коллектор метрик с интеграцией Prometheus."""

    def __init__(self, core: KokaoCore, port: int = 8000,
                 enable_prometheus: bool = True, registry=None, auto_port: bool = True):
        """
        Инициализация коллектора.

        Args:
            core: Экземпляр KokaoCore
            port: Порт для Prometheus метрик
            enable_prometheus: Включить ли Prometheus экспорт
            registry: Реестр Prometheus (для тестов)
            auto_port: Автоматически искать свободный порт если занят (для тестов)
        """
        self.core = core
        self.port = port
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.metrics = KokaoMetrics(registry=registry)
        self.training_metrics = TrainingMetrics()
        self._start_time: Optional[float] = None
        self._actual_port: Optional[int] = None

        if self.enable_prometheus:
            # Пытаемся запустить сервер на указанном порту
            try:
                start_http_server(port)
                self._actual_port = port
                logger.info(f"Prometheus metrics server started on port {port}")
            except OSError as e:
                if auto_port:
                    # Пытаемся найти свободный порт
                    free_port = _find_free_port(port)
                    if free_port > 0:
                        start_http_server(free_port)
                        self._actual_port = free_port
                        logger.info(f"Port {port} busy, started Prometheus metrics server on port {free_port}")
                    else:
                        logger.error("Could not find free port for Prometheus metrics server")
                        self.enable_prometheus = False
                else:
                    logger.error(f"Could not start Prometheus metrics server on port {port}: {e}")
                    self.enable_prometheus = False

    def get_actual_port(self) -> Optional[int]:
        """Получить фактический порт сервера Prometheus."""
        return self._actual_port

    def start_training(self) -> None:
        """Начало тренировки."""
        self._start_time = time.time()

    def end_training(self) -> None:
        """Окончание тренировки."""
        if self._start_time:
            self.training_metrics.total_time = time.time() - self._start_time
            self._start_time = None

    def record_batch(self, loss: float, batch_size: int) -> None:
        """
        Запись батча обучения.

        Args:
            loss: Функция потерь
            batch_size: Размер батча
        """
        self.training_metrics.total_iterations += 1
        self.training_metrics.total_samples += batch_size
        self.training_metrics.last_loss = loss

        # Скользящее среднее
        n = self.training_metrics.total_iterations
        self.training_metrics.avg_loss = (
            (self.training_metrics.avg_loss * (n - 1) + loss) / n
        )

        # Запись в Prometheus
        if self.enable_prometheus:
            self.metrics.record_train_iteration(loss, 0.0, batch_size)

    def record_signal(self, x: torch.Tensor) -> float:
        """
        Запись сигнала с метриками.

        Args:
            x: Входной вектор

        Returns:
            Значение сигнала
        """
        signal = self.core.signal(x)

        if self.enable_prometheus:
            self.metrics.record_signal(signal)

            # Запись норм весов
            eff_w_plus, eff_w_minus = self.core._get_effective_weights()
            self.metrics.record_weights(
                eff_w_plus.norm().item(),
                eff_w_minus.norm().item()
            )

        return signal

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики."""
        stats = {
            'total_iterations': self.training_metrics.total_iterations,
            'total_samples': self.training_metrics.total_samples,
            'last_loss': self.training_metrics.last_loss,
            'avg_loss': self.training_metrics.avg_loss,
            'total_time': self.training_metrics.total_time
        }

        if self.training_metrics.total_time > 0:
            stats['samples_per_second'] = (
                self.training_metrics.total_samples / self.training_metrics.total_time
            )

        return stats

    def export_metrics(self, path: str) -> str:
        """
        Экспорт метрик в файл.

        Args:
            path: Путь для сохранения

        Returns:
            Путь к файлу
        """
        import json

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.get_stats(), f, indent=2)

        return str(output_path)


class GrafanaDashboard:
    """Генератор дашборда Grafana."""

    @staticmethod
    def generate_dashboard_json(port: int = 8000) -> Dict[str, Any]:
        """
        Генерация JSON дашборда для Grafana.

        Args:
            port: Порт Prometheus

        Returns:
            JSON дашборда
        """
        dashboard = {
            "dashboard": {
                "title": "Kokao Engine Metrics",
                "panels": [
                    {
                        "id": 1,
                        "title": "Training Loss",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "kokao_current_loss",
                                "legendFormat": "Loss"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Samples per Second",
                        "type": "gauge",
                        "targets": [
                            {
                                "expr": "kokao_samples_per_second",
                                "legendFormat": "Throughput"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Signal Value",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "kokao_signal_value",
                                "legendFormat": "Signal"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "Weights Norm",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "kokao_weights_norm{channel='plus'}",
                                "legendFormat": "W+ Norm"
                            },
                            {
                                "expr": "kokao_weights_norm{channel='minus'}",
                                "legendFormat": "W- Norm"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 6, "y": 8}
                    }
                ],
                "refresh": "5s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                }
            }
        }

        return dashboard

    @staticmethod
    def save_dashboard(path: str = "kokao_dashboard.json") -> str:
        """
        Сохранение дашборда в файл.

        Args:
            path: Путь для сохранения

        Returns:
            Путь к файлу
        """
        import json

        dashboard = GrafanaDashboard.generate_dashboard_json()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(dashboard, f, indent=2)

        return str(output_path)


def create_metrics_collector(core: KokaoCore,
                             port: int = 8000) -> MetricsCollector:
    """
    Создание коллектора метрик.

    Args:
        core: Экземпляр KokaoCore
        port: Порт для Prometheus

    Returns:
        MetricsCollector
    """
    return MetricsCollector(core, port=port)
