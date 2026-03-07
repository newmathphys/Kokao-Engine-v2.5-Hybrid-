"""
Модуль инференса через ONNX Runtime.
Основан на экспорте модели в ONNX и ускоренном инференсе.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from .core import KokaoCore

# Опциональные импорты
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    ort = None


class ONNXRuntimeCore:
    """Обёртка для инференса KokaoCore через ONNX Runtime."""

    def __init__(self, core: KokaoCore, optimize: bool = True):
        """
        Инициализация ONNX Runtime.

        Args:
            core: Экземпляр KokaoCore
            optimize: Применять ли оптимизации графа
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX modules not available. Install with: pip install onnx onnxruntime"
            )

        self.core = core
        self.session: Optional[ort.InferenceSession] = None
        self.optimize = optimize
        self.model_path: Optional[str] = None

    def export_onnx(self, path: str, opset_version: int = 14) -> str:
        """
        Экспорт модели в ONNX.

        Args:
            path: Путь для сохранения
            opset_version: Версия ONNX opset

        Returns:
            Путь к сохранённому файлу
        """
        self.core.eval()

        # Создаём фиктивный вход
        dummy_input = torch.randn(1, self.core.config.input_dim)

        # Экспортируем
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            self.core,
            dummy_input,
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=self.optimize,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # Валидация модели
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)

        self.model_path = path
        return path

    def load_session(self, path: Optional[str] = None,
                     providers: Optional[List[str]] = None) -> None:
        """
        Загрузка ONNX сессии.

        Args:
            path: Путь к ONNX модели (если None, использует последнюю экспортированную)
            providers: Список провайдеров (по умолчанию ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        if path is None:
            path = self.model_path

        if path is None:
            raise ValueError("No ONNX model path provided")

        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

        sess_options = ort.SessionOptions()
        if self.optimize:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            path,
            sess_options=sess_options,
            providers=providers
        )

    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        Инференс через ONNX Runtime.

        Args:
            x: Входные данные (может быть batch)

        Returns:
            Выходные данные (сигналы)
        """
        if self.session is None:
            raise RuntimeError("Session not loaded. Call load_session() first.")

        # Обеспечиваем правильную форму
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Запуск инференса
        ort_inputs = {self.session.get_inputs()[0].name: x.astype(np.float32)}
        ort_outputs = self.session.run(None, ort_inputs)

        return ort_outputs[0].flatten()

    def infer_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Пакетный инференс.

        Args:
            X: Входные данные (Batch, input_dim)

        Returns:
            Сигналы для каждого элемента батча
        """
        return self.infer(X)

    def benchmark(self, X: np.ndarray, num_runs: int = 100) -> Dict[str, float]:
        """
        Бенчмарк производительности.

        Args:
            X: Входные данные
            num_runs: Количество запусков

        Returns:
            Статистика времени выполнения
        """
        import time

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.infer(X)
            end = time.perf_counter()
            times.append(end - start)

        return {
            'mean': np.mean(times),
            'min': np.min(times),
            'max': np.max(times),
            'std': np.std(times),
            'throughput': len(X) / np.mean(times)  # образцов в секунду
        }

    def compare_with_pytorch(self, X: torch.Tensor) -> Dict[str, Any]:
        """
        Сравнение результатов с оригинальной PyTorch моделью.

        Args:
            X: Входные данные

        Returns:
            Статистика различий
        """
        # PyTorch inference
        self.core.eval()
        with torch.no_grad():
            pytorch_output = self.core.forward(X).cpu().numpy()

        # ONNX Runtime inference
        onnx_output = self.infer(X.cpu().numpy())

        # Сравнение
        diff = np.abs(pytorch_output - onnx_output)

        return {
            'max_diff': float(np.max(diff)),
            'mean_diff': float(np.mean(diff)),
            'pytorch_mean': float(np.mean(pytorch_output)),
            'onnx_mean': float(np.mean(onnx_output)),
            'match': np.allclose(pytorch_output, onnx_output, atol=1e-5)
        }


def export_and_benchmark(core: KokaoCore,
                         path: str = "model.onnx",
                         batch_size: int = 32) -> Dict[str, Any]:
    """
    Экспорт и бенчмарк модели.

    Args:
        core: Экземпляр KokaoCore
        path: Путь для сохранения ONNX
        batch_size: Размер батча для бенчмарка

    Returns:
        Результаты бенчмарка
    """
    # Экспорт
    onnx_core = ONNXRuntimeCore(core)
    onnx_core.export_onnx(path)

    # Загрузка сессии
    onnx_core.load_session()

    # Бенчмарк
    X = np.random.randn(batch_size, core.config.input_dim).astype(np.float32)
    benchmark_results = onnx_core.benchmark(X, num_runs=100)

    return {
        'model_path': path,
        'benchmark': benchmark_results
    }
