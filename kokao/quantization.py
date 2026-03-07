"""Модуль квантования моделей для оптимизации размера и скорости."""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import logging
from pathlib import Path
import json

from .core import KokaoCore
from .core_base import CoreConfig

logger = logging.getLogger(__name__)


class QuantizationConfig:
    """Конфигурация квантования."""

    def __init__(self, bits: int = 8, method: str = 'dynamic',
                 per_channel: bool = False, symmetric: bool = True):
        """
        Инициализация конфигурации.

        Args:
            bits: Количество бит (4, 8)
            method: Метод квантования ('dynamic', 'static', 'float16')
            per_channel: Квантование по каналам
            symmetric: Симметричное квантование
        """
        self.bits = bits
        self.method = method
        self.per_channel = per_channel
        self.symmetric = symmetric

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return {
            'bits': self.bits,
            'method': self.method,
            'per_channel': self.per_channel,
            'symmetric': self.symmetric
        }


class QuantizedKokaoCore:
    """
    Квантованная версия KokaoCore.
    """

    def __init__(self, core: KokaoCore, config: QuantizationConfig):
        """
        Инициализация квантованного ядра.

        Args:
            core: Исходное ядро
            config: Конфигурация квантования
        """
        self.original_core = core
        self.config = config
        self.is_quantized = False

        # Квантованные веса
        self.quantized_w_plus = None
        self.quantized_w_minus = None

        # Параметры квантования
        self.scale_plus = None
        self.zero_point_plus = None
        self.scale_minus = None
        self.zero_point_minus = None

    def quantize(self) -> 'QuantizedKokaoCore':
        """Выполнение квантования."""
        if self.config.method == 'float16':
            self._quantize_float16()
        elif self.config.bits == 8:
            self._quantize_int8()
        elif self.config.bits == 4:
            self._quantize_int4()
        else:
            raise ValueError(f"Unsupported bits: {self.config.bits}")

        self.is_quantized = True
        logger.info(f"Quantized KokaoCore with {self.config.bits}-bit {self.config.method}")
        return self

    def _quantize_float16(self) -> None:
        """Квантование в float16."""
        with torch.no_grad():
            eff_w_plus, eff_w_minus = self.original_core._get_effective_weights()
            self.quantized_w_plus = eff_w_plus.half()
            self.quantized_w_minus = eff_w_minus.half()

    def _quantize_int8(self) -> None:
        """Квантование в int8."""
        with torch.no_grad():
            eff_w_plus, eff_w_minus = self.original_core._get_effective_weights()

            # Вычисление параметров квантования
            self.scale_plus, self.zero_point_plus = self._compute_quant_params(
                eff_w_plus, qmin=-128, qmax=127
            )
            self.scale_minus, self.zero_point_minus = self._compute_quant_params(
                eff_w_minus, qmin=-128, qmax=127
            )

            # Квантование
            self.quantized_w_plus = self._quantize_tensor(
                eff_w_plus, self.scale_plus, self.zero_point_plus, -128, 127
            )
            self.quantized_w_minus = self._quantize_tensor(
                eff_w_minus, self.scale_minus, self.zero_point_minus, -128, 127
            )

    def _quantize_int4(self) -> None:
        """Квантование в int4."""
        with torch.no_grad():
            eff_w_plus, eff_w_minus = self.original_core._get_effective_weights()

            # Вычисление параметров квантования
            self.scale_plus, self.zero_point_plus = self._compute_quant_params(
                eff_w_plus, qmin=-8, qmax=7
            )
            self.scale_minus, self.zero_point_minus = self._compute_quant_params(
                eff_w_minus, qmin=-8, qmax=7
            )

            # Квантование
            self.quantized_w_plus = self._quantize_tensor(
                eff_w_plus, self.scale_plus, self.zero_point_plus, -8, 7
            )
            self.quantized_w_minus = self._quantize_tensor(
                eff_w_minus, self.scale_minus, self.zero_point_minus, -8, 7
            )

    def _compute_quant_params(self, tensor: torch.Tensor,
                               qmin: int, qmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисление параметров квантования.

        Args:
            tensor: Тензор для квантования
            qmin: Минимальное квантованное значение
            qmax: Максимальное квантованное значение

        Returns:
            (scale, zero_point)
        """
        if self.config.symmetric:
            # Симметричное квантование
            max_abs = tensor.abs().max()
            scale = max_abs / max(qmax, 1)
            zero_point = torch.zeros_like(scale)
        else:
            # Асимметричное квантование
            min_val = tensor.min()
            max_val = tensor.max()

            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale

        return scale, zero_point

    def _quantize_tensor(self, tensor: torch.Tensor, scale: torch.Tensor,
                          zero_point: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        """
        Квантование тензора.

        Args:
            tensor: Тензор для квантования
            scale: Масштаб
            zero_point: Нулевая точка
            qmin: Минимальное значение
            qmax: Максимальное значение

        Returns:
            Квантованный тензор
        """
        # Квантование
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)

        return quantized.to(torch.int8 if qmax <= 127 else torch.int32)

    def dequantize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Деквантование весов.

        Returns:
            (w_plus, w_minus) деквантованные веса
        """
        if not self.is_quantized:
            return self.original_core._get_effective_weights()

        if self.config.method == 'float16':
            return self.quantized_w_plus.float(), self.quantized_w_minus.float()

        # Деквантование
        w_plus = (self.quantized_w_plus - self.zero_point_plus) * self.scale_plus
        w_minus = (self.quantized_w_minus - self.zero_point_minus) * self.scale_minus

        return w_plus, w_minus

    def signal(self, x: torch.Tensor) -> float:
        """
        Вычисление сигнала с квантованными весами.

        Args:
            x: Входной вектор

        Returns:
            Сигнал
        """
        w_plus, w_minus = self.dequantize()

        s_plus = torch.dot(x, w_plus)
        s_minus = torch.dot(x, w_minus)

        epsilon = 1e-6
        s_abs = s_minus.abs().clamp(min=epsilon)
        s = (s_plus / s_abs * torch.sign(s_minus)).item()

        return s

    def get_compression_ratio(self) -> float:
        """Получение коэффициента сжатия."""
        if not self.is_quantized:
            return 1.0

        original_bits = 32  # float32
        compressed_bits = self.config.bits

        return original_bits / compressed_bits

    def get_size_reduction(self) -> float:
        """Получение процента уменьшения размера."""
        ratio = self.get_compression_ratio()
        return (1 - 1 / ratio) * 100

    def save(self, path: str) -> None:
        """Сохранение квантованной модели."""
        state = {
            'config': self.config.to_dict(),
            'quantized_w_plus': self.quantized_w_plus.cpu().tolist() if self.quantized_w_plus is not None else None,
            'quantized_w_minus': self.quantized_w_minus.cpu().tolist() if self.quantized_w_minus is not None else None,
            'scale_plus': self.scale_plus.cpu().tolist() if self.scale_plus is not None else None,
            'zero_point_plus': self.zero_point_plus.cpu().tolist() if self.zero_point_plus is not None else None,
            'scale_minus': self.scale_minus.cpu().tolist() if self.scale_minus is not None else None,
            'zero_point_minus': self.zero_point_minus.cpu().tolist() if self.zero_point_minus is not None else None,
            'original_config': self.original_core.config.dict()
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'QuantizedKokaoCore':
        """Загрузка квантованной модели."""
        with open(path) as f:
            state = json.load(f)

        config = QuantizationConfig(**state['config'])
        core_config = CoreConfig(**state['original_config'])
        core = KokaoCore(core_config)

        quantized_core = cls(core, config)

        if state['quantized_w_plus'] is not None:
            quantized_core.quantized_w_plus = torch.tensor(state['quantized_w_plus'])
            quantized_core.quantized_w_minus = torch.tensor(state['quantized_w_minus'])

            if state['scale_plus'] is not None:
                quantized_core.scale_plus = torch.tensor(state['scale_plus'])
                quantized_core.zero_point_plus = torch.tensor(state['zero_point_plus'])
                quantized_core.scale_minus = torch.tensor(state['scale_minus'])
                quantized_core.zero_point_minus = torch.tensor(state['zero_point_minus'])

            quantized_core.is_quantized = True

        return quantized_core


def quantize_model(core: KokaoCore, bits: int = 8,
                   method: str = 'dynamic') -> QuantizedKokaoCore:
    """
    Быстрое квантование модели.

    Args:
        core: Модель для квантования
        bits: Количество бит
        method: Метод квантования

    Returns:
        Квантованная модель
    """
    config = QuantizationConfig(bits=bits, method=method)
    quantized = QuantizedKokaoCore(core, config)
    return quantized.quantize()


def benchmark_quantization(core: KokaoCore, test_inputs: Optional[List[torch.Tensor]] = None
                           ) -> Dict[str, Any]:
    """
    Бенчмарк квантования.

    Args:
        core: Исходная модель
        test_inputs: Тестовые входы

    Returns:
        Результаты бенчмарка
    """
    import time

    if test_inputs is None:
        test_inputs = [torch.randn(core.config.input_dim) for _ in range(100)]

    results = {}

    # Бенчмарк оригинальной модели
    start = time.time()
    for x in test_inputs:
        core.signal(x)
    original_time = time.time() - start

    # Бенчмарк float16
    q16 = quantize_model(core, bits=16, method='float16')
    start = time.time()
    for x in test_inputs:
        q16.signal(x)
    float16_time = time.time() - start

    # Бенчмарк int8
    q8 = quantize_model(core, bits=8)
    start = time.time()
    for x in test_inputs:
        q8.signal(x)
    int8_time = time.time() - start

    # Вычисление ошибок
    errors_16 = []
    errors_8 = []

    for x in test_inputs[:10]:
        orig = core.signal(x)
        q16_sig = q16.signal(x)
        q8_sig = q8.signal(x)

        errors_16.append(abs(orig - q16_sig))
        errors_8.append(abs(orig - q8_sig))

    results = {
        'original_time_ms': original_time / len(test_inputs) * 1000,
        'float16_time_ms': float16_time / len(test_inputs) * 1000,
        'int8_time_ms': int8_time / len(test_inputs) * 1000,
        'float16_speedup': original_time / float16_time,
        'int8_speedup': original_time / int8_time,
        'float16_avg_error': np.mean(errors_16),
        'int8_avg_error': np.mean(errors_8),
        'float16_compression': q16.get_compression_ratio(),
        'int8_compression': q8.get_compression_ratio(),
        'float16_size_reduction': q16.get_size_reduction(),
        'int8_size_reduction': q8.get_size_reduction()
    }

    return results
