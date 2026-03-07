# kokao/core_base.py
"""
Базовые классы и конфигурация для KokaoCore.
"""
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing_extensions import Literal


class CoreConfig(BaseModel):
    """Конфигурация ядра KokaoCore."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='ignore')

    input_dim: int = Field(..., gt=0, description="Размерность входного вектора")
    device: str = Field("cpu", pattern="^(cpu|cuda|mps)$", description="Устройство для тензоров")
    dtype: Literal["float32", "float64"] = "float32"
    target_sum: float = Field(100.0, gt=0, description="Целевая сумма модулей весов (ресурс)")
    max_history: int = Field(100, ge=0, description="Максимальный размер истории весов")
    use_log_domain: bool = False
    seed: Optional[int] = None  # Seed для воспроизводимости

    @field_validator("device")
    @classmethod
    def check_device_available(cls, v: str) -> str:
        if v == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA не доступна, выберите 'cpu'")
        if v == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS не доступна, выберите 'cpu'")
        return v


class KokaoCoreBase(ABC):
    """Абстрактный базовый класс для всех реализаций ядра."""

    config: CoreConfig

    @abstractmethod
    def signal(self, x: torch.Tensor) -> float:
        """Вычисляет скалярный сигнал S для одного входного вектора."""

    @abstractmethod
    def train(self, x: torch.Tensor, target: float, lr: float = 0.01, mode: str = "gradient") -> float:
        """
        Один шаг обучения. Возвращает значение функции потерь до обновления.
        Параметр mode определяет способ обновления весов (зависит от реализации).
        """

    @abstractmethod
    def forget(self, rate: float = 0.1, lambda_l1: float = 0.0) -> None:
        """Забывание (экспоненциальный спад) и L1-регуляризация."""
