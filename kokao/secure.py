"""Модуль безопасности для проверки входных данных в Kokao Engine."""
import torch
import functools
import logging
from typing import Callable, Any
from .core import KokaoCore

logger = logging.getLogger(__name__)


def validate_tensor_input(func: Callable) -> Callable:
    """
    Декоратор для проверки входных тензоров на корректность.
    
    Проверяет:
    - тип данных (torch.Tensor)
    - отсутствие NaN/Inf
    - соответствие размерности
    """
    @functools.wraps(func)
    def wrapper(self, x: torch.Tensor, *args, **kwargs):
        # Проверяем тип
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(x)}")
        
        # Проверяем на NaN
        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values")
        
        # Проверяем на Inf
        if torch.isinf(x).any():
            raise ValueError("Input tensor contains Inf values")
        
        # Проверяем размерность, если у объекта есть атрибут input_dim
        if hasattr(self, 'config') and hasattr(self.config, 'input_dim'):
            expected_dim = self.config.input_dim
            if x.ndim == 1 and x.shape[0] != expected_dim:
                raise ValueError(f"Expected input dimension {expected_dim}, got {x.shape[0]}")
            elif x.ndim > 1 and x.shape[-1] != expected_dim:
                raise ValueError(f"Expected input last dimension {expected_dim}, got {x.shape[-1]}")
        
        return func(self, x, *args, **kwargs)
    
    return wrapper


class SecureKokao:
    """
    Обёртка безопасности для KokaoCore.
    
    Проверяет все входные тензоры перед передачей в ядро.
    """
    
    def __init__(self, core: KokaoCore):
        """
        Инициализация безопасной обёртки.
        
        Args:
            core: Экземпляр KokaoCore
        """
        self._core = core
    
    @validate_tensor_input
    def signal(self, x: torch.Tensor) -> float:
        """
        Вычисление сигнала с проверкой валидности входа.
        
        Args:
            x: Входной тензор
            
        Returns:
            Скалярный сигнал
        """
        return self._core.signal(x)
    
    @validate_tensor_input
    def train(self, x: torch.Tensor, target: float, lr: float = 0.01,
              mode: str = "gradient") -> float:
        """
        Обучение с проверкой валидности входа.
        
        Args:
            x: Входной тензор
            target: Целевое значение
            lr: Скорость обучения
            mode: Режим обучения ('gradient' или 'kosyakov')
            
        Returns:
            Значение функции потерь до обновления
        """
        return self._core.train(x, target, lr, mode)
    
    @validate_tensor_input
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход с проверкой валидности входа.
        
        Args:
            x: Входной тензор
            
        Returns:
            Выходной тензор
        """
        return self._core.forward(x)
    
    def forget(self, rate: float = 0.1, lambda_l1: float = 0.0) -> None:
        """
        Забывание без проверки тензоров (требует только числовые параметры).
        
        Args:
            rate: Скорость забывания
            lambda_l1: Коэффициент L1-регуляризации
        """
        return self._core.forget(rate, lambda_l1)
    
    def to_inverse_problem(self):
        """
        Создание обратной задачи из текущего состояния ядра.
        
        Returns:
            Экземпляр InverseProblem
        """
        return self._core.to_inverse_problem()
    
    def __getattr__(self, name: str) -> Any:
        """
        Проксируем все остальные атрибуты к внутреннему ядру.
        
        Args:
            name: Имя атрибута
            
        Returns:
            Значение атрибута из внутреннего ядра
        """
        attr = getattr(self._core, name)
        if callable(attr):
            # Если атрибут - функция, не требующая тензоров, просто возвращаем её
            return attr
        else:
            # Если атрибут - свойство, возвращаем его значение
            return attr