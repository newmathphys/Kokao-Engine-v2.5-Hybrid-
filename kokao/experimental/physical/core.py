"""
Экспериментальное ядро с физическими расширениями.
Наследует KokaoCore и добавляет опции:
- лоренц-фактор
- изоспиновое квантование весов
- солитонную активацию
"""
import torch
from typing import Optional, Literal

from kokao.core import KokaoCore
from kokao.core_base import CoreConfig

from .lorentz import lorentz_factor
from .isospin import isospin_projection
from .solitonic import solitonic_activation


class PhysicalCore(KokaoCore):
    """
    Расширенное ядро с физическими эффектами.
    
    Args:
        config: конфигурация ядра
        use_lorentz: использовать лоренц-фактор
        isospin_mode: режим изоспина ('+3', '+4' или None)
        use_solitonic: использовать солитонную активацию
        lorentz_c: параметр скорости света для лоренц-фактора
    """
    
    def __init__(
        self,
        config: CoreConfig,
        use_lorentz: bool = False,
        isospin_mode: Optional[Literal['+3', '+4']] = None,
        use_solitonic: bool = False,
        lorentz_c: float = 1.0,
        **kwargs
    ):
        # Инициализируем атрибуты до вызова super().__init__
        # так как _normalize() может вызвать _get_effective_weights()
        self.use_lorentz = use_lorentz
        self.isospin_mode = isospin_mode
        self.use_solitonic = use_solitonic
        self.lorentz_c = lorentz_c
        
        super().__init__(config, **kwargs)

    def _get_effective_weights(self):
        """Может применять изоспиновую проекцию к весам."""
        w_plus, w_minus = super()._get_effective_weights()
        # Изоспиновая проекция применяется только в режиме инференса
        # чтобы не ломать градиенты при обучении
        if self.isospin_mode and not self.training:
            w_plus = isospin_projection(w_plus, self.isospin_mode)
            w_minus = isospin_projection(w_minus, self.isospin_mode)
        return w_plus, w_minus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет сигнал с учётом физических расширений.

        Args:
            x: входной вектор

        Returns:
            Сигнал с учётом выбранных эффектов.
        """
        if self.use_solitonic:
            w_plus, w_minus = self._get_effective_weights()
            return solitonic_activation(x, w_plus, w_minus)
        else:
            s = super().forward(x)
            if self.use_lorentz:
                gamma = lorentz_factor(x, self.lorentz_c)
                s = s * gamma
            return s
