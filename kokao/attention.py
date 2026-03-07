"""
Модуль внимания, реализующий обратную проекцию от коры к таламусу.
Основан на книге Косякова Ю.Б., п.2.3.4.
"""

import torch
from typing import List
from .core import KokaoCore


class ThalamicAttention:
    """Механизм внимания через модуляцию входного сигнала."""

    def __init__(self, core: KokaoCore):
        """
        Инициализация механизма внимания.

        Args:
            core: Экземпляр KokaoCore
        """
        self.core = core

    def _importance(self) -> torch.Tensor:
        """
        Вычисляет важность признаков на основе текущих весов ядра.

        Returns:
            Тензор важности признаков в диапазоне (0, 1)
        """
        w_p, w_m = self.core._get_effective_weights()
        return torch.sigmoid(w_p - w_m)

    def modulate_input(
        self,
        x: torch.Tensor,
        context_gate: float = 1.0
    ) -> torch.Tensor:
        """
        Модулирует входной сигнал: усиливает важные признаки, подавляет неважные.

        Args:
            x: Входной вектор
            context_gate: Сила модуляции (0 = нет модуляции)

        Returns:
            Модулированный входной вектор
        """
        imp = self._importance()
        factor = 1.0 + context_gate * (imp - 0.5) * 2.0
        return x * factor

    def top_down_focus(
        self,
        x: torch.Tensor,
        target_indices: List[int],
        strength: float = 1.0
    ) -> float:
        """
        Принудительное внимание на заданные признаки (имитация команды сверху).

        Args:
            x: Входной вектор
            target_indices: Индексы признаков для усиления
            strength: Сила усиления

        Returns:
            Сигнал с усиленными весами для выбранных признаков
        """
        w_p, w_m = self.core._get_effective_weights()
        mask = torch.zeros_like(w_p)
        mask[target_indices] = 1.0
        w_p_boosted = w_p + strength * mask
        w_m_boosted = w_m + strength * mask

        # Временно изменяем веса ядра
        original_w_plus = self.core.w_plus.clone()
        original_w_minus = self.core.w_minus.clone()

        self.core.w_plus = torch.nn.Parameter(w_p_boosted, requires_grad=False)
        self.core.w_minus = torch.nn.Parameter(w_m_boosted, requires_grad=False)

        s_focused = self.core.signal(x)

        # Восстанавливаем оригинальные веса
        self.core.w_plus = torch.nn.Parameter(original_w_plus, requires_grad=True)
        self.core.w_minus = torch.nn.Parameter(original_w_minus, requires_grad=True)

        return s_focused
