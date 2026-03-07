"""
Модуль анализа устойчивости интуитивной системы к шуму.
Основан на книге Косякова Ю.Б., п.1.6.3.
"""

import torch
from .core import KokaoCore


class RobustnessAnalyzer:
    """Анализатор устойчивости сигнала к зашумлению входных данных."""

    def __init__(self, core: KokaoCore):
        """
        Инициализация анализатора.

        Args:
            core: Экземпляр KokaoCore для анализа
        """
        self.core = core

    def signal_with_noise(self, x: torch.Tensor, noise_level: float = 0.1):
        """
        Вычисляет сигнал для чистого и зашумлённого вектора.

        Args:
            x: Входной вектор
            noise_level: Уровень шума (стандартное отклонение)

        Returns:
            Кортеж (сигнал чистого, сигнал зашумлённого)
        """
        s_clean = self.core.signal(x)
        noise = torch.randn_like(x) * noise_level
        s_noisy = self.core.signal(x + noise)
        return s_clean, s_noisy

    def noise_tolerance_threshold(
        self,
        x: torch.Tensor,
        max_deviation: float = 0.1,
        max_noise: float = 10.0,
        step: float = 0.005
    ) -> float:
        """
        Находит уровень шума, при котором отклонение сигнала превышает max_deviation.

        Args:
            x: Входной вектор
            max_deviation: Максимально допустимое отклонение сигнала
            max_noise: Максимальный уровень шума для поиска
            step: Шаг увеличения шума

        Returns:
            Пороговый уровень шума
        """
        s_target = self.core.signal(x)
        noise_level = 0.0
        while noise_level <= max_noise:
            _, s_noisy = self.signal_with_noise(x, noise_level)
            if abs(s_noisy - s_target) > max_deviation:
                return noise_level
            noise_level += step
        return max_noise

    def feature_snr(self, x: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
        """
        Оценивает отношение сигнал/шум для каждого признака.

        Args:
            x: Входной вектор
            noise_level: Уровень шума

        Returns:
            Тензор отношений сигнал/шум для каждого признака
        """
        return torch.abs(x) / (noise_level + 1e-8)

    def feature_importance_for_stability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Важность признаков для устойчивости = (|w⁺|+|w⁻|) * |x|.

        Args:
            x: Входной вектор

        Returns:
            Тензор важности признаков
        """
        w_p, w_m = self.core._get_effective_weights()
        weight_magnitude = w_p + w_m
        return weight_magnitude * torch.abs(x)
