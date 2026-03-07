"""
Экспериментальные топологические методы.

Включают:
- Константу K = 1838.684 (отношение массы нейтрона к массе электрона)
- Проверку сигнала на фундаментальный диапазон [1/K, K]
- Нормализацию на единичную 3-сферу
- Обратную задачу с проекцией на сферу и проверкой диапазона
"""
import torch
import warnings
from typing import Optional

# Импортируем ядро (стандартное)
from kokao.core import KokaoCore

# Фундаментальная константа (отношение масс нейтрона и электрона)
K = 1838.684


def check_fundamental_range(signal: float, warn: bool = True) -> bool:
    """
    Проверяет, лежит ли сигнал в фундаментальном диапазоне [1/K, K].

    Args:
        signal: значение сигнала
        warn: если True и сигнал вне диапазона, выводит предупреждение

    Returns:
        True если сигнал в диапазоне, иначе False
    """
    if signal < 1 / K or signal > K:
        if warn:
            warnings.warn(
                f"Signal {signal:.3f} outside fundamental range "
                f"[1/{K:.1f}, {K:.1f}]",
                UserWarning
            )
        return False
    return True


def normalize_to_sphere(x: torch.Tensor) -> torch.Tensor:
    """
    Нормирует вектор на единичную 3-сферу (сохраняет сигнал из-за масштабной инвариантности).

    Args:
        x: входной вектор

    Returns:
        Нормированный вектор
    """
    return x / (torch.norm(x) + 1e-9)


class TopologicalInverse:
    """
    Экспериментальная обратная задача с топологической проверкой.
    Использует стандартную аналитическую проекцию, затем проверяет сигнал
    и при необходимости применяет нормализацию на сферу.
    """
    def __init__(
        self,
        core: KokaoCore,
        check_range: bool = True,
        warn_out_of_range: bool = True,
        normalize_to_sphere: bool = False
    ):
        self.core = core
        self.check_range = check_range
        self.warn_out_of_range = warn_out_of_range
        self.normalize_to_sphere = normalize_to_sphere

    def solve(
        self,
        target: float,
        x_init: Optional[torch.Tensor] = None,
        project_positive: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Решает обратную задачу, используя аналитическую проекцию,
        затем применяет дополнительные топологические преобразования.

        Args:
            target: целевой сигнал (>0)
            x_init: начальное приближение (если None, используется случайный вектор)
            project_positive: если True, обрезает отрицательные компоненты
            **kwargs: игнорируются (для совместимости)

        Returns:
            Вектор x, для которого S(x) ≈ target
        """
        # Получаем веса ядра
        w_plus, w_minus = self.core._get_effective_weights()

        # Вычисляем эффективный вектор
        v = w_plus - target * w_minus
        v_norm_sq = torch.dot(v, v) + 1e-9

        if x_init is None:
            x = torch.randn(
                self.core.config.input_dim,
                device=self.core.device,
                dtype=self.core.dtype
            )
        else:
            x = x_init.clone().detach().to(self.core.device, self.core.dtype)

        # Аналитическая проекция на гиперплоскость v·x = 0
        proj = x - (torch.dot(v, x) / v_norm_sq) * v

        if project_positive:
            proj.clamp_(min=0.0)

        # Проверка диапазона (опционально)
        if self.check_range:
            with torch.no_grad():
                s = self.core.signal(proj)
                check_fundamental_range(s, self.warn_out_of_range)

        # Нормализация на сферу (опционально)
        if self.normalize_to_sphere:
            proj = normalize_to_sphere(proj)

        return proj.detach()
