"""
Обратная задача с проекцией и проверкой по фундаментальному диапазону.
"""
import torch
import warnings
from typing import Optional

from kokao.core import KokaoCore

from .constants import K


class PhysicalInverse:
    """
    Обратная задача с физическими проверками.
    
    Args:
        core: ядро KokaoCore
        check_range: проверять фундаментальный диапазон [1/K, K]
        warn: выводить предупреждения при выходе за диапазон
    """
    
    def __init__(
        self,
        core: KokaoCore,
        check_range: bool = True,
        warn: bool = True
    ):
        self.core = core
        self.check_range = check_range
        self.warn = warn

    def solve(
        self,
        target: float,
        x_init: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Решает обратную задачу с аналитической проекцией.
        
        Args:
            target: целевой сигнал (>0)
            x_init: начальное приближение (если None, используется случайный вектор)
        
        Returns:
            Вектор x, для которого S(x) ≈ target
        """
        w_plus, w_minus = self.core._get_effective_weights()
        v = w_plus - target * w_minus
        v_norm_sq = torch.dot(v, v) + 1e-12

        if x_init is None:
            x = torch.randn_like(w_plus)
        else:
            x = x_init.clone().detach().to(self.core.device, self.core.dtype)

        # аналитическая проекция на гиперплоскость v·x = 0
        proj = x - (torch.dot(v, x) / v_norm_sq) * v

        if self.check_range:
            with torch.no_grad():
                s = self.core.signal(proj)
                if s < 1/K or s > K:
                    msg = f"Signal {s:.3f} outside fundamental range [1/{K:.1f}, {K:.1f}]"
                    if self.warn:
                        warnings.warn(msg, UserWarning)
                    else:
                        print(msg)
        
        return proj.detach()
