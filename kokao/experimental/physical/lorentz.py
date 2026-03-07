"""
Лоренц-фактор для релятивистских поправок.
"""
import torch


def lorentz_factor(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Вычисляет лоренц-фактор γ = 1 / √(1 - v²/c²).

    Args:
        x: входной вектор (интерпретируется как скорость)
        c: предельная скорость (по умолчанию 1.0 для нормированных единиц)

    Returns:
        Скалярный лоренц-фактор.
    """
    v_sq = torch.sum(x**2)
    beta_sq = v_sq / (c**2 + 1e-9)
    # Ограничиваем beta_sq < 1 для избежания комплексных чисел
    beta_sq = torch.clamp(beta_sq, max=1 - 1e-9)
    gamma = 1.0 / torch.sqrt(1 - beta_sq)
    return gamma


def lorentz_boost(x: torch.Tensor, v: torch.Tensor, c: float = 1.0):
    """
    Применяет лоренцево преобразование к вектору.
    
    Args:
        x: пространственно-временной вектор (t, x, y, z)
        v: скорость буста (3-вектор)
        c: скорость света
    
    Returns:
        Преобразованный вектор.
    """
    gamma = lorentz_factor(v, c)
    beta = v / c
    
    # Разделяем временную и пространственные компоненты
    t = x[..., 0:1]
    r = x[..., 1:]
    
    # Параллельная и перпендикулярная компоненты
    r_parallel = torch.sum(r * beta, dim=-1, keepdim=True) * beta / (torch.sum(beta**2) + 1e-9)
    r_perp = r - r_parallel
    
    # Преобразование
    t_new = gamma * (t + torch.sum(r * beta, dim=-1, keepdim=True) / c)
    r_parallel_new = gamma * (r_parallel + beta * t * c)
    
    return torch.cat([t_new, r_perp + r_parallel_new], dim=-1)
