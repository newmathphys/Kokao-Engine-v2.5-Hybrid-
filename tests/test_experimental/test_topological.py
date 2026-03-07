"""
Тесты для экспериментальных топологических методов.
"""
import pytest
import torch
from kokao.core import KokaoCore, CoreConfig
from kokao.experimental.topological import (
    K,
    check_fundamental_range,
    normalize_to_sphere,
    TopologicalInverse
)


@pytest.mark.experimental
def test_check_fundamental_range():
    """Проверка функции проверки диапазона."""
    # K = 1838.684, диапазон [1/K, K] ≈ [0.00054, 1838.684]
    assert check_fundamental_range(1.0, warn=False) is True
    assert check_fundamental_range(1000.0, warn=False) is True  # внутри диапазона
    assert check_fundamental_range(0.0001, warn=False) is False  # ниже 1/K
    assert check_fundamental_range(2000.0, warn=False) is False  # выше K
    with pytest.warns(UserWarning):
        check_fundamental_range(5000.0, warn=True)


@pytest.mark.experimental
def test_normalize_to_sphere():
    """Проверка нормализации на сферу."""
    x = torch.randn(10)
    x_norm = normalize_to_sphere(x)
    assert torch.allclose(x_norm.norm(), torch.tensor(1.0), atol=1e-6)


@pytest.mark.experimental
def test_topological_inverse_solve():
    """Проверка работы экспериментальной обратной задачи."""
    config = CoreConfig(input_dim=5, seed=42)
    core = KokaoCore(config)
    inv = TopologicalInverse(core, check_range=False, normalize_to_sphere=False)

    target = 0.8
    x = inv.solve(target)

    s = core.signal(x)
    assert abs(s - target) < 1e-5, f"Error: {abs(s - target)}"


@pytest.mark.experimental
def test_topological_inverse_check_range():
    """Проверка, что предупреждение о внедиапазонном сигнале срабатывает."""
    from kokao.experimental.topological import check_fundamental_range
    
    # Тестируем напрямую функцию проверки диапазона
    # Сигнал 5000.0 > K (1838.684), должен вызвать предупреждение
    with pytest.warns(UserWarning, match="outside fundamental range"):
        check_fundamental_range(5000.0, warn=True)
    
    # Сигнал 0.0001 < 1/K (~0.00054), должен вызвать предупреждение
    with pytest.warns(UserWarning, match="outside fundamental range"):
        check_fundamental_range(0.0001, warn=True)
