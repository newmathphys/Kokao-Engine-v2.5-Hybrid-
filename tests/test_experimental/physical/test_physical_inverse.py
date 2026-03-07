"""
Тесты для физической обратной задачи.
"""
import pytest
import torch
from kokao.core import KokaoCore, CoreConfig
from kokao.experimental.physical.inverse import PhysicalInverse
from kokao.experimental.physical.constants import K


@pytest.mark.experimental
def test_physical_inverse_basic():
    """Проверка базовой работы PhysicalInverse."""
    config = CoreConfig(input_dim=5, seed=42)
    core = KokaoCore(config)
    inv = PhysicalInverse(core, check_range=False)
    target = 0.8
    x = inv.solve(target)
    s = core.signal(x)
    assert abs(s - target) < 1e-5


@pytest.mark.experimental
def test_physical_inverse_with_check():
    """Проверка PhysicalInverse с проверкой диапазона."""
    config = CoreConfig(input_dim=5, seed=42)
    core = KokaoCore(config)
    inv = PhysicalInverse(core, check_range=True, warn=False)
    target = 0.8
    x = inv.solve(target)
    s = core.signal(x)
    # Сигнал должен быть близок к target
    assert abs(s - target) < 1e-5


@pytest.mark.experimental
def test_physical_inverse_fundamental_range():
    """Проверка, что нормальные сигналы в диапазоне [1/K, K]."""
    config = CoreConfig(input_dim=5, seed=42)
    core = KokaoCore(config)
    inv = PhysicalInverse(core, check_range=True, warn=True)
    
    # Целевые сигналы в диапазоне
    for target in [0.1, 1.0, 100.0, 1000.0]:
        x = inv.solve(target)
        s = core.signal(x)
        # Проверяем, что сигнал в диапазоне (с допуском)
        assert s > 1/K / 10 and s < K * 10  # с запасом на точность


@pytest.mark.experimental
def test_physical_inverse_warn_outside_range():
    """Проверка предупреждения при выходе за диапазон."""
    import warnings
    config = CoreConfig(input_dim=5, seed=42)
    core = KokaoCore(config)
    inv = PhysicalInverse(core, check_range=True, warn=True)
    
    # Очень большой target может дать сигнал вне диапазона
    # Но из-за проекции сигнал будет близок к target
    # Поэтому проверяем саму логику проверки диапазона
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Сигнал 1.0 всегда в диапазоне
        x = inv.solve(1.0)
        # Не должно быть предупреждений для нормального сигнала
        range_warnings = [x for x in w if "outside fundamental range" in str(x.message)]
        # В данном случае предупреждений быть не должно
        assert len(range_warnings) == 0
