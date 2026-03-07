"""
Тесты для физического ядра PhysicalCore.
"""
import pytest
import torch
from kokao.core import CoreConfig
from kokao.experimental.physical.core import PhysicalCore


@pytest.mark.experimental
def test_physical_core_basic():
    """Проверка базовой работы PhysicalCore."""
    config = CoreConfig(input_dim=10, seed=42)
    core = PhysicalCore(config)
    x = torch.randn(10)
    s = core.signal(x)
    assert isinstance(s, float)


@pytest.mark.experimental
def test_physical_core_isospin():
    """Проверка PhysicalCore с изоспиновым режимом."""
    config = CoreConfig(input_dim=10, seed=42)
    core = PhysicalCore(config, isospin_mode='+3')
    x = torch.randn(10)
    core.training = False  # переключаем в режим инференса для изоспина
    s = core.signal(x)
    assert isinstance(s, float)


@pytest.mark.experimental
def test_physical_core_lorentz():
    """Проверка PhysicalCore с лоренц-фактором."""
    config = CoreConfig(input_dim=10, seed=42)
    core = PhysicalCore(config, use_lorentz=True, lorentz_c=1.0)
    x = torch.randn(10) * 0.5  # малые скорости
    s = core.signal(x)
    assert isinstance(s, float)


@pytest.mark.experimental
def test_physical_core_solitonic():
    """Проверка PhysicalCore с солитонной активацией."""
    config = CoreConfig(input_dim=10, seed=42)
    core = PhysicalCore(config, use_solitonic=True)
    x = torch.randn(10)
    s = core.signal(x)
    # Солитонная активация возвращает значение в [-1, 1]
    assert -1 <= s <= 1


@pytest.mark.experimental
def test_physical_core_train():
    """Проверка обучения PhysicalCore."""
    config = CoreConfig(input_dim=10, seed=42)
    core = PhysicalCore(config)  # без изоспина для обучения
    x = torch.randn(10)
    loss = core.train(x, target=0.8, lr=0.01)
    assert isinstance(loss, float)
    assert loss >= 0
