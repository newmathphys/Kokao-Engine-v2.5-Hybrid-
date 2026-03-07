"""Тесты для модуля fantasy.py."""
import pytest
import torch
from kokao import KokaoCore, CoreConfig
from kokao.fantasy import FantasyEngine


@pytest.fixture
def core():
    """Создание тестового ядра."""
    return KokaoCore(CoreConfig(input_dim=5))


@pytest.fixture
def engine(core):
    """Создание движка фантазирования."""
    return FantasyEngine(core)


def test_add_etalon(engine):
    """Тест добавления эталона."""
    x = torch.randn(5)
    engine.add_etalon(x)
    assert len(engine.etalon_pool) == 1


def test_combine_concepts(engine):
    """Тест комбинирования концепций."""
    a = torch.randn(5)
    b = torch.randn(5)
    hybrid = engine.combine_concepts(a, b, alpha=0.3)
    assert hybrid.shape == (5,)


def test_random_fantasy(engine):
    """Тест случайной фантазии."""
    for _ in range(3):
        engine.add_etalon(torch.randn(5))

    fantasy = engine.random_fantasy(num_concepts=2)
    assert fantasy.shape == (5,)


def test_combine_concepts_alpha_values(engine):
    """Тест различных значений alpha."""
    a = torch.randn(5)
    b = torch.randn(5)

    hybrid_0 = engine.combine_concepts(a, b, alpha=0.0)
    hybrid_1 = engine.combine_concepts(a, b, alpha=1.0)
    hybrid_05 = engine.combine_concepts(a, b, alpha=0.5)

    assert hybrid_0.shape == (5,)
    assert hybrid_1.shape == (5,)
    assert hybrid_05.shape == (5,)


def test_random_fantasy_insufficient_etagons(engine):
    """Тест ошибки при недостатке эталонов."""
    engine.add_etalon(torch.randn(5))

    with pytest.raises(ValueError):
        engine.random_fantasy(num_concepts=3)
