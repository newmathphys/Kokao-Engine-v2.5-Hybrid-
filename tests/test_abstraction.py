"""Тесты для модуля abstraction.py."""
import pytest
import torch
from kokao import KokaoCore, CoreConfig
from kokao.abstraction import AbstractionEngine


@pytest.fixture
def core():
    """Создание тестового ядра."""
    return KokaoCore(CoreConfig(input_dim=5))


@pytest.fixture
def engine(core):
    """Создание движка абстракции."""
    return AbstractionEngine(core)


def test_extract_prototype_mean(engine):
    """Тест извлечения прототипа (mean)."""
    examples = [torch.randn(5) for _ in range(10)]
    proto = engine.extract_prototype(examples, method="mean")
    assert proto.shape == (5,)


def test_extract_prototype_median(engine):
    """Тест извлечения прототипа (median)."""
    examples = [torch.randn(5) for _ in range(10)]
    proto = engine.extract_prototype(examples, method="median")
    assert proto.shape == (5,)


def test_extract_prototype_pca(engine):
    """Тест извлечения прототипа (pca)."""
    examples = [torch.randn(5) for _ in range(10)]
    proto = engine.extract_prototype(examples, method="pca")
    assert proto.shape == (5,)


def test_hierarchical_abstraction(engine):
    """Тест иерархической абстракции."""
    groups = [[torch.randn(5) for _ in range(5)] for _ in range(4)]
    protos = engine.hierarchical_abstraction(groups, levels=2)

    assert len(protos) == 2
    assert len(protos[0]) == 4
    assert len(protos[1]) == 2


def test_extract_prototype_empty(engine):
    """Тест с пустым списком примеров."""
    proto = engine.extract_prototype([])
    assert proto.shape == (5,)
    assert torch.allclose(proto, torch.zeros(5))


def test_extract_prototype_unknown_method(engine):
    """Тест ошибки при неизвестном методе."""
    examples = [torch.randn(5) for _ in range(3)]

    with pytest.raises(ValueError):
        engine.extract_prototype(examples, method="unknown")
