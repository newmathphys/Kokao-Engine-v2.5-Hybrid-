"""Тесты для модуля attention.py."""
import pytest
import torch
from kokao import KokaoCore, CoreConfig
from kokao.attention import ThalamicAttention


@pytest.fixture
def core():
    """Создание тестового ядра."""
    return KokaoCore(CoreConfig(input_dim=5))


@pytest.fixture
def attention(core):
    """Создание механизма внимания."""
    return ThalamicAttention(core)


def test_modulate_input(attention):
    """Тест модуляции входного сигнала."""
    x = torch.randn(5)
    x_mod = attention.modulate_input(x)
    assert x_mod.shape == x.shape
    assert not torch.allclose(x_mod, x)


def test_top_down_focus(attention):
    """Тест принудительного внимания."""
    x = torch.randn(5)
    s_normal = attention.core.signal(x)
    s_focused = attention.top_down_focus(x, target_indices=[0, 2], strength=10.0)
    assert s_focused != s_normal


def test_importance(attention):
    """Тест вычисления важности признаков."""
    imp = attention._importance()
    assert imp.shape == (5,)
    assert torch.all(imp >= 0) and torch.all(imp <= 1)


def test_modulate_input_no_context(attention):
    """Тест модуляции без контекста (context_gate=0)."""
    x = torch.randn(5)
    x_mod = attention.modulate_input(x, context_gate=0.0)
    assert torch.allclose(x_mod, x)
