"""Тесты для модуля guidance.py."""
import pytest
import torch
from kokao import KokaoCore, CoreConfig
from kokao.guidance import GuidanceSystem


@pytest.fixture
def core():
    """Создание тестового ядра."""
    return KokaoCore(CoreConfig(input_dim=5))


@pytest.fixture
def guidance(core):
    """Создание системы наведения."""
    return GuidanceSystem(core)


def test_compute_control_vector(guidance):
    """Тест вычисления вектора управления."""
    x = torch.randn(5)
    guidance.set_target(0.8)
    delta = guidance.compute_control_vector(x)
    assert delta.shape == (5,)


def test_step(guidance):
    """Тест одного шага наведения."""
    x = torch.randn(5)
    guidance.set_target(0.8)
    x_new = guidance.step(x)
    assert x_new.shape == (5,)


def test_simulate(guidance):
    """Тест симуляции наведения."""
    x = torch.randn(5)
    guidance.set_target(0.8)
    traj = guidance.simulate(x, steps=5)
    assert len(traj) == 6
    assert all(t.shape == (5,) for t in traj)


def test_set_target(guidance):
    """Тест установки целевого сигнала."""
    guidance.set_target(0.5)
    assert guidance.target_signal == 0.5
    guidance.set_target(1.2)
    assert guidance.target_signal == 1.2
