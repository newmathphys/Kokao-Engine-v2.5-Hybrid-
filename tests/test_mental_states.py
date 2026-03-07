"""Тесты для модуля mental_states.py."""
import pytest
import torch
from kokao import KokaoCore, CoreConfig
from kokao.mental_states import MentalStateManager


@pytest.fixture
def core():
    """Создание тестового ядра."""
    return KokaoCore(CoreConfig(input_dim=5))


@pytest.fixture
def manager(core):
    """Создание менеджера состояний."""
    return MentalStateManager(core)


def test_add_to_memory(manager):
    """Тест добавления в память."""
    x = torch.randn(5)
    manager.add_to_memory(x)
    assert len(manager.memory_pool) == 1


def test_sleep_cycle(manager):
    """Тест цикла сна (консолидация памяти)."""
    x = torch.randn(5)
    manager.add_to_memory(x)
    manager.sleep_cycle(epochs=2)
    # Проверяем, что обучение прошло без ошибок


def test_hypnosis_imprint(manager):
    """Тест гипнотического внушения."""
    x = torch.randn(5)
    s_before = manager.core.signal(x)
    manager.hypnosis_imprint(x, target_signal=0.9, strength=0.1)
    s_after = manager.core.signal(x)
    assert s_after != s_before


def test_sleep_cycle_empty_memory(manager):
    """Тест цикла сна с пустой памятью."""
    manager.sleep_cycle(epochs=2)
    # Не должно быть ошибок


def test_hypnosis_skip_normalize(manager):
    """Тест гипноза без нормализации."""
    x = torch.randn(5)
    manager.hypnosis_imprint(x, target_signal=0.9, strength=0.1, skip_normalize=True)
    # Проверяем, что прошло без ошибок
