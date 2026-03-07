"""Тесты для модуля poetic_chain.py."""
import pytest
import torch
from kokao import KokaoCore, CoreConfig
from kokao.poetic_chain import PoeticChain


@pytest.fixture
def core():
    """Создание тестового ядра."""
    return KokaoCore(CoreConfig(input_dim=3))


@pytest.fixture
def chain(core):
    """Создание цепочки."""
    return PoeticChain(core)


def test_run_sequence(chain):
    """Тест запуска последовательности."""
    def rule(s):
        return torch.tensor([0.1, 0.0, -0.1]) * s

    x0 = torch.randn(3)
    history = chain.run_sequence(x0, rule, steps=5)

    assert len(history) == 5
    assert 'signal' in history[0]
    assert 'state' in history[0]
    assert 'delta' in history[0]
    assert 'step' in history[0]


def test_learn_transition(chain):
    """Тест обучения функции перехода."""
    # Используем простую линейную зависимость: delta = W * s + b
    # W: (3,), b: (3,) => delta: (3,)
    true_W = torch.randn(3)
    true_b = torch.randn(3)

    dataset = []
    for _ in range(100):
        s = torch.randn(1).item()
        delta = true_W * s + true_b
        dataset.append((s, delta))

    learned = chain.learn_transition(dataset, lr=0.1, epochs=50)

    s_test = 0.5
    pred = learned(s_test)
    true = true_W * s_test + true_b

    assert torch.allclose(pred, true, atol=0.5)


def test_run_sequence_zero_steps(chain):
    """Тест последовательности с нулевым количеством шагов."""
    def rule(s):
        return torch.tensor([0.1, 0.0, -0.1])

    x0 = torch.randn(3)
    history = chain.run_sequence(x0, rule, steps=0)

    assert len(history) == 0
