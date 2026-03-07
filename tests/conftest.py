"""
Общие фикстуры и конфигурация для тестов Kokao Engine.
"""
import pytest
import torch
import sys
import os
from pathlib import Path

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kokao import KokaoCore, CoreConfig, MathExactCore, MathExactConfig


# =============================================================================
# ФИКСТУРЫ ДЛЯ ЯДРА
# =============================================================================

@pytest.fixture
def core_config():
    """Конфигурация ядра по умолчанию."""
    return CoreConfig(input_dim=10, seed=42)


@pytest.fixture
def core(core_config):
    """Базовое ядро для тестов."""
    return KokaoCore(core_config)


@pytest.fixture
def core_small():
    """Малое ядро для быстрых тестов."""
    return KokaoCore(CoreConfig(input_dim=5, seed=42))


@pytest.fixture
def core_large():
    """Большое ядро для тестов производительности."""
    return KokaoCore(CoreConfig(input_dim=100, seed=42))


@pytest.fixture
def trained_core(core):
    """Предварительно обученное ядро."""
    x = torch.randn(10)
    for _ in range(50):
        core.train(x, target=0.8, lr=0.01)
    return core


# =============================================================================
# ФИКСТУРЫ ДЛЯ ОБРАТНОЙ ЗАДАЧИ
# =============================================================================

@pytest.fixture
def math_core():
    """Ядро точных математических методов."""
    return MathExactCore(MathExactConfig(dtype=torch.float64))


@pytest.fixture
def sample_weights():
    """Пример весов для тестов обратной задачи."""
    w_plus = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    w_minus = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float64)
    return w_plus, w_minus


# =============================================================================
# ФИКСТУРЫ ДЛЯ ТЕСТОВ ПРОИЗВОДИТЕЛЬНОСТИ
# =============================================================================

@pytest.fixture
def batch_data():
    """Батч данных для тестов."""
    return torch.randn(32, 10), torch.randn(32)


@pytest.fixture
def large_batch_data():
    """Большой батч для тестов производительности."""
    return torch.randn(1000, 50), torch.randn(1000)


# =============================================================================
# ФИКСТУРЫ ДЛЯ ИНТЕГРАЦИОННЫХ ТЕСТОВ
# =============================================================================

@pytest.fixture
def temp_model_path(tmp_path):
    """Временный путь для сохранения модели."""
    return tmp_path / "test_model.json"


# =============================================================================
# МАРКЕРЫ PYTEST
# =============================================================================

def pytest_configure(config):
    """Регистрация маркеров."""
    config.addinivalue_line("markers", "slow: медленные тесты")
    config.addinivalue_line("markers", "benchmark: тесты производительности")
    config.addinivalue_line("markers", "integration: интеграционные тесты")
    config.addinivalue_line("markers", "gpu: тесты требующие GPU")
    config.addinivalue_line("markers", "requires_package: тесты требующие опциональную зависимость")
    config.addinivalue_line("markers", "experimental: экспериментальные тесты (may require extra dependencies)")


# =============================================================================
# УТИЛИТЫ ДЛЯ ТЕСТОВ
# =============================================================================

def check_tensor_finite(tensor, name="tensor"):
    """Проверка тензора на NaN/Inf."""
    assert torch.isfinite(tensor).all(), f"{name} содержит NaN или Inf"


def check_relative_close(actual, expected, rtol=1e-5, atol=1e-8):
    """Проверка относительной близости значений."""
    if isinstance(actual, torch.Tensor):
        assert torch.allclose(actual, expected, rtol=rtol, atol=atol), \
            f"Значения не близки: max_diff = {(actual - expected).abs().max().item()}"
    else:
        assert abs(actual - expected) < atol + rtol * abs(expected), \
            f"Значения не близки: diff = {abs(actual - expected)}"
