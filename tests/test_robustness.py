"""Тесты для модуля robustness.py."""
import pytest
import torch
from kokao import KokaoCore, CoreConfig
from kokao.robustness import RobustnessAnalyzer


@pytest.fixture
def core():
    """Создание тестового ядра."""
    return KokaoCore(CoreConfig(input_dim=5))


@pytest.fixture
def analyzer(core):
    """Создание анализатора устойчивости."""
    return RobustnessAnalyzer(core)


def test_signal_with_noise(analyzer):
    """Тест вычисления сигнала с шумом."""
    x = torch.randn(5)
    s_clean, s_noisy = analyzer.signal_with_noise(x, noise_level=0.1)
    assert isinstance(s_clean, float)
    assert isinstance(s_noisy, float)
    assert abs(s_clean - s_noisy) < 1.0


def test_noise_tolerance_threshold(analyzer):
    """Тест определения порога устойчивости к шуму."""
    x = torch.randn(5)
    thr = analyzer.noise_tolerance_threshold(x, max_deviation=0.2)
    assert thr >= 0.0


def test_feature_snr(analyzer):
    """Тест оценки отношения сигнал/шум."""
    x = torch.randn(5)
    snr = analyzer.feature_snr(x)
    assert snr.shape == (5,)
    assert torch.all(snr >= 0)


def test_feature_importance(analyzer):
    """Тест важности признаков для устойчивости."""
    x = torch.randn(5)
    imp = analyzer.feature_importance_for_stability(x)
    assert imp.shape == (5,)
    assert torch.all(imp >= 0)
