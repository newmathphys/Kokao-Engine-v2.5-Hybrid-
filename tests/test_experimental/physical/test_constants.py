"""
Тесты для физических констант.
"""
import pytest
import math
from kokao.experimental.physical.constants import K, S3, ALPHA, A0


@pytest.mark.experimental
def test_k_constant():
    """Проверка константы K (отношение масс нейтрона и электрона)."""
    assert K == 1838.684
    assert K > 1


@pytest.mark.experimental
def test_s3_constant():
    """Проверка объёма 3-сферы."""
    expected = 2 * math.pi**2
    assert abs(S3 - expected) < 1e-10


@pytest.mark.experimental
def test_alpha_constant():
    """Проверка постоянной тонкой структуры."""
    assert 0 < ALPHA < 1
    assert abs(ALPHA - 1/137.035999) < 1e-9


@pytest.mark.experimental
def test_a0_constant():
    """Проверка боровского радиуса."""
    assert A0 > 0
    assert A0 < 1e-9  # порядка 10^-10 метров
