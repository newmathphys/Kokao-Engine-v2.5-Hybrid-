"""Тесты для интуитивно-эталонной системы (обновлённые для векторизованной версии)."""
import pytest
import torch
from kokao.etalon import IntuitiveEtalonSystem
from kokao.core_base import CoreConfig


@pytest.fixture
def system():
    """Фикстура для тестов."""
    config = CoreConfig(input_dim=5)
    return IntuitiveEtalonSystem(config)


def test_learn_etalon(system):
    """Тест обучения эталону."""
    x = torch.randn(5)
    system.learn_etalon("obj1", x)
    assert system.get_etalon_count() == 1
    stored = system.get_etalon("obj1")
    assert stored is not None
    assert torch.allclose(stored, x)


def test_learn_blurry_etalon(system):
    """Тест обучения размытому эталону."""
    x1 = torch.randn(5)
    x2 = torch.randn(5)
    system.learn_etalon("obj1", x1, blurry=True)
    system.add_to_blurred_etalon("obj1", x2)
    expected = (x1 + x2) / 2
    stored = system.get_etalon("obj1")
    assert stored is not None
    assert torch.allclose(stored, expected)


def test_recognize(system):
    """Тест распознавания."""
    x = torch.randn(5)
    system.learn_etalon("obj1", x)
    best = system.recognize(x)
    assert best == "obj1"
    # Проверим, что активировался
    active = system.get_active_etalon()
    assert active is not None


def test_recognize_threshold(system):
    """Тест порога распознавания."""
    x1 = torch.randn(5)
    x2 = torch.randn(5)  # другой
    system.learn_etalon("obj1", x1)
    best = system.recognize(x2, threshold=0.9)  # высокий порог
    assert best is None


def test_activate_etalon(system):
    """Тест активации эталона."""
    x = torch.randn(5)
    system.learn_etalon("obj1", x)
    vec = system.activate_etalon("obj1")
    assert vec is not None
    active = system.get_active_etalon()
    assert active is not None


def test_get_active_etalon(system):
    """Тест получения активного эталона."""
    x = torch.randn(5)
    system.learn_etalon("obj1", x)
    system.activate_etalon("obj1")
    active = system.get_active_etalon()
    assert active is not None
    assert torch.allclose(active, x)


def test_forget_etalon(system):
    """Тест забывания эталона."""
    x = torch.ones(5) * 0.5
    system.learn_etalon("obj1", x)
    system.forget_etalon("obj1", decay_rate=0.5)
    new_vec = system.get_etalon("obj1")
    # должно уменьшиться, но не ниже порога
    assert torch.all(new_vec <= x * 0.5)
    assert torch.all(new_vec >= system._LOW_THRESHOLD)


def test_reset_activation(system):
    """Тест сброса активации."""
    x = torch.randn(5)
    system.learn_etalon("obj1", x)
    system.activate_etalon("obj1")
    system.reset_activation()
    # Проверяем через get_active_etalon - должен вернуть None после сброса
    active = system.get_active_etalon()
    assert active is None


def test_get_etalon_count(system):
    """Тест подсчета эталонов."""
    assert system.get_etalon_count() == 0
    system.learn_etalon("obj1", torch.randn(5))
    system.learn_etalon("obj2", torch.randn(5))
    assert system.get_etalon_count() == 2


def test_get_all_etalons(system):
    """Тест получения всех эталонов."""
    x1 = torch.randn(5)
    x2 = torch.randn(5)
    system.learn_etalon("obj1", x1)
    system.learn_etalon("obj2", x2)
    etalons = system.get_all_etalons()
    assert len(etalons) == 2
    assert "obj1" in etalons
    assert "obj2" in etalons
    assert torch.allclose(etalons["obj1"], x1)
    assert torch.allclose(etalons["obj2"], x2)


def test_recognize_multiple_etalons(system):
    """Тест распознавания среди нескольких эталонов."""
    x1 = torch.randn(5)
    x2 = torch.randn(5)
    x3 = torch.randn(5)
    system.learn_etalon("obj1", x1)
    system.learn_etalon("obj2", x2)
    system.learn_etalon("obj3", x3)

    # Распознаем x1 - должен найти obj1
    best = system.recognize(x1, threshold=0.5)
    assert best == "obj1"


def test_nonexistent_etalon_activation(system):
    """Тест активации несуществующего эталона."""
    vec = system.activate_etalon("nonexistent")
    assert vec is None


def test_forget_nonexistent_etalon(system):
    """Тест забывания несуществующего эталона (должно игнорироваться)."""
    # Не должно вызывать ошибку
    system.forget_etalon("nonexistent")
    assert True


def test_remove_etalon(system):
    """Тест удаления эталона."""
    system.learn_etalon("obj1", torch.randn(5))
    system.learn_etalon("obj2", torch.randn(5))
    assert system.get_etalon_count() == 2
    
    removed = system.remove_etalon("obj1")
    assert removed is True
    assert system.get_etalon_count() == 1
    assert system.get_etalon("obj1") is None


def test_validate_nan_input(system):
    """Тест валидации: NaN должен вызывать ошибку."""
    x_nan = torch.tensor([1.0, float('nan'), 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="NaN"):
        system.learn_etalon("bad", x_nan)


def test_validate_inf_input(system):
    """Тест валидации: Inf должен вызывать ошибку."""
    x_inf = torch.tensor([1.0, float('inf'), 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="Inf"):
        system.learn_etalon("bad", x_inf)


def test_recognize_batch(system):
    """Тест пакетного распознавания."""
    # Используем разные направления, а не просто масштабированные
    x1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    x2 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
    system.learn_etalon("obj1", x1)
    system.learn_etalon("obj2", x2)
    
    X_batch = torch.stack([x1, x2])
    results = system.recognize_batch(X_batch)
    
    assert len(results) == 2
    assert results[0] == "obj1"
    assert results[1] == "obj2"


def test_get_statistics(system):
    """Тест получения статистики."""
    stats = system.get_statistics()
    assert stats['count'] == 0
    
    system.learn_etalon("obj1", torch.ones(5))
    stats = system.get_statistics()
    assert stats['count'] == 1
    assert 'mean_energy' in stats


if __name__ == "__main__":
    pytest.main([__file__])
