"""Тесты для нормальной интуитивно-эталонной системы."""
import pytest
import torch
from kokao.normal_etalon import NormalIntuitiveEtalonSystem
from kokao.core_base import CoreConfig


@pytest.fixture
def system():
    """Фикстура для тестов."""
    config = CoreConfig(input_dim=5)
    return NormalIntuitiveEtalonSystem(config)


def test_learn_pair(system):
    """Тест обучения пары образ-действие."""
    img = torch.randn(5)
    act = torch.randn(5)
    system.learn_image_action_pair(img, act, reward=1.0)
    # проверим, что матрица изменилась (была нулевой)
    assert not torch.allclose(system.association_matrix, torch.zeros(5, 5))


def test_predict_action(system):
    """Тест предсказания действия по образу."""
    img = torch.randn(5)
    act = system.predict_action(img)
    assert act.shape == (5,)


def test_predict_image(system):
    """Тест предсказания образа по действию."""
    act = torch.randn(5)
    img = system.predict_image(act)
    assert img.shape == (5,)


def test_imagine_refine(system):
    """Тест фантазирования и уточнения."""
    init = torch.randn(5)
    refined = system.imagine_and_refine(init, iterations=3)
    assert refined.shape == (5,)


def test_activate_image_etalon(system):
    """Тест активации эталона образа."""
    img = torch.randn(5)
    system.activate_image_etalon(img)
    active = system.get_active_image_etalon()
    assert active is not None
    assert torch.allclose(active, img)


def test_activate_action_etalon(system):
    """Тест активации эталона действия."""
    act = torch.randn(5)
    system.activate_action_etalon(act)
    active = system.get_active_action_etalon()
    assert active is not None
    assert torch.allclose(active, act)


def test_reset_activation(system):
    """Тест сброса активации."""
    img = torch.randn(5)
    act = torch.randn(5)
    system.activate_image_etalon(img)
    system.activate_action_etalon(act)
    system.reset_activation()
    assert system.get_active_image_etalon() is None
    assert system.get_active_action_etalon() is None


def test_strengthen_association(system):
    """Тест усиления ассоциации."""
    img = torch.ones(5)
    act = torch.ones(5)
    initial_matrix = system.association_matrix.clone()
    system.strengthen_association(img, act, strength=0.1)
    # Матрица должна увеличиться
    assert torch.all(system.association_matrix >= initial_matrix)


def test_weaken_association(system):
    """Тест ослабления ассоциации."""
    img = torch.ones(5)
    act = torch.ones(5)
    system.strengthen_association(img, act, strength=0.1)
    initial_matrix = system.association_matrix.clone()
    system.weaken_association(img, act, strength=0.05)
    # Матрица должна уменьшиться
    assert torch.all(system.association_matrix <= initial_matrix)


def test_get_association_strength(system):
    """Тест получения силы ассоциации."""
    img = torch.ones(5)
    act = torch.ones(5)
    system.strengthen_association(img, act, strength=0.1)
    strength = system.get_association_strength(img, act)
    assert strength > 0


def test_clear_associations(system):
    """Тест очистки ассоциаций."""
    img = torch.randn(5)
    act = torch.randn(5)
    system.learn_image_action_pair(img, act, reward=1.0)
    system.clear_associations()
    assert torch.allclose(system.association_matrix, torch.zeros(5, 5))


def test_learn_pair_with_different_rewards(system):
    """Тест обучения с разными наградами."""
    img = torch.randn(5)
    act = torch.randn(5)

    # Обучение с положительной наградой
    system.learn_image_action_pair(img, act, reward=1.0)
    matrix_after_positive = system.association_matrix.clone()

    # Обучение с нулевой наградой (не должно менять матрицу)
    system.learn_image_action_pair(img, act, reward=0.0)
    matrix_after_zero = system.association_matrix.clone()

    # Матрицы должны быть равны (нулевая награда не меняет ассоциации)
    assert torch.allclose(matrix_after_positive, matrix_after_zero)


def test_imagine_multiple_iterations(system):
    """Тест фантазирования с несколькими итерациями."""
    init = torch.randn(5)

    # Разное количество итераций
    result_1 = system.imagine_and_refine(init, iterations=1)
    result_5 = system.imagine_and_refine(init, iterations=5)
    result_10 = system.imagine_and_refine(init, iterations=10)

    # Все должны иметь правильную размерность
    assert result_1.shape == (5,)
    assert result_5.shape == (5,)
    assert result_10.shape == (5,)


def test_bidirectional_prediction(system):
    """Тест двунаправленного предсказания."""
    img = torch.randn(5)

    # Образ -> Действие -> Образ
    act = system.predict_action(img)
    img_reconstructed = system.predict_image(act)

    # Проверка размерности
    assert img_reconstructed.shape == img.shape
