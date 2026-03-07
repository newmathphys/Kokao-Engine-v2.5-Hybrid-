"""Тесты для самопланирующей системы с целями."""
import pytest
import torch
from kokao.goal_system import SelfPlanningSystem
from kokao.core_base import CoreConfig


@pytest.fixture
def system():
    """Фикстура для тестов."""
    config = CoreConfig(input_dim=5)
    return SelfPlanningSystem(config)


def test_get_active_goal(system):
    """Тест получения активной цели."""
    active = system.get_active_goal()
    assert active in ["energy", "safety", "status", "affection", "self_expression"]


def test_experience_pleasure(system):
    """Тест переживания удовольствия."""
    old_value = system.goals["physiological"]["energy"]["value"]
    old_dep = system.goals["physiological"]["energy"]["deprivation"]
    system.experience_pleasure("energy", reward=0.2)
    new_value = system.goals["physiological"]["energy"]["value"]
    new_dep = system.goals["physiological"]["energy"]["deprivation"]
    assert new_value > old_value
    assert new_dep <= old_dep


def test_experience_displeasure(system):
    """Тест переживания неудовольствия."""
    old_value = system.goals["physiological"]["energy"]["value"]
    old_dep = system.goals["physiological"]["energy"]["deprivation"]
    system.experience_displeasure("energy", penalty=0.1)
    new_value = system.goals["physiological"]["energy"]["value"]
    new_dep = system.goals["physiological"]["energy"]["deprivation"]
    assert new_value < old_value
    assert new_dep > old_dep


def test_fatigue_increases_for_other_goals(system):
    """Тест увеличения утомляемости для других целей."""
    # после удовольствия для energy, утомляемость других целей должна возрасти
    old_fatigue_status = system.fatigue["status"]
    system.experience_pleasure("energy", reward=0.2)
    new_fatigue_status = system.fatigue["status"]
    assert new_fatigue_status > old_fatigue_status


def test_plan_action_sequence(system):
    """Тест планирования последовательности действий."""
    img = torch.randn(5)
    result = system.plan_action_sequence(img, steps=3)
    assert result.shape == (5,)


def test_get_deprivation(system):
    """Тест получения депривации."""
    depr = system.get_current_deprivation("energy")
    assert depr >= 0


def test_get_goal_value(system):
    """Тест получения значения цели."""
    value = system.get_goal_value("energy")
    assert 0.0 <= value <= 1.0


def test_set_goal_target(system):
    """Тест установки целевого значения."""
    system.set_goal_target("energy", 0.9)
    assert system.goals["physiological"]["energy"]["target"] == 0.9


def test_get_goal_mismatch(system):
    """Тест получения рассогласования цели."""
    mismatch = system.get_goal_mismatch("energy")
    # mismatch = target - value
    expected = system.goals["physiological"]["energy"]["target"] - \
               system.goals["physiological"]["energy"]["value"]
    assert abs(mismatch - expected) < 1e-6


def test_get_all_deprivations(system):
    """Тест получения всех деприваций."""
    deprivations = system.get_all_deprivations()
    assert len(deprivations) == 5  # 5 целей
    assert all(d >= 0 for d in deprivations.values())


def test_get_total_deprivation(system):
    """Тест получения суммарной депривации."""
    total = system.get_total_deprivation()
    assert total >= 0


def test_satisfy_goal(system):
    """Тест удовлетворения цели."""
    # Сначала уменьшим значение чтобы было куда увеличивать
    system.goals["physiological"]["energy"]["value"] = 0.5
    old_value = system.goals["physiological"]["energy"]["value"]
    system.satisfy_goal("energy", amount=0.3)
    new_value = system.goals["physiological"]["energy"]["value"]
    assert new_value > old_value
    assert new_value <= 1.0  # Не больше 1


def test_get_goal_priority(system):
    """Тест получения приоритета цели."""
    # Сначала увеличим депривацию
    system.experience_displeasure("energy", penalty=0.5)
    priority = system.get_goal_priority("energy")
    assert priority > 0


def test_reset_fatigue(system):
    """Тест сброса утомляемости."""
    system.experience_pleasure("energy", reward=0.2)
    # Теперь утомляемость status > 0
    assert system.fatigue["status"] > 0
    system.reset_fatigue()
    assert all(f == 0.0 for f in system.fatigue.values())


def test_get_goal_hierarchy(system):
    """Тест получения иерархии целей."""
    hierarchy = system.get_goal_hierarchy()
    assert "physiological" in hierarchy
    assert "social" in hierarchy
    assert "abstract" in hierarchy
    assert "energy" in hierarchy["physiological"]
    assert "status" in hierarchy["social"]
    assert "self_expression" in hierarchy["abstract"]


def test_pleasure_resets_fatigue_for_goal(system):
    """Тест что удовольствие сбрасывает утомляемость для цели."""
    # Искусственно увеличим утомляемость
    system.fatigue["energy"] = 0.5
    system.experience_pleasure("energy", reward=0.2)
    assert system.fatigue["energy"] == 0.0


def test_deprivation_cannot_be_negative(system):
    """Тест что депривация не может быть отрицательной."""
    system.experience_pleasure("energy", reward=10.0)  # Большая награда
    assert system.goals["physiological"]["energy"]["deprivation"] >= 0


def test_value_cannot_exceed_one(system):
    """Тест что значение цели не может превышать 1."""
    system.satisfy_goal("energy", amount=10.0)  # Большое удовлетворение
    assert system.goals["physiological"]["energy"]["value"] <= 1.0


def test_active_goal_changes_with_deprivation(system):
    """Тест что активная цель меняется с депривацией."""
    # Уменьшим ценность всех целей кроме safety
    system.goals["physiological"]["energy"]["value"] = 0.3
    system.goals["social"]["status"]["value"] = 0.3
    system.goals["social"]["affection"]["value"] = 0.3
    system.goals["abstract"]["self_expression"]["value"] = 0.3
    # Увеличим депривацию для safety
    system.experience_displeasure("safety", penalty=0.8)
    active = system.get_active_goal()
    # safety должна стать приоритетнее из-за высокой депривации
    assert active == "safety"


def test_plan_with_empty_associations(system):
    """Тест планирования с пустыми ассоциациями."""
    system.normal_system.clear_associations()
    img = torch.randn(5)
    result = system.plan_action_sequence(img, steps=3)
    # Результат должен быть нулевым (нет ассоциаций)
    assert torch.allclose(result, torch.zeros(5))
