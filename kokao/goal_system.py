"""
Самопланирующая система с целями (Глава 4).

Реализация системы с:
- Удовольствием как разностью целей
- Депривацией (удовольствие снижается с насыщением)
- Иерархией целей (от базовых к абстрактным)
- Механизмом утомляемости
"""
import torch
from typing import Optional, Dict, Any
from .core_base import CoreConfig
from .normal_etalon import NormalIntuitiveEtalonSystem


class SelfPlanningSystem:
    """
    Самопланирующая система (Глава 4).
    
    Улучшено:
    - Удовольствие = разность целей (см. "удовольствие — это сигнал о том, что рассогласование уменьшилось")
    - Депривация (удовольствие снижается с насыщением)
    - Иерархия целей (от базовых к абстрактным)
    """

    def __init__(self, config: CoreConfig):
        """
        Инициализация системы.

        Args:
            config: Конфигурация ядра
        """
        self.config = config
        self.normal_system = NormalIntuitiveEtalonSystem(config)

        # Долговременные цели (как в лобной коре)
        # Уровни: базовые (еда, безопасность), социальные (статус, любовь), абстрактные (самовыражение)
        self.goals: Dict[str, Dict[str, Dict[str, float]]] = {
            "physiological": {
                "energy": {"target": 0.8, "value": 1.0, "deprivation": 0.0},
                "safety": {"target": 0.9, "value": 1.0, "deprivation": 0.0}
            },
            "social": {
                "status": {"target": 0.7, "value": 0.8, "deprivation": 0.0},
                "affection": {"target": 0.6, "value": 0.7, "deprivation": 0.0}
            },
            "abstract": {
                "self_expression": {"target": 0.5, "value": 0.5, "deprivation": 0.0}
            }
        }

        # Механизм утомляемости (снижение ценности цели при долгом преследовании)
        self.fatigue: Dict[str, float] = {
            g: 0.0 for level in self.goals for g in self.goals[level]
        }
        self.decay_rate = 0.01

    def get_active_goal(self) -> Optional[str]:
        """
        Выбор текущей активной цели (на основе ценности, утомляемости и депривации).

        Returns:
            ID активной цели или None
        """
        best_goal = None
        best_score = -1

        for level, goals in self.goals.items():
            for goal_id, data in goals.items():
                value = data["value"]
                deprivation = data["deprivation"]
                fatigue = self.fatigue.get(goal_id, 0.0)
                # Чем выше депривация, тем цель приоритетнее; утомляемость снижает приоритет
                score = value * (1.0 - fatigue) * (1.0 + deprivation)
                if score > best_score:
                    best_score = score
                    best_goal = goal_id

        return best_goal

    def experience_pleasure(self, goal_id: str, reward: float) -> None:
        """
        Удовольствие = уменьшение рассогласования с целью (Глава 4.3.2).

        Args:
            goal_id: Идентификатор цели
            reward: Величина награды
        """
        for level, goals in self.goals.items():
            if goal_id in goals:
                goals[goal_id]["value"] += reward
                goals[goal_id]["deprivation"] = max(0.0, goals[goal_id]["deprivation"] - reward)
                self.fatigue[goal_id] = 0.0
                break

        # Утомляемость других целей (переключение фокуса)
        for g in self.fatigue:
            if g != goal_id:
                self.fatigue[g] = min(1.0, self.fatigue[g] + self.decay_rate)

    def experience_displeasure(self, goal_id: str, penalty: float) -> None:
        """
        Неприятность = увеличение рассогласования с целью.

        Args:
            goal_id: Идентификатор цели
            penalty: Величина наказания
        """
        for level, goals in self.goals.items():
            if goal_id in goals:
                goals[goal_id]["value"] -= penalty
                goals[goal_id]["deprivation"] += penalty
                break

    def plan_action_sequence(
        self,
        current_image: torch.Tensor,
        steps: int = 5
    ) -> torch.Tensor:
        """
        Планирование последовательности действий (Глава 4).
        Использует фантазирование для прокручивания сценариев.

        Args:
            current_image: Текущий образ ситуации
            steps: Количество шагов планирования

        Returns:
            Спланированный вектор действия
        """
        current_action = self.normal_system.predict_action(current_image)
        for _ in range(steps):
            # Имитация результата действия
            imagined_image = self.normal_system.association_matrix.T @ current_action
            current_action = self.normal_system.association_matrix @ imagined_image
        return current_action

    def get_current_deprivation(self, goal_id: str) -> float:
        """
        Получить текущую депривацию по цели (голод, страх и т.д.).

        Args:
            goal_id: Идентификатор цели

        Returns:
            Уровень депривации
        """
        for level, goals in self.goals.items():
            if goal_id in goals:
                return goals[goal_id]["deprivation"]
        return 0.0

    def get_goal_value(self, goal_id: str) -> float:
        """
        Получить текущее значение цели.

        Args:
            goal_id: Идентификатор цели

        Returns:
            Значение цели
        """
        for level, goals in self.goals.items():
            if goal_id in goals:
                return goals[goal_id]["value"]
        return 0.0

    def set_goal_target(self, goal_id: str, target: float) -> None:
        """
        Установить целевое значение для цели.

        Args:
            goal_id: Идентификатор цели
            target: Целевое значение
        """
        for level, goals in self.goals.items():
            if goal_id in goals:
                goals[goal_id]["target"] = target
                break

    def get_goal_mismatch(self, goal_id: str) -> float:
        """
        Получить рассогласование цели (target - value).

        Args:
            goal_id: Идентификатор цели

        Returns:
            Рассогласование
        """
        for level, goals in self.goals.items():
            if goal_id in goals:
                return goals[goal_id]["target"] - goals[goal_id]["value"]
        return 0.0

    def get_all_deprivations(self) -> Dict[str, float]:
        """
        Получить все уровни депривации.

        Returns:
            Словарь {goal_id: deprivation}
        """
        deprivations = {}
        for level, goals in self.goals.items():
            for goal_id, data in goals.items():
                deprivations[goal_id] = data["deprivation"]
        return deprivations

    def get_total_deprivation(self) -> float:
        """
        Получить суммарную депривацию по всем целям.

        Returns:
            Суммарная депривация
        """
        return sum(d for d in self.get_all_deprivations().values())

    def satisfy_goal(self, goal_id: str, amount: float) -> None:
        """
        Удовлетворение цели (увеличение value).

        Args:
            goal_id: Идентификатор цели
            amount: Количество удовлетворения
        """
        for level, goals in self.goals.items():
            if goal_id in goals:
                goals[goal_id]["value"] = min(1.0, goals[goal_id]["value"] + amount)
                # Уменьшаем депривацию
                goals[goal_id]["deprivation"] = max(0.0, goals[goal_id]["deprivation"] - amount)
                break

    def get_goal_priority(self, goal_id: str) -> float:
        """
        Получить приоритет цели (на основе депривации и утомляемости).

        Args:
            goal_id: Идентификатор цели

        Returns:
            Приоритет цели
        """
        for level, goals in self.goals.items():
            if goal_id in goals:
                deprivation = goals[goal_id]["deprivation"]
                fatigue = self.fatigue.get(goal_id, 0.0)
                return deprivation * (1.0 - fatigue)
        return 0.0

    def reset_fatigue(self) -> None:
        """Сбросить утомляемость всех целей."""
        for g in self.fatigue:
            self.fatigue[g] = 0.0

    def get_goal_hierarchy(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Получить полную иерархию целей.

        Returns:
            Иерархия целей
        """
        return self.goals
