"""Модуль симуляции тестирования на проникновение для KokaoCore."""
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
from enum import Enum

from .core import KokaoCore
from .core_base import CoreConfig

logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Типы атак для тестирования."""
    FGSM = "fgsm"  # Fast Gradient Sign Method
    PGD = "pgd"    # Projected Gradient Descent
    CW = "cw"      # Carlini-Wagner
    RANDOM = "random"  # Случайные возмущения
    TARGETED = "targeted"  # Целевая атака


@dataclass
class AttackResult:
    """Результат атаки."""
    attack_type: str
    success_rate: float
    avg_perturbation: float
    original_accuracy: float
    attacked_accuracy: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return {
            'attack_type': self.attack_type,
            'success_rate': self.success_rate,
            'avg_perturbation': self.avg_perturbation,
            'original_accuracy': self.original_accuracy,
            'attacked_accuracy': self.attacked_accuracy,
            'details': self.details
        }


class AdversarialAttack:
    """
    Базовый класс для состязательных атак.
    """

    def __init__(self, core: KokaoCore, epsilon: float = 0.1):
        """
        Инициализация атаки.

        Args:
            core: Модель для атаки
            epsilon: Максимальное возмущение
        """
        self.core = core
        self.epsilon = epsilon

    def attack(self, x: torch.Tensor, 
               target: Optional[float] = None) -> torch.Tensor:
        """
        Выполнение атаки.

        Args:
            x: Исходный вход
            target: Целевое значение (для целевых атак)

        Returns:
            Возмущенный вход
        """
        raise NotImplementedError


class FGSMAttack(AdversarialAttack):
    """
    Fast Gradient Sign Method (FGSM) атака.
    """

    def attack(self, x: torch.Tensor, 
               target: Optional[float] = None) -> torch.Tensor:
        """
        Выполнение FGSM атаки.

        Args:
            x: Исходный вход
            target: Целевое значение

        Returns:
            Возмущенный вход
        """
        x_adv = x.clone().requires_grad_(True)

        # Прямой проход
        output = self.core.forward(x_adv)

        # Целевое значение (если не указано, используем 0)
        if target is None:
            target = 0.0

        # Потеря
        loss = (output - target) ** 2
        loss.backward()

        # Создание возмущения
        with torch.no_grad():
            perturbation = self.epsilon * torch.sign(x_adv.grad)
            x_adv = x + perturbation

            # Проекция на epsilon-окрестность
            x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)

        return x_adv


class PGDAttack(AdversarialAttack):
    """
    Projected Gradient Descent (PGD) атака.
    """

    def __init__(self, core: KokaoCore, epsilon: float = 0.1,
                 num_steps: int = 10, step_size: float = 0.01):
        """
        Инициализация PGD атаки.

        Args:
            core: Модель для атаки
            epsilon: Максимальное возмущение
            num_steps: Количество шагов
            step_size: Размер шага
        """
        super().__init__(core, epsilon)
        self.num_steps = num_steps
        self.step_size = step_size

    def attack(self, x: torch.Tensor,
               target: Optional[float] = None) -> torch.Tensor:
        """
        Выполнение PGD атаки.

        Args:
            x: Исходный вход
            target: Целевое значение

        Returns:
            Возмущенный вход
        """
        x_adv = x.clone().requires_grad_(True)

        # Начальное возмущение
        with torch.no_grad():
            noise = torch.randn_like(x) * self.epsilon * 0.1
            x_adv = x + noise
            x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)

        # Итеративное обновление
        for _ in range(self.num_steps):
            x_adv.requires_grad_(True)
            output = self.core.forward(x_adv)

            if target is None:
                target = 0.0

            loss = (output - target) ** 2
            loss.backward()

            with torch.no_grad():
                x_adv = x_adv - self.step_size * torch.sign(x_adv.grad)
                x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)

        return x_adv.detach()


class RandomAttack(AdversarialAttack):
    """
    Атака случайными возмущениями.
    """

    def attack(self, x: torch.Tensor,
               target: Optional[float] = None) -> torch.Tensor:
        """
        Выполнение случайной атаки.

        Args:
            x: Исходный вход
            target: Целевое значение

        Returns:
            Возмущенный вход
        """
        with torch.no_grad():
            noise = torch.randn_like(x) * self.epsilon
            x_adv = x + noise
        return x_adv


class PenetrationTester:
    """
    Тестер на проникновение для моделей KokaoCore.
    """

    def __init__(self, core: KokaoCore):
        """
        Инициализация тестера.

        Args:
            core: Модель для тестирования
        """
        self.core = core
        self.results: List[AttackResult] = []

    def create_attack(self, attack_type: AttackType,
                      **kwargs) -> AdversarialAttack:
        """
        Создание атаки по типу.

        Args:
            attack_type: Тип атаки
            **kwargs: Дополнительные параметры

        Returns:
            Атака
        """
        if attack_type == AttackType.FGSM:
            return FGSMAttack(self.core, kwargs.get('epsilon', 0.1))
        elif attack_type == AttackType.PGD:
            return PGDAttack(
                self.core,
                kwargs.get('epsilon', 0.1),
                kwargs.get('num_steps', 10),
                kwargs.get('step_size', 0.01)
            )
        elif attack_type == AttackType.RANDOM:
            return RandomAttack(self.core, kwargs.get('epsilon', 0.1))
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

    def evaluate_attack(self, attack: AdversarialAttack,
                        test_data: List[Tuple[torch.Tensor, float]],
                        target: Optional[float] = None,
                        tolerance: float = 0.5) -> AttackResult:
        """
        Оценка эффективности атаки.

        Args:
            attack: Атака для оценки
            test_data: Тестовые данные
            target: Целевое значение атаки
            tolerance: Допуск для успешной атаки

        Returns:
            Результат атаки
        """
        original_correct = 0
        attacked_correct = 0
        perturbations = []

        for x, true_target in test_data:
            # Оригинальное предсказание
            original_signal = self.core.signal(x)
            original_error = abs(original_signal - true_target)

            if original_error < tolerance:
                original_correct += 1

            # Атакованное предсказание
            attack_target = target if target is not None else true_target
            x_adv = attack.attack(x, attack_target)
            attacked_signal = self.core.signal(x_adv)
            attacked_error = abs(attacked_signal - true_target)

            if attacked_error < tolerance:
                attacked_correct += 1

            # Размер возмущения
            perturbation = torch.norm(x_adv - x).item()
            perturbations.append(perturbation)

        num_samples = len(test_data)
        original_accuracy = original_correct / num_samples
        attacked_accuracy = attacked_correct / num_samples

        # Успешность атаки
        if original_accuracy > 0:
            success_rate = (original_correct - attacked_correct) / original_correct
        else:
            success_rate = 0.0

        return AttackResult(
            attack_type=type(attack).__name__,
            success_rate=success_rate,
            avg_perturbation=np.mean(perturbations),
            original_accuracy=original_accuracy,
            attacked_accuracy=attacked_accuracy,
            details={
                'num_samples': num_samples,
                'epsilon': attack.epsilon,
                'target': target
            }
        )

    def run_penetration_test(self, test_data: List[Tuple[torch.Tensor, float]],
                              attack_types: Optional[List[AttackType]] = None,
                              epsilon_values: Optional[List[float]] = None
                              ) -> List[AttackResult]:
        """
        Запуск полного теста на проникновение.

        Args:
            test_data: Тестовые данные
            attack_types: Типы атак для тестирования
            epsilon_values: Значения epsilon для тестирования

        Returns:
            Результаты всех атак
        """
        if attack_types is None:
            attack_types = [AttackType.FGSM, AttackType.PGD, AttackType.RANDOM]

        if epsilon_values is None:
            epsilon_values = [0.01, 0.05, 0.1, 0.2]

        self.results = []

        for attack_type in attack_types:
            for epsilon in epsilon_values:
                logger.info(f"Testing {attack_type.value} with epsilon={epsilon}")

                attack = self.create_attack(attack_type, epsilon=epsilon)
                result = self.evaluate_attack(attack, test_data)
                self.results.append(result)

                logger.info(
                    f"  Success rate: {result.success_rate:.2%}, "
                    f"Attacked accuracy: {result.attacked_accuracy:.2%}"
                )

        return self.results

    def get_weakest_point(self) -> Optional[AttackResult]:
        """
        Получение наиболее уязвимого места.

        Returns:
            Результат самой успешной атаки
        """
        if not self.results:
            return None

        return max(self.results, key=lambda r: r.success_rate)

    def get_security_score(self) -> float:
        """
        Вычисление оценки безопасности.

        Returns:
            Оценка от 0 до 100
        """
        if not self.results:
            return 100.0

        # Средняя успешность атак
        avg_success_rate = np.mean([r.success_rate for r in self.results])

        # Оценка безопасности (100 - средний успех атак)
        return max(0, 100 - avg_success_rate * 100)

    def generate_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Генерация отчета о тестировании.

        Args:
            save_path: Путь для сохранения отчета

        Returns:
            Отчет
        """
        report = {
            'model_info': {
                'input_dim': self.core.config.input_dim,
                'device': str(self.core.device),
                'version': self.core.version
            },
            'security_score': self.get_security_score(),
            'weakest_point': self.get_weakest_point().to_dict() if self.get_weakest_point() else None,
            'all_results': [r.to_dict() for r in self.results],
            'summary': {
                'num_attacks': len(self.results),
                'avg_success_rate': np.mean([r.success_rate for r in self.results]) if self.results else 0,
                'avg_attacked_accuracy': np.mean([r.attacked_accuracy for r in self.results]) if self.results else 0
            }
        }

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report

    def recommend_defenses(self) -> List[str]:
        """
        Рекомендации по защите.

        Returns:
            Список рекомендаций
        """
        recommendations = []

        if not self.results:
            return ["Проведите тестирование для получения рекомендаций"]

        weakest = self.get_weakest_point()

        if weakest.success_rate > 0.8:
            recommendations.append(
                "Критическая уязвимость! Используйте состязательное обучение."
            )
        elif weakest.success_rate > 0.5:
            recommendations.append(
                "Высокая уязвимость. Добавьте шум к данным обучения."
            )

        if weakest.avg_perturbation < 0.05:
            recommendations.append(
                "Модель уязвима к малым возмущениям. Увеличьте регуляризацию."
            )

        if weakest.attacked_accuracy < 0.3:
            recommendations.append(
                "Рассмотрите использование ансамбля моделей."
            )

        # Общие рекомендации
        recommendations.extend([
            "Регулярно проводите тестирование на уязвимости",
            "Мониторьте входные данные на аномалии",
            "Используйте детекторы состязательных атак"
        ])

        return recommendations


def run_quick_penetration_test(core: KokaoCore,
                                num_samples: int = 50
                                ) -> Dict[str, Any]:
    """
    Быстрый тест на проникновение.

    Args:
        core: Модель для тестирования
        num_samples: Количество тестовых сэмплов

    Returns:
        Результаты теста
    """
    # Генерация тестовых данных
    test_data = []
    for _ in range(num_samples):
        x = torch.randn(core.config.input_dim)
        target = core.signal(x)
        test_data.append((x, target))

    # Запуск тестера
    tester = PenetrationTester(core)
    tester.run_penetration_test(test_data)

    return tester.generate_report()
