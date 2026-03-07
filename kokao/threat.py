"""Модуль анализа угроз для систем на основе KokaoCore."""
import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Set
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
from enum import Enum
from collections import defaultdict

from .core import KokaoCore
from .core_base import CoreConfig

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Уровни угроз."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Типы угроз."""
    ADVERSARIAL_ATTACK = "adversarial_attack"
    DATA_POISONING = "data_poisoning"
    MODEL_INVERSION = "model_inversion"
    MEMBERSHIP_INFERENCE = "membership_inference"
    MODEL_EXTRACTION = "model_extraction"
    BACKDOOR = "backdoor"
    EVASION = "evasion"
    EXPLORATORY = "exploratory"


@dataclass
class Threat:
    """Класс угрозы."""
    threat_id: str
    threat_type: ThreatType
    level: ThreatLevel
    description: str
    indicators: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return {
            'threat_id': self.threat_id,
            'threat_type': self.threat_type.value,
            'level': self.level.value,
            'description': self.description,
            'indicators': self.indicators,
            'mitigations': self.mitigations,
            'detected_at': self.detected_at,
            'confidence': self.confidence
        }


@dataclass
class ThreatReport:
    """Отчет об угрозах."""
    report_id: str
    generated_at: str
    threats: List[Threat] = field(default_factory=list)
    overall_risk: ThreatLevel = ThreatLevel.LOW
    summary: Dict[str, Any] = field(default_factory=dict)

    def add_threat(self, threat: Threat) -> None:
        """Добавление угрозы."""
        self.threats.append(threat)
        self._update_overall_risk()

    def _update_overall_risk(self) -> None:
        """Обновление общего уровня риска."""
        if not self.threats:
            self.overall_risk = ThreatLevel.LOW
            return

        level_order = {
            ThreatLevel.LOW: 0,
            ThreatLevel.MEDIUM: 1,
            ThreatLevel.HIGH: 2,
            ThreatLevel.CRITICAL: 3
        }

        max_level = max(self.threats, key=lambda t: level_order[t.level]).level
        self.overall_risk = max_level

        # Подсчет статистики
        self.summary = {
            'total_threats': len(self.threats),
            'by_level': {
                level.value: sum(1 for t in self.threats if t.level == level)
                for level in ThreatLevel
            },
            'by_type': {
                ttype.value: sum(1 for t in self.threats if t.threat_type == ttype)
                for ttype in ThreatType
            },
            'avg_confidence': np.mean([t.confidence for t in self.threats])
        }

    def save(self, path: str) -> None:
        """Сохранение отчета."""
        report_dict = {
            'report_id': self.report_id,
            'generated_at': self.generated_at,
            'threats': [t.to_dict() for t in self.threats],
            'overall_risk': self.overall_risk.value,
            'summary': self.summary
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(report_dict, f, indent=2)


class ThreatDetector:
    """
    Детектор угроз для систем KokaoCore.
    """

    def __init__(self, core: KokaoCore, baseline_data: Optional[List[torch.Tensor]] = None):
        """
        Инициализация детектора.

        Args:
            core: Модель для защиты
            baseline_data: Базовые данные для сравнения
        """
        self.core = core
        self.baseline_data = baseline_data or []
        self.threat_history: List[Threat] = []
        self.threat_counter = 0

        # Статистика входов
        self.input_stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None
        }

        # Обновление статистики если есть базовые данные
        if self.baseline_data:
            self._update_baseline_stats()

    def _update_baseline_stats(self) -> None:
        """Обновление статистики базовых данных."""
        if not self.baseline_data:
            return

        data_stack = torch.stack(self.baseline_data)
        self.input_stats = {
            'mean': data_stack.mean(dim=0),
            'std': data_stack.std(dim=0),
            'min': data_stack.min(dim=0).values,
            'max': data_stack.max(dim=0).values
        }

    def _generate_threat_id(self) -> str:
        """Генерация ID угрозы."""
        self.threat_counter += 1
        return f"THREAT-{datetime.now().strftime('%Y%m%d')}-{self.threat_counter:04d}"

    def detect_adversarial_input(self, x: torch.Tensor,
                                  threshold: float = 3.0) -> Optional[Threat]:
        """
        Обнаружение состязательного входа.

        Args:
            x: Входные данные для проверки
            threshold: Порог обнаружения (в стандартных отклонениях)

        Returns:
            Угроза если обнаружена
        """
        if self.input_stats['mean'] is None:
            return None

        # Вычисление z-score
        z_score = torch.abs(x - self.input_stats['mean']) / (self.input_stats['std'] + 1e-10)

        # Проверка на аномалии
        if (z_score > threshold).any():
            anomaly_features = (z_score > threshold).sum().item()

            threat = Threat(
                threat_id=self._generate_threat_id(),
                threat_type=ThreatType.ADVERSARIAL_ATTACK,
                level=ThreatLevel.HIGH if anomaly_features > 5 else ThreatLevel.MEDIUM,
                description=f"Обнаружен вход с {anomaly_features} аномальными признаками",
                indicators=[
                    f"Z-score превышает {threshold} для {anomaly_features} признаков",
                    f"Максимальный Z-score: {z_score.max().item():.2f}"
                ],
                mitigations=[
                    "Отклонить входные данные",
                    "Применить входную нормализацию",
                    "Использовать детектор аномалий"
                ],
                confidence=min(1.0, anomaly_features / (self.core.config.input_dim * 0.5))
            )

            self.threat_history.append(threat)
            return threat

        return None

    def detect_model_extraction_attempt(self, query_sequence: List[torch.Tensor],
                                         response_sequence: List[float],
                                         threshold: float = 0.9
                                         ) -> Optional[Threat]:
        """
        Обнаружение попытки извлечения модели.

        Args:
            query_sequence: Последовательность запросов
            response_sequence: Последовательность ответов
            threshold: Порог корреляции

        Returns:
            Угроза если обнаружена
        """
        if len(query_sequence) < 10:
            return None

        # Анализ паттернов запросов
        # Попытки извлечения обычно используют систематические запросы

        # Вычисление корреляции между запросами и ответами
        correlations = []
        for i in range(len(query_sequence) - 1):
            x1, x2 = query_sequence[i], query_sequence[i + 1]
            similarity = torch.cosine_similarity(x1.flatten(), x2.flatten(), dim=0)
            correlations.append(similarity.item())

        # Проверка на систематичность
        if len(correlations) > 0:
            avg_correlation = np.mean(correlations)

            if avg_correlation > threshold:
                threat = Threat(
                    threat_id=self._generate_threat_id(),
                    threat_type=ThreatType.MODEL_EXTRACTION,
                    level=ThreatLevel.HIGH,
                    description="Обнаружена попытка извлечения модели через систематические запросы",
                    indicators=[
                        f"Средняя корреляция между запросами: {avg_correlation:.2f}",
                        f"Количество запросов: {len(query_sequence)}"
                    ],
                    mitigations=[
                        "Ограничить частоту запросов",
                        "Добавить шум к ответам",
                        "Блокировать подозрительные IP"
                    ],
                    confidence=avg_correlation
                )

                self.threat_history.append(threat)
                return threat

        return None

    def detect_data_poisoning(self, training_data: List[Tuple[torch.Tensor, float]],
                               contamination_threshold: float = 0.1
                               ) -> Optional[Threat]:
        """
        Обнаружение отравления данных обучения.

        Args:
            training_data: Данные для проверки
            contamination_threshold: Порог загрязнения

        Returns:
            Угроза если обнаружена
        """
        if len(training_data) < 10:
            return None

        # Анализ распределения меток
        targets = [t for _, t in training_data]
        target_std = np.std(targets)
        target_mean = np.mean(targets)

        # Поиск выбросов
        outliers = []
        for i, (x, t) in enumerate(training_data):
            if abs(t - target_mean) > 3 * target_std:
                outliers.append(i)

        contamination_rate = len(outliers) / len(training_data)

        if contamination_rate > contamination_threshold:
            threat = Threat(
                threat_id=self._generate_threat_id(),
                threat_type=ThreatType.DATA_POISONING,
                level=ThreatLevel.CRITICAL if contamination_rate > 0.3 else ThreatLevel.HIGH,
                description=f"Обнаружено потенциальное отравление данных ({contamination_rate:.1%})",
                indicators=[
                    f"Найдено {len(outliers)} выбросов из {len(training_data)}",
                    f"Уровень загрязнения: {contamination_rate:.1%}"
                ],
                mitigations=[
                    "Очистить данные обучения",
                    "Использовать робастные методы обучения",
                    "Применить фильтрацию выбросов"
                ],
                confidence=contamination_rate
            )

            self.threat_history.append(threat)
            return threat

        return None

    def detect_membership_inference(self, query: torch.Tensor,
                                     confidence: float,
                                     threshold: float = 0.95
                                     ) -> Optional[Threat]:
        """
        Обнаружение попытки вывода о принадлежности.

        Args:
            query: Запрос
            confidence: Уверенность модели
            threshold: Порог уверенности

        Returns:
            Угроза если обнаружена
        """
        # Атаки вывода о принадлежности часто используют высокую уверенность
        if confidence > threshold:
            threat = Threat(
                threat_id=self._generate_threat_id(),
                threat_type=ThreatType.MEMBERSHIP_INFERENCE,
                level=ThreatLevel.MEDIUM,
                description="Потенциальная попытка вывода о принадлежности к обучающей выборке",
                indicators=[
                    f"Уверенность модели: {confidence:.2%}",
                    f"Порог: {threshold:.2%}"
                ],
                mitigations=[
                    "Применить дифференциальную приватность",
                    "Ограничить уверенность выходов",
                    "Добавить калибровку модели"
                ],
                confidence=(confidence - threshold) / (1 - threshold)
            )

            self.threat_history.append(threat)
            return threat

        return None

    def analyze_threat_landscape(self) -> ThreatReport:
        """
        Анализ ландшафта угроз.

        Returns:
            Отчет об угрозах
        """
        report = ThreatReport(
            report_id=f"REPORT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            generated_at=datetime.now().isoformat()
        )

        # Добавление всех обнаруженных угроз
        for threat in self.threat_history:
            report.add_threat(threat)

        # Анализ трендов
        if self.threat_history:
            threat_types = defaultdict(int)
            for threat in self.threat_history:
                threat_types[threat.threat_type.value] += 1

            most_common = max(threat_types.items(), key=lambda x: x[1])
            report.summary['most_common_threat'] = most_common[0]
            report.summary['most_common_count'] = most_common[1]

        return report

    def get_recommendations(self) -> List[str]:
        """
        Получение рекомендаций по безопасности.

        Returns:
            Список рекомендаций
        """
        recommendations = []

        if not self.threat_history:
            return [
                "Продолжайте мониторинг системы",
                "Регулярно обновляйте базовые данные",
                "Проводите периодические аудиты безопасности"
            ]

        # Анализ типов угроз
        threat_types = set(t.threat_type for t in self.threat_history)

        if ThreatType.ADVERSARIAL_ATTACK in threat_types:
            recommendations.extend([
                "Внедрите защиту от состязательных атак",
                "Используйте состязательное обучение",
                "Примените входную валидацию"
            ])

        if ThreatType.DATA_POISONING in threat_types:
            recommendations.extend([
                "Усильте валидацию данных обучения",
                "Используйте робастные методы обучения",
                "Внедрите систему обнаружения аномалий"
            ])

        if ThreatType.MODEL_EXTRACTION in threat_types:
            recommendations.extend([
                "Ограничьте частоту API запросов",
                "Добавьте шум к предсказаниям",
                "Мониторьте паттерны использования"
            ])

        # Общие рекомендации
        recommendations.extend([
            "Регулярно обновляйте модель безопасности",
            "Проводите красные команды (red team exercises)",
            "Документируйте все инциденты безопасности"
        ])

        return recommendations


class ThreatIntelligence:
    """
    База знаний об угрозах (Threat Intelligence).
    """

    def __init__(self):
        """Инициализация базы знаний."""
        self.known_threats: Dict[str, Dict[str, Any]] = {}
        self._load_known_threats()

    def _load_known_threats(self) -> None:
        """Загрузка известных угроз."""
        self.known_threats = {
            'FGSM': {
                'type': ThreatType.ADVERSARIAL_ATTACK,
                'description': 'Fast Gradient Sign Method - быстрая градиентная атака',
                'indicators': ['Малые возмущения входа', 'Направленность по градиенту'],
                'mitigations': ['Состязательное обучение', 'Входная нормализация']
            },
            'PGD': {
                'type': ThreatType.ADVERSARIAL_ATTACK,
                'description': 'Projected Gradient Descent - итеративная атака',
                'indicators': ['Многократные запросы', 'Малые изменения между запросами'],
                'mitigations': ['Ограничение запросов', 'Рандомизация']
            },
            'Model Inversion': {
                'type': ThreatType.MODEL_INVERSION,
                'description': 'Попытка восстановления обучающих данных',
                'indicators': ['Систематические запросы', 'Анализ выходов'],
                'mitigations': ['Дифференциальная приватность', 'Ограничение доступа']
            }
        }

    def get_threat_info(self, threat_name: str) -> Optional[Dict[str, Any]]:
        """Получение информации об угрозе."""
        return self.known_threats.get(threat_name)

    def search_threats(self, keyword: str) -> List[Dict[str, Any]]:
        """Поиск угроз по ключевому слову."""
        results = []
        keyword_lower = keyword.lower()

        for name, info in self.known_threats.items():
            if (keyword_lower in name.lower() or
                keyword_lower in info['description'].lower()):
                results.append({
                    'name': name,
                    **info
                })

        return results


def create_threat_detector(core: KokaoCore,
                           baseline_samples: int = 100
                           ) -> ThreatDetector:
    """
    Создание детектора угроз с базовыми данными.

    Args:
        core: Модель для защиты
        baseline_samples: Количество базовых сэмплов

    Returns:
        Настроенный детектор угроз
    """
    # Генерация базовых данных
    baseline_data = [
        torch.randn(core.config.input_dim)
        for _ in range(baseline_samples)
    ]

    return ThreatDetector(core, baseline_data)
