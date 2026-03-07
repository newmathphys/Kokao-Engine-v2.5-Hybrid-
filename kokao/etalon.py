"""
Интуитивно-эталонная система (Глава 2).

Реализация простейшей интуитивно-эталонной системы с поддержкой:
- Размытых эталонов (сумма образов, общие черты усиливаются)
- Нижнего порога проводимости
- Активации эталонов (как во сне или фантазировании)
- Векторизованного распознавания
- Потокобезопасности (threading.RLock)
"""
import torch
import threading
from typing import Optional, Dict, Any, List, Tuple
from .core import KokaoCore
from .core_base import CoreConfig


class IntuitiveEtalonSystem:
    """
    Простейшая интуитивно-эталонная система (Глава 2).
    
    Потокобезопасна: все публичные методы используют RLock.
    """
    _LOW_THRESHOLD = 0.1
    _MAX_ENERGY = 1000.0
    _MIN_ENERGY = 0.01
    _TARGET_NORM = 10.0

    def __init__(self, config: CoreConfig):
        self.config = config
        self.intuitive_core = KokaoCore(config)
        
        # Блокировка для потокобезопасности
        self._lock = threading.RLock()

        # Векторизованное хранение: матрица эталонов
        self.etalon_matrix = torch.empty(0, config.input_dim, device=config.device)
        self.id_map: List[str] = []
        self.activated_flags: List[bool] = []
        self.blurry_flags: List[bool] = []

    def _validate_vector(self, x: torch.Tensor, etalon_id: str) -> torch.Tensor:
        """Валидация и нормализация вектора."""
        # Проверка на NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(f"Invalid etalon {etalon_id}: contains NaN or Inf")

        # Проверка энергии
        energy = torch.norm(x)
        if energy > self._MAX_ENERGY or energy < self._MIN_ENERGY:
            x = x / (energy + 1e-6) * self._TARGET_NORM

        return x

    def _find_etalon_index(self, etalon_id: str) -> int:
        """Найти индекс эталона по ID."""
        try:
            return self.id_map.index(etalon_id)
        except ValueError:
            raise ValueError(f"Etalon {etalon_id} not found")

    def learn_etalon(self, etalon_id: str, x: torch.Tensor, blurry: bool = False) -> None:
        """
        Обучение эталону.

        Args:
            etalon_id: Идентификатор эталона
            x: Вектор образа
            blurry: Если True, сохраняем как "размытый эталон"
        """
        with self._lock:
            x = self._validate_vector(x, etalon_id)
            x = x.clone().detach()

            if etalon_id in self.id_map:
                idx = self._find_etalon_index(etalon_id)
                self.etalon_matrix[idx] = x
            else:
                self.etalon_matrix = torch.cat([self.etalon_matrix, x.unsqueeze(0)], dim=0)
                self.id_map.append(etalon_id)
                self.activated_flags.append(False)
                self.blurry_flags.append(blurry)

    def add_to_blurred_etalon(self, etalon_id: str, x: torch.Tensor) -> None:
        """
        Уточнение размытого эталона.
        Добавляет новую ситуацию к уже существующему размытому эталону.
        """
        with self._lock:
            if etalon_id not in self.id_map:
                raise ValueError(f"Etalon {etalon_id} not found")

            idx = self._find_etalon_index(etalon_id)
            if not self.blurry_flags[idx]:
                raise ValueError(f"Etalon {etalon_id} is not blurry")

            x = self._validate_vector(x, etalon_id)
            self.etalon_matrix[idx] = (self.etalon_matrix[idx] + x) / 2.0

    def recognize(self, x: torch.Tensor, threshold: float = 0.1) -> Optional[str]:
        """
        Распознавание.
        Активирует ближайший эталон, если косинусное сходство > threshold.
        """
        with self._lock:
            if self.etalon_matrix.shape[0] == 0:
                return None

            x = self._validate_vector(x, "input")

            # Векторизованное распознавание
            x_norm = torch.norm(x)
            etalon_norms = torch.norm(self.etalon_matrix, dim=1)
            similarities = torch.mm(x.unsqueeze(0), self.etalon_matrix.T) / (x_norm * etalon_norms + 1e-8)

            best_similarity, best_idx = torch.max(similarities, dim=1)
            best_idx = best_idx.item()
            best_similarity = best_similarity.item()

            if best_similarity > threshold:
                self.activated_flags[best_idx] = True
                return self.id_map[best_idx]
            return None

    def recognize_batch(self, X: torch.Tensor, threshold: float = 0.1) -> List[Optional[str]]:
        """Пакетное распознавание."""
        with self._lock:
            if self.etalon_matrix.shape[0] == 0:
                return [None] * X.shape[0]

            x_norms = torch.norm(X, dim=1, keepdim=True)
            etalon_norms = torch.norm(self.etalon_matrix, dim=1)
            similarities = torch.mm(X, self.etalon_matrix.T) / (x_norms * etalon_norms + 1e-8)

            best_similarities, best_indices = torch.max(similarities, dim=1)

            results = []
            for i, (sim, idx) in enumerate(zip(best_similarities, best_indices)):
                if sim > threshold:
                    self.activated_flags[idx.item()] = True
                    results.append(self.id_map[idx.item()])
                else:
                    results.append(None)

            return results

    def activate_etalon(self, etalon_id: str) -> Optional[torch.Tensor]:
        """Активация эталона внутренне (как во сне или фантазировании)."""
        with self._lock:
            if etalon_id in self.id_map:
                idx = self._find_etalon_index(etalon_id)
                self.activated_flags[idx] = True
                return self.etalon_matrix[idx].clone()
            return None

    def get_active_etalon(self) -> Optional[torch.Tensor]:
        """Получить активированный эталон (или None, если нет активных)."""
        with self._lock:
            for i, activated in enumerate(self.activated_flags):
                if activated:
                    return self.etalon_matrix[i].clone()
            return None

    def get_all_active_etalons(self) -> List[torch.Tensor]:
        """Получить все активированные эталоны."""
        with self._lock:
            return [
                self.etalon_matrix[i].clone()
                for i, activated in enumerate(self.activated_flags)
                if activated
            ]

    def forget_etalon(self, etalon_id: str, decay_rate: float = 0.01) -> None:
        """Забывание эталона."""
        with self._lock:
            if etalon_id in self.id_map:
                idx = self._find_etalon_index(etalon_id)
                vec = self.etalon_matrix[idx] * (1.0 - decay_rate)
                vec = torch.max(vec, torch.full_like(vec, self._LOW_THRESHOLD))
                self.etalon_matrix[idx] = vec

    def reset_activation(self) -> None:
        """Сбросить активацию всех эталонов."""
        with self._lock:
            self.activated_flags = [False] * len(self.id_map)

    def get_etalon_count(self) -> int:
        """Получить количество эталонов."""
        with self._lock:
            return len(self.id_map)

    def get_all_etalons(self) -> Dict[str, torch.Tensor]:
        """Получить все эталоны."""
        with self._lock:
            return {eid: self.etalon_matrix[i].clone() for i, eid in enumerate(self.id_map)}

    def get_etalon(self, etalon_id: str) -> Optional[torch.Tensor]:
        """Получить конкретный эталон по ID."""
        with self._lock:
            if etalon_id in self.id_map:
                idx = self._find_etalon_index(etalon_id)
                return self.etalon_matrix[idx].clone()
            return None

    def remove_etalon(self, etalon_id: str) -> bool:
        """Удалить эталон."""
        with self._lock:
            if etalon_id not in self.id_map:
                return False

            idx = self._find_etalon_index(etalon_id)
            mask = torch.ones(self.etalon_matrix.shape[0], dtype=torch.bool)
            mask[idx] = False
            self.etalon_matrix = self.etalon_matrix[mask]

            del self.id_map[idx]
            del self.activated_flags[idx]
            del self.blurry_flags[idx]

            return True

    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику системы эталонов."""
        with self._lock:
            if self.etalon_matrix.shape[0] == 0:
                return {'count': 0}

            energies = torch.norm(self.etalon_matrix, dim=1)
            return {
                'count': len(self.id_map),
                'active_count': sum(self.activated_flags),
                'blurry_count': sum(self.blurry_flags),
                'mean_energy': energies.mean().item(),
                'min_energy': energies.min().item(),
                'max_energy': energies.max().item(),
            }
