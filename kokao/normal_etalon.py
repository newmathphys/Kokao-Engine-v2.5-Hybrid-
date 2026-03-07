"""
Нормальная интуитивно-эталонная система (Глава 3).

Реализация системы с:
- Отдельными весами для левого полушария (образы) и правого (действия)
- Hebbian learning только между активными эталонами
- Ассоциативной корой для связей "образ → действие"
- Ограничением роста ассоциативной матрицы
- Механизмом забывания ассоциаций
"""
import torch
import threading
from typing import Optional
from .core import KokaoCore
from .core_base import CoreConfig


class NormalIntuitiveEtalonSystem:
    """
    Нормальная интуитивно-эталонная система (Глава 3).

    Улучшено:
    - Отдельные веса для левого полушария (образы) и правого (действия)
    - Hebbian learning только между активными эталонами
    - Ограничение роста association_matrix (max_norm)
    - Забывание ассоциаций (decay)
    """
    _MAX_ASSOC_NORM = 100.0  # Максимальная норма ассоциативной матрицы
    _DECAY_RATE = 0.001  # Скорость забывания ассоциаций

    def __init__(self, config: CoreConfig):
        """
        Инициализация системы.

        Args:
            config: Конфигурация ядра
        """
        self.config = config
        # Блокировка для потокобезопасности
        self._lock = threading.RLock()
        
        # Левое полушарие: эталоны образов (S⁺, S⁻)
        self.image_core = KokaoCore(config)
        # Правое полушарие: эталоны действий (S⁺, S⁻)
        self.action_core = KokaoCore(config)

        # Ассоциативная кора: матрица связей "образ → действие"
        self.association_matrix = torch.zeros(config.input_dim, config.input_dim)

        # Активные эталоны
        self.active_image_etalon: Optional[torch.Tensor] = None
        self.active_action_etalon: Optional[torch.Tensor] = None

    def _normalize_associations(self) -> None:
        """
        Нормализация ассоциативной матрицы.
        Ограничивает норму матрицы для предотвращения бесконечного роста.
        """
        current_norm = torch.norm(self.association_matrix)
        if current_norm > self._MAX_ASSOC_NORM:
            scale = self._MAX_ASSOC_NORM / current_norm
            self.association_matrix *= scale

    def _forget_associations(self) -> None:
        """
        Медленное забывание ассоциаций.
        Экспоненциальный спад для предотвращения насыщения.
        """
        self.association_matrix *= (1.0 - self._DECAY_RATE)

    def learn_image_action_pair(
        self,
        image_vec: torch.Tensor,
        action_vec: torch.Tensor,
        target_image: float = 0.8,
        target_action: float = 0.8,
        reward: float = 1.0,
        lr: float = 0.01
    ) -> None:
        """
        Обучение пары "образ + действие" (Глава 3).

        Args:
            image_vec: Вектор образа
            action_vec: Вектор действия
            target_image: Целевой сигнал для образа
            target_action: Целевой сигнал для действия
            reward: Награда за пару
            lr: Скорость обучения
        """
        with self._lock:
            # Обучаем образ (левое полушарие)
            self.image_core.train(image_vec, target_image, lr=lr, mode="gradient")
            # Обучаем действие (правое полушарие)
            self.action_core.train(action_vec, target_action, lr=lr, mode="gradient")

            # Укрепляем связь (Hebbian learning) только если reward > 0
            if reward > 0:
                # Внешнее произведение, усиленное наградой
                self.association_matrix += reward * 0.01 * torch.outer(image_vec, action_vec)
                
                # Ограничиваем рост матрицы
                self._normalize_associations()

    def predict_action(self, image_vec: torch.Tensor) -> torch.Tensor:
        """
        По образу — предсказываем действие (как в ассоциативной коре).

        Args:
            image_vec: Вектор образа

        Returns:
            Предсказанный вектор действия
        """
        with self._lock:
            # Применяем небольшое забывание при каждом предсказании
            self._forget_associations()
            return self.association_matrix @ image_vec

    def predict_image(self, action_vec: torch.Tensor) -> torch.Tensor:
        """
        По действию — предсказываем образ (обратная связь).

        Args:
            action_vec: Вектор действия

        Returns:
            Предсказанный вектор образа
        """
        with self._lock:
            return self.association_matrix.T @ action_vec

    def imagine_and_refine(
        self,
        initial_action: torch.Tensor,
        iterations: int = 10
    ) -> torch.Tensor:
        """
        Фантазирование (Глава 3.3.2).
        Использует внутреннее обучение: действие → образ → новое действие.

        Args:
            initial_action: Начальный вектор действия
            iterations: Количество итераций фантазирования

        Returns:
            Уточненный вектор действия
        """
        with self._lock:
            current_action = initial_action.clone()
            for _ in range(iterations):
                # 1. Образ по действию (транспонированная ассоциативная матрица)
                image_pred = self.association_matrix.T @ current_action
                # 2. Новое действие по образу
                current_action = self.association_matrix @ image_pred
            return current_action

    def activate_image_etalon(self, image_vec: torch.Tensor) -> None:
        """
        Активация эталона образа.

        Args:
            image_vec: Вектор образа для активации
        """
        with self._lock:
            self.active_image_etalon = image_vec.clone().detach()

    def activate_action_etalon(self, action_vec: torch.Tensor) -> None:
        """
        Активация эталона действия.

        Args:
            action_vec: Вектор действия для активации
        """
        with self._lock:
            self.active_action_etalon = action_vec.clone().detach()

    def get_active_image_etalon(self) -> Optional[torch.Tensor]:
        """
        Получить активированный образ.

        Returns:
            Вектор активного образа или None
        """
        with self._lock:
            return self.active_image_etalon

    def get_active_action_etalon(self) -> Optional[torch.Tensor]:
        """
        Получить активированный образ действия.

        Returns:
            Вектор активного действия или None
        """
        with self._lock:
            return self.active_action_etalon

    def reset_activation(self) -> None:
        """Сбросить активацию эталонов."""
        with self._lock:
            self.active_image_etalon = None
            self.active_action_etalon = None

    def strengthen_association(
        self,
        image_vec: torch.Tensor,
        action_vec: torch.Tensor,
        strength: float = 0.1
    ) -> None:
        """
        Усиление ассоциации между образом и действием.

        Args:
            image_vec: Вектор образа
            action_vec: Вектор действия
            strength: Сила усиления
        """
        with self._lock:
            self.association_matrix += strength * torch.outer(image_vec, action_vec)
            self._normalize_associations()

    def weaken_association(
        self,
        image_vec: torch.Tensor,
        action_vec: torch.Tensor,
        strength: float = 0.1
    ) -> None:
        """
        Ослабление ассоциации между образом и действием.

        Args:
            image_vec: Вектор образа
            action_vec: Вектор действия
            strength: Сила ослабления
        """
        with self._lock:
            self.association_matrix -= strength * torch.outer(image_vec, action_vec)

    def get_association_strength(self, image_vec: torch.Tensor, action_vec: torch.Tensor) -> float:
        """
        Получить силу ассоциации между образом и действием.

        Args:
            image_vec: Вектор образа
            action_vec: Вектор действия

        Returns:
            Сила ассоциации
        """
        with self._lock:
            return torch.dot(image_vec, self.association_matrix @ action_vec).item()

    def clear_associations(self) -> None:
        """Очистить все ассоциации."""
        with self._lock:
            self.association_matrix = torch.zeros(self.config.input_dim, self.config.input_dim)

    def forget_associations(self, decay_rate: Optional[float] = None) -> None:
        """
        Явное забывание ассоциаций.

        Args:
            decay_rate: Скорость забывания (None = использовать значение по умолчанию)
        """
        with self._lock:
            rate = decay_rate if decay_rate is not None else self._DECAY_RATE
            self.association_matrix *= (1.0 - rate)

    def get_association_norm(self) -> float:
        """
        Получить текущую норму ассоциативной матрицы.

        Returns:
            Норма матрицы
        """
        with self._lock:
            return torch.norm(self.association_matrix).item()

    def set_max_association_norm(self, max_norm: float) -> None:
        """
        Установить максимальную норму ассоциативной матрицы.

        Args:
            max_norm: Новая максимальная норма
        """
        with self._lock:
            self._MAX_ASSOC_NORM = max_norm
