"""
Модуль фантазирования – комбинирование эталонов для создания новых.
Основан на книге Косякова Ю.Б., п.3.3.2.
"""

import torch
from typing import List
from .core import KokaoCore


class FantasyEngine:
    """Создание новых концепций путём смешивания существующих."""

    def __init__(
        self,
        core: KokaoCore,
        etalon_pool: List[torch.Tensor] = None
    ):
        """
        Инициализация движка фантазирования.

        Args:
            core: Экземпляр KokaoCore
            etalon_pool: Пул эталонов для комбинирования
        """
        self.core = core
        self.etalon_pool = etalon_pool if etalon_pool is not None else []

    def add_etalon(self, x: torch.Tensor):
        """
        Добавление эталона в пул.

        Args:
            x: Вектор эталона
        """
        self.etalon_pool.append(x.clone().detach())

    def combine_concepts(
        self,
        concept_a: torch.Tensor,
        concept_b: torch.Tensor,
        alpha: float = 0.5,
        refine_steps: int = 20
    ) -> torch.Tensor:
        """
        Создаёт гибрид двух концепций (линейная интерполяция + уточнение).

        Args:
            concept_a: Первая концепция
            concept_b: Вторая концепция
            alpha: Коэффициент смешивания (0 = только b, 1 = только a)
            refine_steps: Шагов уточнения через обратную задачу

        Returns:
            Вектор гибридной концепции
        """
        hybrid_raw = alpha * concept_a + (1.0 - alpha) * concept_b
        solver = self.core.to_inverse_problem()
        target_s = self.core.signal(hybrid_raw)
        return solver.solve(S_target=target_s, max_steps=refine_steps)

    def random_fantasy(
        self,
        num_concepts: int = 2,
        noise: float = 0.1
    ) -> torch.Tensor:
        """
        Случайная фантазия из пула эталонов.

        Args:
            num_concepts: Количество концепций для смешивания
            noise: Уровень шума

        Returns:
            Вектор случайной фантазии

        Raises:
            ValueError: Если в пуле недостаточно эталонов
        """
        if len(self.etalon_pool) < num_concepts:
            raise ValueError("Недостаточно эталонов в пуле")

        indices = torch.randperm(len(self.etalon_pool))[:num_concepts]
        selected = [self.etalon_pool[i] for i in indices]

        weights = torch.rand(num_concepts)
        weights /= weights.sum()

        hybrid = sum(w * v for w, v in zip(weights, selected))
        hybrid = hybrid + torch.randn_like(hybrid) * noise

        solver = self.core.to_inverse_problem()
        target_s = self.core.signal(hybrid)
        return solver.solve(S_target=target_s, max_steps=30)
