# kokao/inverse.py
"""
Обратная задача: генерация входного вектора x по целевому сигналу S_target.

Решает уравнение S(x) = S_target, где S(x) = S⁺(x)/S⁻(x) с фиксированными весами.

Улучшения v2.0.2:
- Адаптивный learning rate (ReduceLROnPlateau)
- Поддержка L-BFGS оптимизатора
- Multi-restart для избежания локальных минимумов
- Усиленная регуляризация
"""
import torch
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class InverseProblem:
    """
    Решает S(x) = S_target методом градиентного спуска по x.
    
    Поддерживаемые оптимизаторы:
    - Adam (по умолчанию)
    - L-BFGS (для гладкой сходимости)
    - SGD с адаптивным LR
    """
    _EPSILON = 1e-6
    _MAX_STEPS_BASE = 500  # Увеличено с 200
    _MAX_STEPS_EXTREME = 1000  # Увеличено с 500
    _NUM_RESTARTS = 5  # Количество попыток для multi-restart

    def __init__(self, w_plus: torch.Tensor, w_minus: torch.Tensor):
        self.w_plus = w_plus.clone().detach()
        self.w_minus = w_minus.clone().detach()
        self.device = w_plus.device
        self.dtype = w_plus.dtype
        self.input_dim = w_plus.shape[0]

        # Оценим диапазон достижимых сигналов
        self._estimate_signal_range()

    def _estimate_signal_range(self):
        """Оценка диапазона достижимых сигналов для текущих весов."""
        w_plus_norm = torch.norm(self.w_plus)
        w_minus_norm = torch.norm(self.w_minus)

        # Приблизительная оценка (для x с norm=1)
        self.max_achievable_signal = w_plus_norm / self._EPSILON
        self.min_achievable_signal = -w_plus_norm / self._EPSILON

        logger.debug(f"Signal range estimate: [{self.min_achievable_signal:.2f}, {self.max_achievable_signal:.2f}]")

    def _compute_S(self, x: torch.Tensor) -> torch.Tensor:
        """Вычисляет S(x) с сохранением знака S⁻."""
        s_plus = torch.dot(x, self.w_plus)
        s_minus = torch.dot(x, self.w_minus)
        s_abs = s_minus.abs().clamp(min=self._EPSILON)
        return s_plus / s_abs * torch.sign(s_minus)

    def _check_target_feasibility(self, S_target: float) -> Tuple[bool, str]:
        """
        Проверка достижимости целевого сигнала.

        Returns:
            (достижимость, сообщение)
        """
        # Проверка на экстремальные значения
        if abs(S_target) > 100:
            return False, f"S_target={S_target} слишком велик (рекомендуется |S| < 100)"
        if abs(S_target) < self._EPSILON:
            return True, "S_target близок к 0, может потребоваться больше итераций"
        return True, "OK"

    def solve(
        self,
        S_target: float,
        x_init: Optional[torch.Tensor] = None,
        lr: float = 0.1,  # Увеличено с 0.05
        max_steps: Optional[int] = None,
        clamp_range: Tuple[float, float] = (-1.0, 1.0),
        verbose: Optional[bool] = None,
        reg_l2: float = 0.05,  # Увеличено с 0.001
        reg_l1: float = 0.01,  # Новая L1 регуляризация
        tol: float = 1e-6,
        optimizer_type: str = 'adam',  # 'adam', 'lbfgs', 'sgd'
        num_restarts: int = 5,  # Увеличено с 1 до 5
        use_smart_init: bool = True,  # Умная инициализация
    ) -> torch.Tensor:
        """
        Находит x, минимизирующий (S(x) - S_target)².

        Args:
            S_target: Целевой сигнал
            x_init: Начальное приближение
            lr: Learning rate (по умолчанию 0.1)
            max_steps: Максимальное число итераций (None = автовыбор)
            clamp_range: Диапазон ограничений на x
            verbose: Выводить ли прогресс (None = использовать GLOBAL_DEBUG)
            reg_l2: L2 регуляризация (по умолчанию 0.05)
            reg_l1: L1 регуляризация (по умолчанию 0.01)
            tol: Порог сходимости для early stopping
            optimizer_type: Тип оптимизатора ('adam', 'lbfgs', 'sgd')
            num_restarts: Количество попыток с разными начальными точками
            use_smart_init: Использовать умную инициализацию

        Returns:
            Сгенерированный вектор x
        """
        from . import DEBUG as GLOBAL_DEBUG

        if verbose is None:
            verbose = GLOBAL_DEBUG

        # Проверка достижимости цели
        feasible, message = self._check_target_feasibility(S_target)
        if not feasible:
            logger.warning(message)
        elif verbose:
            logger.info(f"S_target={S_target}: {message}")

        # Автовыбор max_steps для экстремальных значений
        if max_steps is None:
            if abs(S_target) > 10:
                max_steps = self._MAX_STEPS_EXTREME
                if verbose:
                    logger.info(f"Экстремальный S_target, увеличено max_steps до {max_steps}")
            else:
                max_steps = self._MAX_STEPS_BASE

        # Multi-restart: запускаем несколько раз с разными начальными точками
        best_x = None
        best_loss = float('inf')
        
        restarts = max(1, num_restarts)
        
        for restart_idx in range(restarts):
            if verbose and restarts > 1:
                print(f"\nRestart {restart_idx + 1}/{restarts}")
            
            # Инициализация x
            if x_init is None:
                # Разная инициализация для каждого restart
                torch.manual_seed(42 + restart_idx)
                
                # Умная инициализация на основе весов
                if use_smart_init:
                    # Инициализация на основе соотношения весов
                    # Для положительных S_target: x ~ w_plus
                    # Для отрицательных S_target: x ~ -w_minus
                    if S_target >= 0:
                        x = self.w_plus.clone() * 0.1 * (1 + 0.1 * torch.randn_like(self.w_plus))
                    else:
                        x = -self.w_minus.clone() * 0.1 * (1 + 0.1 * torch.randn_like(self.w_minus))
                else:
                    x = torch.randn(self.input_dim, device=self.device, dtype=self.dtype) * 0.1
            else:
                x = x_init.clone().detach().to(self.device, self.dtype)

            x.requires_grad_(True)
            
            # Выбор оптимизатора
            if optimizer_type == 'lbfgs':
                # L-BFGS для гладкой сходимости
                optimizer = torch.optim.LBFGS(
                    [x], lr=lr,
                    max_iter=max_steps,
                    history_size=20,
                    line_search_fn='strong_wolfe'
                )
                
                def closure():
                    optimizer.zero_grad()
                    s = self._compute_S(x)
                    # L2 + L1 регуляризация
                    loss = (s - S_target) ** 2 + reg_l2 * torch.norm(x, p=2) ** 2 + reg_l1 * torch.norm(x, p=1)
                    loss.backward()
                    return loss
                
                # L-BFGS делает все шаги внутри step()
                loss = optimizer.step(closure)
                x.grad = None  # Очистка градиента
                
            elif optimizer_type == 'sgd':
                # SGD с адаптивным LR и momentum
                optimizer = torch.optim.SGD([x], lr=lr, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
                )
                
                prev_loss = float('inf')
                for step in range(max_steps):
                    optimizer.zero_grad()
                    s = self._compute_S(x)
                    # L2 + L1 регуляризация
                    loss = (s - S_target) ** 2 + reg_l2 * torch.norm(x, p=2) ** 2 + reg_l1 * torch.norm(x, p=1)
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)
                    x.data.clamp_(*clamp_range)
                    
                    if abs(prev_loss - loss.item()) < tol:
                        break
                    prev_loss = loss.item()
                    
            else:  # adam (по умолчанию)
                # Adam с адаптивным LR
                optimizer = torch.optim.Adam([x], lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
                )
                
                prev_loss = float('inf')
                for step in range(max_steps):
                    optimizer.zero_grad()
                    s = self._compute_S(x)
                    # L2 + L1 регуляризация
                    loss = (s - S_target) ** 2 + reg_l2 * torch.norm(x, p=2) ** 2 + reg_l1 * torch.norm(x, p=1)
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)
                    x.data.clamp_(*clamp_range)
                    
                    if abs(prev_loss - loss.item()) < tol:
                        break
                    prev_loss = loss.item()
            
            # Вычисляем финальную потерю (без регуляризации для оценки качества)
            with torch.no_grad():
                s_final = self._compute_S(x)
                final_loss = (s_final - S_target) ** 2
            
            # Сохраняем лучшее решение
            if final_loss < best_loss:
                best_loss = final_loss
                best_x = x.clone().detach()
            
            # Если уже хорошая точность, прекращаем restarts
            if final_loss < tol:
                if verbose:
                    print(f"Early stop: loss {final_loss:.6f} < tol {tol}")
                break
        
        if verbose and restarts > 1:
            print(f"Best loss after {restarts} restarts: {best_loss:.6f}")
            print(f"Final signal: {self._compute_S(best_x).item():.4f} (target: {S_target})")
            
        return best_x.detach()

    def solve_batch(
        self,
        S_targets: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
        lr: float = 0.1,
        max_steps: int = 100,
        clamp_range: Tuple[float, float] = (-1.0, 1.0),
        reg: float = 0.01,
    ) -> torch.Tensor:
        """
        Решает обратную задачу для батча целевых сигналов.

        Args:
            S_targets: (batch,) - целевые сигналы
            x_init: (batch, input_dim) - начальное приближение
            lr: Learning rate
            max_steps: Максимальное число итераций
            clamp_range: Диапазон ограничений
            reg: L2 регуляризация

        Returns:
            (batch, input_dim) - сгенерированные векторы
        """
        batch_size = S_targets.shape[0]

        if x_init is None:
            X = torch.randn(batch_size, self.input_dim, device=self.device, dtype=self.dtype)
        else:
            X = x_init.clone().detach().to(self.device, self.dtype)

        X.requires_grad_(True)
        optimizer = torch.optim.Adam([X], lr=lr)

        for step in range(max_steps):
            optimizer.zero_grad()

            s_plus = torch.einsum('bi,i->b', X, self.w_plus)
            s_minus = torch.einsum('bi,i->b', X, self.w_minus)
            s_abs = s_minus.abs().clamp(min=self._EPSILON)
            S = s_plus / s_abs * torch.sign(s_minus)

            loss = ((S - S_targets) ** 2).mean() + reg * torch.norm(X, p=2) ** 2
            loss.backward()
            optimizer.step()
            X.data.clamp_(*clamp_range)

        return X.detach()
