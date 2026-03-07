# kokao/math_exact.py
"""
Модуль точных математических методов для Kokao Engine.

Предоставляет аналитические и численные методы высокой точности для:
- Точного решения обратной задачи (SVD, псевдообращение)
- Аналитического вычисления градиентов и якобианов
- Метода Ньютона-Рафсона для нелинейных уравнений
- Спектрального анализа весовых матриц
- Точной нормализации с контролем ошибок

Модуль использует:
- torch.linalg для линейной алгебры
- torch.autograd для автоматического дифференцирования
- Аналитические формулы там, где это возможно
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class InversionMethod(Enum):
    """Методы решения обратной задачи."""
    GRADIENT_DESCENT = "gradient_descent"
    NEWTON_RAPHSON = "newton_raphson"
    SVD_PSEUDOINVERSE = "svd_pseudoinverse"
    MOORE_PENROSE = "moore_penrose"
    LEVENBERG_MARQUARDT = "levenberg_marquardt"


@dataclass
class MathExactConfig:
    """Конфигурация для точных математических методов."""
    dtype: torch.dtype = torch.float64  # Повышенная точность по умолчанию
    svd_full: bool = True  # Полный SVD разложение
    pinv_rcond: float = 1e-6  # Порог для псевдообращения
    newton_max_iter: int = 100  # Максимум итераций Ньютона
    newton_tol: float = 1e-12  # Точность Ньютона
    gradient_check_eps: float = 1e-8  # Шаг для проверки градиентов
    spectral_norm_power_iter: int = 10  # Итерации для спектральной нормы


class MathExactCore:
    """
    Ядро точных математических методов для Kokao Engine.
    
    Предоставляет методы высокой точности для анализа и решения задач
    двухканального интуитивного ядра.
    """
    
    def __init__(self, config: Optional[MathExactConfig] = None):
        """
        Инициализация ядра точных методов.
        
        Args:
            config: Конфигурация точных методов
        """
        self.config = config if config is not None else MathExactConfig()
        self.dtype = self.config.dtype
        
    # =========================================================================
    # МЕТОД 1: SVD ПСЕВДООБРАЩЕНИЕ ДЛЯ ОБРАТНОЙ ЗАДАЧИ
    # =========================================================================
    
    def solve_inverse_svd(
        self,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor,
        S_target: float,
        x_init: Optional[torch.Tensor] = None,
        method: InversionMethod = InversionMethod.SVD_PSEUDOINVERSE
    ) -> torch.Tensor:
        """
        Точное решение обратной задачи через SVD разложение.
        
        Для уравнения S(x) = S⁺(x) / S⁻(x) = S_target, где:
        - S⁺(x) = x · w⁺
        - S⁻(x) = x · w⁻
        
        Решение: x = S_target · (w⁻)⁺ + α · (w⁺)⁺
        где ⁺ обозначает псевдообращение Мура-Пенроуза.
        
        Args:
            w_plus: Веса возбуждающего канала (n,)
            w_minus: Веса тормозящего канала (n,)
            S_target: Целевой сигнал (скаляр)
            x_init: Начальное приближение (опционально)
            method: Метод решения
            
        Returns:
            x: Решение уравнения (n,)
        """
        w_plus = w_plus.to(dtype=self.dtype)
        w_minus = w_minus.to(dtype=self.dtype)
        n = w_plus.shape[0]
        
        if method == InversionMethod.SVD_PSEUDOINVERSE:
            return self._svd_pseudoinverse_solve(w_plus, w_minus, S_target, x_init)
        elif method == InversionMethod.MOORE_PENROSE:
            return self._moore_penrose_solve(w_plus, w_minus, S_target, x_init)
        elif method == InversionMethod.NEWTON_RAPHSON:
            return self._newton_raphson_solve(w_plus, w_minus, S_target, x_init)
        elif method == InversionMethod.GRADIENT_DESCENT:
            return self._gradient_descent_solve(w_plus, w_minus, S_target, x_init)
        elif method == InversionMethod.LEVENBERG_MARQUARDT:
            return self._levenberg_marquardt_solve(w_plus, w_minus, S_target, x_init)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    
    def _svd_pseudoinverse_solve(
        self,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor,
        S_target: float,
        x_init: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Решение через SVD разложение.
        
        Для уравнения S(x) = S⁺/S⁻ = S_target, используем подход:
        x = α · w⁺ + β · w⁻ где α, β подбираются для выполнения условия
        
        SVD: W = U · Σ · V^T
        """
        # Для простоты используем метод наименьших квадратов
        # Формируем систему: x · w⁺ = S_target, x · w⁻ = 1
        # В матричной форме: W · x^T = [S_target, 1]^T
        
        W = torch.stack([w_plus, w_minus], dim=0)  # (2, n)
        b = torch.tensor([S_target * 1.0, 1.0], dtype=self.dtype, device=w_plus.device)
        
        # SVD разложение
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        # Псевдообращение диагональной матрицы Σ
        S_pinv = torch.zeros_like(S)
        mask = S > self.config.pinv_rcond
        S_pinv[mask] = 1.0 / S[mask]
        
        # Псевдообращенная матрица: W⁺ = V · Σ⁺ · U^T
        W_pinv = Vh.T @ torch.diag(S_pinv) @ U.T  # (n, 2)
        
        # Частное решение
        x_particular = W_pinv @ b  # (n,)
        
        # Если есть начальное приближение, добавляем однородное решение
        if x_init is not None:
            x_init = x_init.to(dtype=self.dtype)
            # Проекция начального приближения на нуль-пространство
            null_component = x_init - W_pinv @ (W @ x_init)
            x = x_particular + null_component
        else:
            x = x_particular
            
        return x.to(w_plus.dtype)
    
    def _moore_penrose_solve(
        self,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor,
        S_target: float,
        x_init: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Решение через псевдообращение Мура-Пенроуза.
        
        Использует torch.linalg.pinv для численно устойчивого вычисления.
        """
        # Формируем матрицу системы
        W = torch.stack([w_plus, w_minus], dim=0)  # (2, n)
        b = torch.tensor([S_target, 1.0], dtype=self.dtype, device=w_plus.device)
        
        # Псевдообращение через torch.linalg.pinv
        W_pinv = torch.linalg.pinv(W, rcond=self.config.pinv_rcond)
        
        # Решение
        x = W_pinv @ b
        
        if x_init is not None:
            x_init = x_init.to(dtype=self.dtype)
            # Добавляем компоненту из нуль-пространства
            null_component = x_init - W_pinv @ (W @ x_init)
            x = x + null_component
            
        return x.to(w_plus.dtype)
    
    # =========================================================================
    # МЕТОД 2: МЕТОД НЬЮТОНА-РАФСОНА
    # =========================================================================
    
    def _newton_raphson_solve(
        self,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor,
        S_target: float,
        x_init: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Метод Ньютона-Рафсона для решения S(x) = S_target.
        
        Итерация: x_{k+1} = x_k - J⁺(x_k) · f(x_k)
        где J - якобиан, f(x) = S(x) - S_target
        """
        n = w_plus.shape[0]
        
        if x_init is None:
            x = torch.zeros(n, dtype=self.dtype, device=w_plus.device)
        else:
            x = x_init.to(dtype=self.dtype).clone()
        
        x.requires_grad_(True)
        optimizer = torch.optim.LBFGS([x], lr=1.0, line_search_fn='strong_wolfe')
        
        def closure():
            optimizer.zero_grad()
            S = self._compute_S_exact(x, w_plus, w_minus)
            loss = (S - S_target) ** 2
            loss.backward()
            return loss
        
        for iteration in range(self.config.newton_max_iter):
            loss = optimizer.step(closure)
            
            if loss.item() < self.config.newton_tol:
                logger.debug(f"Newton-Raphson converged at iteration {iteration}")
                break
                
        return x.detach().to(w_plus.dtype)
    
    def _gradient_descent_solve(
        self,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor,
        S_target: float,
        x_init: Optional[torch.Tensor] = None,
        lr: float = 0.1,
        max_steps: int = 1000
    ) -> torch.Tensor:
        """
        Градиентный спуск с адаптивным learning rate.
        """
        n = w_plus.shape[0]
        
        if x_init is None:
            x = torch.zeros(n, dtype=self.dtype, device=w_plus.device)
        else:
            x = x_init.to(dtype=self.dtype).clone()
        
        x.requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50
        )
        
        best_loss = float('inf')
        best_x = x.clone().detach()
        
        for step in range(max_steps):
            optimizer.zero_grad()
            S = self._compute_S_exact(x, w_plus, w_minus)
            loss = (S - S_target) ** 2
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_x = x.clone().detach()
                
            if loss.item() < self.config.newton_tol:
                break
                
        return best_x.to(w_plus.dtype)
    
    def _levenberg_marquardt_solve(
        self,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor,
        S_target: float,
        x_init: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Метод Левенберга-Марквардта для нелинейных наименьших квадратов.
        """
        n = w_plus.shape[0]
        
        if x_init is None:
            x = torch.zeros(n, dtype=self.dtype, device=w_plus.device)
        else:
            x = x_init.to(dtype=self.dtype).clone()
        
        # Дампирование
        lambda_lm = 0.001
        
        for iteration in range(self.config.newton_max_iter):
            x.requires_grad_(True)
            S = self._compute_S_exact(x, w_plus, w_minus)
            residual = S - S_target
            
            # Якобиан через autograd
            J = torch.autograd.grad(S, x, retain_graph=True)[0]  # (n,)
            
            # H ≈ J^T · J + λ · I
            H = torch.outer(J, J) + lambda_lm * torch.eye(n, dtype=self.dtype, device=x.device)
            g = J * residual.item()
            
            # Решение системы H · Δx = g
            try:
                delta_x = torch.linalg.solve(H, g)
            except RuntimeError:
                # Если матрица вырождена, используем псевдообращение
                delta_x = torch.linalg.lstsq(H, g).solution
            
            x = x - delta_x
            
            if abs(residual.item()) < self.config.newton_tol:
                break
                
            # Адаптация lambda
            if residual.item() < 0:
                lambda_lm *= 0.1
            else:
                lambda_lm *= 10
                
        return x.detach().to(w_plus.dtype)
    
    # =========================================================================
    # МЕТОД 3: АНАЛИТИЧЕСКОЕ ВЫЧИСЛЕНИЕ ГРАДИЕНТОВ
    # =========================================================================
    
    def compute_analytical_gradient(
        self,
        x: torch.Tensor,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor,
        target: float
    ) -> torch.Tensor:
        """
        Аналитическое вычисление градиента функции потерь.
        
        L = (S(x) - target)²
        
        ∂L/∂x = 2 · (S - target) · ∂S/∂x
        
        где ∂S/∂x = (w⁺ · S⁻ - S⁺ · w⁻) / (S⁻)²
        """
        x = x.to(dtype=self.dtype)
        w_plus = w_plus.to(dtype=self.dtype)
        w_minus = w_minus.to(dtype=self.dtype)
        
        S_plus = torch.dot(x, w_plus)
        S_minus = torch.dot(x, w_minus)
        
        # Избегаем деления на ноль
        S_minus_safe = S_minus.clamp(min=self.config.pinv_rcond)
        
        # Сигнал
        S = S_plus / S_minus_safe
        
        # Градиент сигнала по x: ∂S/∂x = (w⁺ · S⁻ - S⁺ · w⁻) / (S⁻)²
        dS_dx = (w_plus * S_minus - w_minus * S_plus) / (S_minus_safe ** 2)
        
        # Градиент функции потерь
        dL_dx = 2.0 * (S - target) * dS_dx
        
        return dL_dx.to(x.dtype)
    
    def compute_jacobian(
        self,
        X: torch.Tensor,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисление якобиана S(x) по x для батча данных.
        
        Args:
            X: Батч векторов (batch_size, n)
            w_plus: Веса (n,)
            w_minus: Веса (n,)
            
        Returns:
            J: Якобиан (batch_size, n)
        """
        X = X.to(dtype=self.dtype)
        w_plus = w_plus.to(dtype=self.dtype)
        w_minus = w_minus.to(dtype=self.dtype)
        
        S_plus = X @ w_plus  # (batch_size,)
        S_minus = X @ w_minus  # (batch_size,)
        
        S_minus_safe = S_minus.clamp(min=self.config.pinv_rcond).unsqueeze(1)  # (batch_size, 1)
        
        # ∂S/∂x_i = (w⁺_i · S⁻ - S⁺ · w⁻_i) / (S⁻)²
        J = (w_plus.unsqueeze(0) * S_minus_safe - 
             w_minus.unsqueeze(0) * S_plus.unsqueeze(1)) / (S_minus_safe ** 2)
        
        return J.to(X.dtype)
    
    def verify_gradient(
        self,
        x: torch.Tensor,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor,
        target: float,
        eps: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Проверка аналитического градиента через конечные разности.
        
        Returns:
            analytical_grad: Аналитический градиент
            numerical_grad: Численный градиент
            relative_error: Относительная ошибка
        """
        if eps is None:
            eps = self.config.gradient_check_eps
            
        x = x.to(dtype=torch.float64)
        w_plus = w_plus.to(dtype=torch.float64)
        w_minus = w_minus.to(dtype=torch.float64)
        
        # Аналитический градиент
        analytical_grad = self.compute_analytical_gradient(x, w_plus, w_minus, target)
        
        # Численный градиент (центральная разность)
        numerical_grad = torch.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.clone()
            x_plus[i] += eps
            x_minus = x.clone()
            x_minus[i] -= eps
            
            S_plus = self._compute_S_exact(x_plus, w_plus, w_minus)
            S_minus = self._compute_S_exact(x_minus, w_plus, w_minus)
            
            loss_plus = (S_plus - target) ** 2
            loss_minus = (S_minus - target) ** 2
            
            numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        # Относительная ошибка
        diff = torch.norm(analytical_grad - numerical_grad)
        sum_norm = torch.norm(analytical_grad) + torch.norm(numerical_grad)
        relative_error = (diff / sum_norm).item() if sum_norm > 0 else 0.0
        
        return analytical_grad, numerical_grad, relative_error
    
    # =========================================================================
    # МЕТОД 4: СПЕКТРАЛЬНЫЙ АНАЛИЗ
    # =========================================================================
    
    def compute_spectrum(
        self,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Спектральный анализ весовых матриц.
        
        Вычисляет:
        - Собственные значения ковариационной матрицы
        - Сингулярные числа
        - Число обусловленности
        - Спектральную норму
        """
        w_plus = w_plus.to(dtype=self.dtype)
        w_minus = w_minus.to(dtype=self.dtype)
        
        # Формируем матрицу весов
        W = torch.stack([w_plus, w_minus], dim=0)  # (2, n)
        
        # Ковариационная матрица
        cov = W @ W.T  # (2, 2)
        
        # Собственные значения
        eigenvalues = torch.linalg.eigvalsh(cov)
        
        # Сингулярные числа
        singular_values = torch.linalg.svdvals(W)
        
        # Число обусловленности
        cond_number = singular_values.max() / singular_values.min().clamp(min=self.config.pinv_rcond)
        
        # Спектральная норма (максимальное сингулярное число)
        spectral_norm = singular_values.max()
        
        return {
            'eigenvalues': eigenvalues,
            'singular_values': singular_values,
            'condition_number': cond_number,
            'spectral_norm': spectral_norm,
            'rank': (singular_values > self.config.pinv_rcond).sum().item()
        }
    
    def compute_spectral_radius(
        self,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor
    ) -> float:
        """
        Спектральный радиус матрицы весов.
        """
        spectrum = self.compute_spectrum(w_plus, w_minus)
        return spectrum['spectral_norm'].item()
    
    # =========================================================================
    # МЕТОД 5: АНАЛИТИЧЕСКАЯ НОРМАЛИЗАЦИЯ
    # =========================================================================
    
    def normalize_weights_analytical(
        self,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor,
        target_sum: float = 100.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Точная аналитическая нормализация весов.
        
        Находит масштабные коэффициенты α, β такие что:
        Σ(α · w⁺) = target_sum
        Σ(β · w⁻) = target_sum
        
        Args:
            w_plus: Исходные веса возбуждающего канала
            w_minus: Исходные веса тормозящего канала
            target_sum: Целевая сумма весов
            
        Returns:
            w_plus_norm: Нормализованные веса w⁺
            w_minus_norm: Нормализованные веса w⁻
        """
        w_plus = w_plus.to(dtype=self.dtype)
        w_minus = w_minus.to(dtype=self.dtype)
        
        # Вычисляем текущие суммы (по модулю для защиты от отрицательных)
        sum_plus = w_plus.abs().sum()
        sum_minus = w_minus.abs().sum()
        
        # Находим масштабные коэффициенты
        alpha = target_sum / sum_plus.clamp(min=self.config.pinv_rcond)
        beta = target_sum / sum_minus.clamp(min=self.config.pinv_rcond)
        
        # Нормализованные веса (со знаком исходных)
        w_plus_norm = w_plus.sign() * w_plus.abs() * alpha
        w_minus_norm = w_minus.sign() * w_minus.abs() * beta
        
        # Контрольная проверка
        actual_sum_plus = w_plus_norm.abs().sum()
        actual_sum_minus = w_minus_norm.abs().sum()
        
        if abs(actual_sum_plus - target_sum) > self.config.pinv_rcond * target_sum:
            logger.warning(f"Нормализация w⁺: сумма = {actual_sum_plus}, ожидалось {target_sum}")
        if abs(actual_sum_minus - target_sum) > self.config.pinv_rcond * target_sum:
            logger.warning(f"Нормализация w⁻: сумма = {actual_sum_minus}, ожидалось {target_sum}")
        
        return w_plus_norm.to(w_plus.dtype), w_minus_norm.to(w_minus.dtype)
    
    def normalize_weights_constrained(
        self,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor,
        target_sum: float = 100.0,
        min_weight: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Нормализация с ограничениями на минимальный вес.
        
        Решает задачу оптимизации:
        min ||w_norm - w||² при условиях:
        - Σ w_norm = target_sum
        - w_norm ≥ min_weight
        """
        w_plus = w_plus.to(dtype=self.dtype)
        w_minus = w_minus.to(dtype=self.dtype)
        
        # Проекция на симплекс с ограничениями
        w_plus_norm = self._project_to_simplex(w_plus, target_sum, min_weight)
        w_minus_norm = self._project_to_simplex(w_minus, target_sum, min_weight)
        
        return w_plus_norm.to(w_plus.dtype), w_minus_norm.to(w_minus.dtype)
    
    def _project_to_simplex(
        self,
        w: torch.Tensor,
        target_sum: float,
        min_weight: float
    ) -> torch.Tensor:
        """
        Проекция вектора на симплекс с ограничениями.
        
        Алгоритм из статьи "Projection onto the probability simplex".
        """
        n = w.shape[0]
        
        # Сортируем по убыванию
        w_sorted = torch.sort(w, descending=True)[0]
        
        # Находим порог
        cumsum = torch.cumsum(w_sorted, dim=0)
        k_array = torch.arange(1, n + 1, dtype=self.dtype, device=w.device)
        
        threshold_candidates = (cumsum - target_sum) / k_array
        
        # Находим k
        condition = w_sorted > threshold_candidates
        k = condition.sum().item()
        
        if k == 0:
            theta = target_sum / n
        else:
            theta = (cumsum[k-1] - target_sum) / k
        
        # Проекция
        w_proj = torch.clamp(w - theta, min=min_weight)
        
        # Корректировка для точной суммы
        current_sum = w_proj.sum()
        if abs(current_sum - target_sum) > self.config.pinv_rcond:
            correction = (target_sum - current_sum) / n
            w_proj = w_proj + correction
            
        return w_proj
    
    # =========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # =========================================================================
    
    def _compute_S_exact(
        self,
        x: torch.Tensor,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor
    ) -> torch.Tensor:
        """Точное вычисление сигнала S(x)."""
        S_plus = torch.dot(x, w_plus)
        S_minus = torch.dot(x, w_minus)
        S_minus_safe = S_minus.clamp(min=self.config.pinv_rcond)
        return S_plus / S_minus_safe
    
    def compute_condition_number(
        self,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor
    ) -> float:
        """Число обусловленности системы."""
        W = torch.stack([w_plus, w_minus], dim=0)
        return torch.linalg.cond(W).item()
    
    def analyze_solvability(
        self,
        w_plus: torch.Tensor,
        w_minus: torch.Tensor,
        S_target: float
    ) -> Dict[str, Any]:
        """
        Анализ разрешимости обратной задачи.
        
        Returns:
            Dictionary с информацией о разрешимости
        """
        spectrum = self.compute_spectrum(w_plus, w_minus)
        cond = spectrum['condition_number']
        
        # Оценка диапазона достижимых сигналов
        w_plus_norm = torch.norm(w_plus)
        w_minus_norm = torch.norm(w_minus)
        
        max_signal = w_plus_norm / self.config.pinv_rcond
        min_signal = -w_plus_norm / self.config.pinv_rcond
        
        feasible = bool(min_signal <= S_target <= max_signal)  # Конвертируем в bool
        
        return {
            'feasible': feasible,
            'condition_number': float(cond),  # Конвертируем в float
            'well_conditioned': bool(cond < 100),
            'max_achievable_signal': float(max_signal),
            'min_achievable_signal': float(min_signal),
            'rank': int(spectrum['rank']),  # Конвертируем в int
            'recommended_method': self._recommend_method(cond, feasible)
        }
    
    def _recommend_method(
        self,
        condition_number: torch.Tensor,
        feasible: bool
    ) -> str:
        """Рекомендация метода решения на основе числа обусловленности."""
        cond = condition_number.item()
        
        if not feasible:
            return "least_squares"  # Метод наименьших квадратов
        
        if cond < 10:
            return "direct"  # Прямое решение
        elif cond < 1000:
            return "svd_pseudoinverse"  # SVD псевдообращение
        else:
            return "regularized"  # Регуляризованное решение


# =============================================================================
# УТИЛИТЫ
# =============================================================================

def create_math_exact_core(
    dtype: torch.dtype = torch.float64,
    **kwargs
) -> MathExactCore:
    """
    Фабричная функция для создания MathExactCore.
    
    Args:
        dtype: Тип данных для вычислений
        **kwargs: Дополнительные параметры конфигурации
        
    Returns:
        Настроенный экземпляр MathExactCore
    """
    config = MathExactConfig(dtype=dtype, **kwargs)
    return MathExactCore(config)


def solve_inverse_exact(
    w_plus: torch.Tensor,
    w_minus: torch.Tensor,
    S_target: float,
    method: str = "svd",
    **kwargs
) -> torch.Tensor:
    """
    Удобная функция для решения обратной задачи точными методами.
    
    Args:
        w_plus: Веса возбуждающего канала
        w_minus: Веса тормозящего канала
        S_target: Целевой сигнал
        method: Метод решения ("svd", "pinv", "newton", "gradient")
        **kwargs: Дополнительные параметры
        
    Returns:
        Решение x
    """
    core = create_math_exact_core()
    
    method_map = {
        "svd": InversionMethod.SVD_PSEUDOINVERSE,
        "pinv": InversionMethod.MOORE_PENROSE,
        "newton": InversionMethod.NEWTON_RAPHSON,
        "gradient": InversionMethod.GRADIENT_DESCENT,
        "lm": InversionMethod.LEVENBERG_MARQUARDT
    }
    
    if method not in method_map:
        raise ValueError(f"Неизвестный метод: {method}. Доступные: {list(method_map.keys())}")
    
    return core.solve_inverse_svd(
        w_plus, w_minus, S_target,
        method=method_map[method],
        **kwargs
    )
