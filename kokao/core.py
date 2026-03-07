# kokao/core.py
"""
Двухканальное интуитивное ядро по теории Косякова.

S = S⁺ / S⁻, где:
    S⁺ = Σ xᵢ wᵢ⁺ (возбуждающий канал)
    S⁻ = Σ xᵢ wᵢ⁻ (тормозящий канал)
"""
import torch
import torch.ao.quantization as quantization
import json
import logging
import threading
import time
from pathlib import Path
from collections import deque
from typing import Optional, Literal, Tuple, Dict, Any

from .core_base import KokaoCoreBase, CoreConfig
from .inverse import InverseProblem

# Импортируем глобальный DEBUG флаг
try:
    from . import DEBUG as GLOBAL_DEBUG
except (ImportError, AttributeError):
    GLOBAL_DEBUG = False

logger = logging.getLogger(__name__)


class KokaoCore(torch.nn.Module, KokaoCoreBase):
    """
    Двухканальное интуитивное ядро по Косякову.
    S⁺ = Σ xᵢ wᵢ⁺, S⁻ = Σ xᵢ wᵢ⁻,  S = S⁺ / S⁻.
    """
    _EPSILON = 1e-6
    _EPSILON64 = 1e-12

    def __init__(self, config: CoreConfig):
        super().__init__()
        self.config = config
        self._lock = threading.RLock()

        dtype = torch.float64 if config.dtype == "float64" else torch.float32
        self.dtype = dtype
        self.EPSILON = self._EPSILON64 if dtype == torch.float64 else self._EPSILON
        self.device = config.device

        # Установка seed для воспроизводимости
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if config.device == "cuda":
                torch.cuda.manual_seed_all(config.seed)

        # Атрибуты для квантования
        self.is_quantized = False
        self.quantized_model = None

        # Параметры весов (requires_grad=True для градиентного обучения)
        self.w_plus = torch.nn.Parameter(
            torch.randn(config.input_dim, device=config.device, dtype=dtype) * 0.1,
            requires_grad=True
        )
        self.w_minus = torch.nn.Parameter(
            torch.randn(config.input_dim, device=config.device, dtype=dtype) * 0.01,
            requires_grad=True
        )

        # Оптимизатор Adam
        self.optimizer = torch.optim.Adam([self.w_plus, self.w_minus], lr=0.01)

        self._normalize()
        
        # История весов (храним статистику для экономии памяти)
        self.history = deque(maxlen=min(config.max_history, 100))
        self.version = 0

    def _get_effective_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Возвращает гарантированно положительные веса (softplus)."""
        if self.is_quantized and self.quantized_model is not None:
            return torch.nn.functional.softplus(self.quantized_model.w_plus), \
                   torch.nn.functional.softplus(self.quantized_model.w_minus)
        else:
            return torch.nn.functional.softplus(self.w_plus), \
                   torch.nn.functional.softplus(self.w_minus)

    @torch.no_grad()
    def _normalize(self) -> None:
        """
        Нормализация весов: сумма эффективных весов каждого канала равна target_sum.
        Использует softplus для гарантии положительности весов.
        """
        target = self.config.target_sum
        eff_w_plus, eff_w_minus = self._get_effective_weights()

        # Проверка: эффективные веса должны быть положительными (благодаря softplus)
        # Если это не так - логируем предупреждение
        if (eff_w_plus < 0).any():
            logger.warning("Negative values detected in eff_w_plus (should not happen with softplus)")
        if (eff_w_minus < 0).any():
            logger.warning("Negative values detected in eff_w_minus (should not happen with softplus)")

        # Вычисляем суммы (без abs(), так как softplus гарантирует положительность)
        total_p = eff_w_plus.sum()
        total_m = eff_w_minus.sum()

        # Защита от нулевой суммы
        if total_p < self.EPSILON:
            logger.warning(f"w_plus sum too small ({total_p.item():.6f}), reinitializing")
            self.w_plus.data.normal_(0.1)
            eff_w_plus, _ = self._get_effective_weights()
            total_p = eff_w_plus.sum()

        if total_m < self.EPSILON:
            logger.warning(f"w_minus sum too small ({total_m.item():.6f}), reinitializing")
            self.w_minus.data.normal_(0.1)
            _, eff_w_minus = self._get_effective_weights()
            total_m = eff_w_minus.sum()

        # Масштабирование к целевой сумме
        target_eff_plus = eff_w_plus * (target / total_p)
        target_eff_minus = eff_w_minus * (target / total_m)

        # Обновляем внутренние параметры (softplus^{-1})
        self.w_plus.copy_(torch.log(torch.expm1(target_eff_plus) + 1e-10))
        self.w_minus.copy_(torch.log(torch.expm1(target_eff_minus) + 1e-10))

        # Защита от NaN
        if torch.isnan(self.w_plus).any():
            logger.error("NaN detected in w_plus during normalization. Reset.")
            self.w_plus.data = torch.ones_like(self.w_plus) * 0.1
        if torch.isnan(self.w_minus).any():
            logger.error("NaN detected in w_minus during normalization. Reset.")
            self.w_minus.data = torch.ones_like(self.w_minus) * 0.1

    @property
    def w(self) -> torch.Tensor:
        """Для обратной совместимости: w = w_plus - w_minus."""
        eff_w_plus, eff_w_minus = self._get_effective_weights()
        return eff_w_plus - eff_w_minus

    def _forward_tensor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Возвращает S⁺ и S⁻ как тензоры (для градиентного режима)."""
        if x.norm() < self.EPSILON:
            logger.warning("Zero/near-zero input in _forward_tensor")
            zero = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            eps = torch.tensor(self.EPSILON, device=self.device, dtype=self.dtype)
            return zero, eps

        eff_w_plus, eff_w_minus = self._get_effective_weights()
        return torch.dot(x, eff_w_plus), torch.dot(x, eff_w_minus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход, возвращает тензор S = S⁺/S⁻.
        
        Использует гладкую аппроксимацию sign() через tanh() для обеспечения
        дифференцируемости в окрестности нуля.
        """
        original_shape = x.shape
        x_flat = x.view(-1, self.config.input_dim)

        eff_w_plus, eff_w_minus = self._get_effective_weights()

        s_plus = torch.einsum('bi,i->b', x_flat, eff_w_plus)
        s_minus = torch.einsum('bi,i->b', x_flat, eff_w_minus)

        # Гладкая аппроксимация sign(): tanh(k*x) где k = 1/EPSILON
        # Это обеспечивает дифференцируемость в окрестности нуля
        s_abs = s_minus.abs().clamp(min=self.EPSILON)
        sign_approx = torch.tanh(s_minus / self.EPSILON)
        s = s_plus / s_abs * sign_approx

        return s.view(original_shape[:-1])

    def signal(self, x: torch.Tensor) -> float:
        """Вычисляет сигнал для одного вектора, возвращает float."""
        if x.ndim != 1 or x.shape[0] != self.config.input_dim:
            raise ValueError(f"Expected 1D tensor of shape ({self.config.input_dim}), got {x.shape}")

        s = self.forward(x).item()

        if self.config.use_log_domain:
            s_abs = abs(s) + self.EPSILON
            log_s = torch.log(torch.tensor(s_abs))
            log_s = torch.nan_to_num(log_s, nan=-1e38, posinf=1e38, neginf=-1e38)
            sign = 1.0 if s >= 0 else -1.0
            return sign * torch.exp(log_s).item()
        return s

    def train(self, x: torch.Tensor, target: float, lr: float = 0.01,
              mode: Literal["gradient", "kosyakov"] = "gradient") -> float:
        """
        Обучение ядра.
        
        Args:
            x: Входной вектор
            target: Целевой сигнал
            lr: Learning rate
            mode: 'gradient' (SGD) или 'kosyakov' (формула 8)
            
        Returns:
            Значение функции потерь
        """
        if x.ndim != 1 or x.shape[0] != self.config.input_dim:
            raise ValueError(f"Expected 1D tensor of shape ({self.config.input_dim}), got {x.shape}")

        with self._lock:
            s_flat = self.forward(x).view(-1)
            loss_val = (s_flat.item() - target) ** 2

            if mode == "gradient":
                loss = (s_flat - target) ** 2
                loss.backward()
                
                # Gradient clipping для защиты от взрывов градиентов
                torch.nn.utils.clip_grad_norm_([self.w_plus, self.w_minus], max_norm=1.0)
                
                with torch.no_grad():
                    self.w_plus -= lr * self.w_plus.grad
                    self.w_minus -= lr * self.w_minus.grad
                    self.w_plus.grad.zero_()
                    self.w_minus.grad.zero_()
                    self._normalize()
            else:  # mode == "kosyakov"
                # Защита от деления на ноль
                sum_x_sq = torch.dot(x, x).item()
                if sum_x_sq < self.EPSILON:
                    logger.warning("Input vector too small for Kosyakov mode, skipping step.")
                    return loss_val
                    
                delta_0 = s_flat.item() - target
                # Формула 8: Δw = (Δ₀·x) / Σx²
                delta_w = (delta_0 / sum_x_sq) * x
                with torch.no_grad():
                    self.w_plus.sub_(delta_w * lr * 0.5)
                    self.w_minus.add_(delta_w * lr * 0.5)
                    self._normalize()

            self.version += 1
            # Храним статистику вместо полных тензоров
            self.history.append({
                'w_plus_norm': self.w_plus.norm().item(),
                'w_minus_norm': self.w_minus.norm().item(),
                'loss': loss_val,
                'timestamp': time.time()
            })
            return loss_val

    def train_adam(self, x: torch.Tensor, target: float) -> float:
        """Обучение через Adam оптимизатор."""
        with self._lock:
            self.optimizer.zero_grad()
            s = self.forward(x)
            loss = (s - target) ** 2
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self._normalize()

            return loss.item()

    def train_batch(
        self,
        X: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.01,
        verbose: Optional[bool] = None,
        max_epochs: int = 1,
    ) -> float:
        """
        Обучение на батче данных.

        Args:
            X: (Batch, input_dim) - входные данные
            targets: (Batch,) - целевые значения
            lr: Learning rate
            verbose: Выводить ли прогресс (None = использовать GLOBAL_DEBUG)
            max_epochs: Количество эпох обучения

        Returns:
            Final loss value
        """
        if verbose is None:
            verbose = GLOBAL_DEBUG

        # Проверка: данные должны быть на том же устройстве, что и модель
        if X.device.type != self.device:
            logger.warning(
                f"Input data on device {X.device.type}, but model on {self.device}. "
                f"Moving data to {self.device}"
            )
            X = X.to(self.device)
        
        if targets.device.type != self.device:
            logger.warning(
                f"Targets on device {targets.device.type}, but model on {self.device}. "
                f"Moving targets to {self.device}"
            )
            targets = targets.to(self.device)

        # Вычисляем интервал для вывода: каждые N эпох, где N = max(1, max_epochs // 10)
        # Для max_epochs=1: вывод на 1 эпохе
        # Для max_epochs=10: вывод на 10 эпохе
        # Для max_epochs=100: вывод каждые 10 эпох
        print_interval = max(1, max_epochs // 10) if max_epochs >= 1 else 1

        with self._lock:
            for epoch in range(max_epochs):
                S = self.forward(X)
                loss = ((S - targets) ** 2).mean()
                loss.backward()

                with torch.no_grad():
                    self.w_plus -= lr * self.w_plus.grad
                    self.w_minus -= lr * self.w_minus.grad
                    self.w_plus.grad.zero_()
                    self.w_minus.grad.zero_()
                    self._normalize()

                if verbose and (epoch + 1) % print_interval == 0:
                    print(f"Epoch {epoch + 1}/{max_epochs}, loss = {loss.item():.6f}")

            return loss.item()

    @torch.no_grad()
    def forget(self, rate: float = 0.1, lambda_l1: float = 0.0) -> None:
        """Забывание: уменьшение весов с возможной L1-регуляризацией."""
        with self._lock:
            eff_w_plus_before, eff_w_minus_before = self._get_effective_weights()

            eff_w_plus_after = eff_w_plus_before * (1.0 - rate)
            eff_w_minus_after = eff_w_minus_before * (1.0 - rate)

            if lambda_l1 > 0:
                eff_w_plus_after = torch.maximum(eff_w_plus_after - lambda_l1, torch.zeros_like(eff_w_plus_after))
                eff_w_minus_after = torch.maximum(eff_w_minus_after - lambda_l1, torch.zeros_like(eff_w_minus_after))

            self.w_plus.copy_(torch.log(torch.expm1(eff_w_plus_after) + 1e-10))
            self.w_minus.copy_(torch.log(torch.expm1(eff_w_minus_after) + 1e-10))
            self.version += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "w_plus": self.w_plus.cpu(),
            "w_minus": self.w_minus.cpu(),
            "config": self.config.model_dump(),
            "version": self.version,
            "is_quantized": self.is_quantized
        }

    def load_state_dict(self, state: Dict[str, Any], device: Optional[str] = None) -> None:
        config = CoreConfig(**state["config"])
        if device:
            config.device = device

        w_plus_data = state["w_plus"]
        w_minus_data = state["w_minus"]

        if isinstance(w_plus_data, list):
            w_plus_data = torch.tensor(w_plus_data, dtype=self.dtype)
        if isinstance(w_minus_data, list):
            w_minus_data = torch.tensor(w_minus_data, dtype=self.dtype)

        self.w_plus = torch.nn.Parameter(w_plus_data.to(config.device, self.dtype), requires_grad=True)
        self.w_minus = torch.nn.Parameter(w_minus_data.to(config.device, self.dtype), requires_grad=True)
        self.version = state.get("version", 0)
        self.is_quantized = state.get("is_quantized", False)
        self.optimizer = torch.optim.Adam([self.w_plus, self.w_minus], lr=0.01)

        with torch.no_grad():
            self._normalize()

    def save(self, path: str) -> None:
        with self._lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            state = {
                "w_plus": self.w_plus.cpu().tolist(),
                "w_minus": self.w_minus.cpu().tolist(),
                "config": self.config.model_dump(),
                "version": self.version,
                "is_quantized": self.is_quantized
            }
            with open(path, "w") as f:
                json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'KokaoCore':
        with open(path) as f:
            state = json.load(f)
        config = CoreConfig(**state["config"])
        if device:
            config.device = device
        core = cls(config)

        w_plus_data = torch.tensor(state["w_plus"], device=config.device, dtype=core.dtype)
        w_minus_data = torch.tensor(state["w_minus"], device=config.device, dtype=core.dtype)

        core.w_plus = torch.nn.Parameter(w_plus_data, requires_grad=True)
        core.w_minus = torch.nn.Parameter(w_minus_data, requires_grad=True)
        core.version = state.get("version", 0)
        core.is_quantized = state.get("is_quantized", False)
        core.optimizer = torch.optim.Adam([core.w_plus, core.w_minus], lr=0.01)

        with torch.no_grad():
            core._normalize()
        return core

    def to_inverse_problem(self) -> InverseProblem:
        """Создаёт InverseProblem с текущими весами."""
        eff_w_plus, eff_w_minus = self._get_effective_weights()
        return InverseProblem(eff_w_plus.clone().detach(), eff_w_minus.clone().detach())

    def quantize_int8(self) -> 'KokaoCore':
        """INT8 квантование для уменьшения размера модели."""
        if self.is_quantized:
            logger.warning("Model is already quantized, returning itself")
            return self

        self.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            self,
            {torch.nn.Parameter},
            dtype=torch.qint8
        )

        quantized_core = KokaoCore(self.config)
        quantized_core.is_quantized = True
        quantized_core.quantized_model = quantized_model

        with torch.no_grad():
            quantized_params = dict(quantized_model.named_parameters())
            if 'w_plus' in quantized_params:
                quantized_core.w_plus = torch.nn.Parameter(quantized_params['w_plus'], requires_grad=True)
            if 'w_minus' in quantized_params:
                quantized_core.w_minus = torch.nn.Parameter(quantized_params['w_minus'], requires_grad=True)

        return quantized_core

    def quantize_int4(self) -> 'KokaoCore':
        """INT4 квантование для уменьшения размера модели."""
        if self.is_quantized:
            logger.warning("Model is already quantized, returning itself")
            return self

        self.eval()
        qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        quantized_core = KokaoCore(self.config)
        quantized_core.qconfig = qconfig
        torch.ao.quantization.prepare(quantized_core, inplace=True)
        torch.ao.quantization.convert(quantized_core, inplace=True)
        quantized_core.is_quantized = True
        return quantized_core

    def __repr__(self) -> str:
        return f"<KokaoCore(dim={self.config.input_dim}, dev={self.device}, ver={self.version}, quantized={self.is_quantized})>"


# =============================================================================
# TORCHSCRIPT INFERENCE WRAPPER
# =============================================================================

class KokaoCoreInference(torch.nn.Module):
    """
    Оптимизированная версия KokaoCore для инференса через TorchScript.
    Не содержит методов обучения, только прямой проход.
    """
    
    def __init__(self, w_plus: torch.Tensor, w_minus: torch.Tensor, eps: float, target_sum: float = 100.0):
        super().__init__()
        self.register_buffer('w_plus', w_plus.detach().clone())
        self.register_buffer('w_minus', w_minus.detach().clone())
        self.register_buffer('eps', torch.tensor(eps))
        self.register_buffer('target_sum', torch.tensor(target_sum))
    
    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход для батча."""
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x_normalized = x / (x_norm + self.eps)
        
        s_plus = torch.einsum('bi,i->b', x_normalized, self.w_plus)
        s_minus = torch.einsum('bi,i->b', x_normalized, self.w_minus)
        
        s_abs = s_minus.abs().clamp(min=self.eps.item())
        return s_plus / s_abs * torch.sign(s_minus)
    
    @torch.jit.export
    def signal_scalar(self, x: torch.Tensor) -> float:
        """Вычисление сигнала для одного вектора."""
        return self.forward(x.unsqueeze(0)).item()


def to_inference(core: KokaoCore) -> KokaoCoreInference:
    """Конвертирует обученное ядро в режим инференса."""
    return KokaoCoreInference(
        core.w_plus.detach(),
        core.w_minus.detach(),
        core.EPSILON,
        core.config.target_sum
    )
