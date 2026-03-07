"""Модуль объяснимого AI для Kokao Engine."""
import logging
from typing import Dict, List, Optional, Union, Callable
import numpy as np
import torch

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from .core import KokaoCore

logger = logging.getLogger(__name__)


class XAIAnalyzer:
    """
    Анализатор объяснимости решений KokaoCore.
    Предоставляет SHAP и LIME объяснения для решений ядра.
    """
    
    def __init__(self, core: KokaoCore):
        """
        Инициализация анализатора объяснимости.
        
        Args:
            core: Экземпляр KokaoCore
        """
        self.core = core
        
        # Создаем функцию для SHAP (возвращает скалярный сигнал)
        self._predict_fn = lambda x: self._predict_batch(x)
    
    def _predict_batch(self, x_batch: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Внутренняя функция для батчевого предсказания.
        
        Args:
            x_batch: Батч входных векторов
            
        Returns:
            Массив сигналов
        """
        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.numpy()
        
        if x_batch.ndim == 1:
            x_batch = x_batch.reshape(1, -1)
        
        signals = []
        for x in x_batch:
            x_tensor = torch.from_numpy(x).float()
            signal = self.core.signal(x_tensor)
            signals.append(signal)
        
        return np.array(signals)
    
    def shap_explain(self, x: Union[torch.Tensor, np.ndarray], method: str = "permutation") -> np.ndarray:
        """
        Вычисление SHAP значений для входного вектора.
        
        Args:
            x: Входной вектор для объяснения
            method: Метод вычисления SHAP ("permutation", "partition", "linear" и т.д.)
            
        Returns:
            Массив SHAP значений для каждого признака
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP не установлен. Установите с помощью: pip install shap"
            )
        
        # Конвертируем в numpy если нужно
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Выбираем подходящий explainer в зависимости от метода
        if method == "permutation":
            explainer = shap.Explainer(self._predict_fn, shap.sample(x, min(100, len(x))))
            shap_values = explainer(x[:1]).values
        elif method == "kernel":
            # Используем KernelExplainer для более общего случая
            explainer = shap.KernelExplainer(self._predict_fn, shap.sample(x, min(100, len(x))))
            shap_values = explainer.shap_values(x[0])
        else:
            # По умолчанию используем Permutation explainer
            explainer = shap.Explainer(self._predict_fn, shap.sample(x, min(100, len(x))))
            shap_values = explainer(x[:1]).values
        
        # Возвращаем значения SHAP для первого (и единственного) экземпляра
        if isinstance(shap_values, list):
            # Если возвращается список (например, для многоклассовой задачи)
            shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        
        if shap_values.ndim > 1:
            # Если много экземпляров, возвращаем только первый
            shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        
        return shap_values
    
    def lime_explain(self, x: Union[torch.Tensor, np.ndarray], 
                     num_samples: int = 5000, 
                     kernel_width: float = 0.5) -> Dict[str, float]:
        """
        Вычисление LIME объяснения для входного вектора.
        
        Args:
            x: Входной вектор для объяснения
            num_samples: Количество сэмплов для аппроксимации
            kernel_width: Ширина ядра для LIME
            
        Returns:
            Словарь с коэффициентами для каждого признака
        """
        if not LIME_AVAILABLE:
            raise ImportError(
                "LIME не установлен. Установите с помощью: pip install lime"
            )
        
        # Конвертируем в numpy если нужно
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        
        if x.ndim != 1:
            raise ValueError(f"LIME expects 1D input, got shape {x.shape}")
        
        # Создаем экземпляр LIME объяснителя
        feature_names = [f"feature_{i}" for i in range(len(x))]
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array([x]),  # Используем текущий вектор как эталон
            feature_names=feature_names,
            mode='regression',
            kernel_width=kernel_width
        )
        
        # Функция предсказания для LIME (должна возвращать массив вероятностей)
        def predict_fn(x_array):
            if isinstance(x_array, torch.Tensor):
                x_array = x_array.numpy()
            
            # Для регрессии возвращаем сигналы как есть
            predictions = []
            for row in x_array:
                signal = self._predict_batch(row.reshape(1, -1))[0]
                predictions.append([signal])
            
            return np.array(predictions)
        
        # Получаем объяснение
        explanation = explainer.explain_instance(
            x, 
            predict_fn, 
            num_features=len(x),
            num_samples=num_samples
        )
        
        # Конвертируем объяснение в словарь
        lime_coeffs = {}
        for feature_idx, weight in explanation.as_list():
            # feature_idx в формате "feature_N"
            feature_num = int(feature_idx.split('_')[1])
            lime_coeffs[feature_num] = weight
        
        return lime_coeffs
    
    def analyze_feature_importance(self, x: Union[torch.Tensor, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Комплексный анализ важности признаков с использованием нескольких методов.
        
        Args:
            x: Входной вектор для анализа
            
        Returns:
            Словарь с результатами разных методов
        """
        results = {}
        
        # Добавляем SHAP если доступен
        if SHAP_AVAILABLE:
            try:
                results['shap_values'] = self.shap_explain(x, method="kernel")
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
                results['shap_values'] = None
        else:
            logger.info("SHAP not available, skipping SHAP analysis")
            results['shap_values'] = None
        
        # Добавляем LIME если доступен
        if LIME_AVAILABLE:
            try:
                results['lime_coeffs'] = self.lime_explain(x)
            except Exception as e:
                logger.warning(f"LIME analysis failed: {e}")
                results['lime_coeffs'] = None
        else:
            logger.info("LIME not available, skipping LIME analysis")
            results['lime_coeffs'] = None
        
        # Добавляем информацию о весах ядра
        with torch.no_grad():
            eff_w_plus, eff_w_minus = self.core._get_effective_weights()
            results['effective_weights_plus'] = eff_w_plus.detach().cpu().numpy()
            results['effective_weights_minus'] = eff_w_minus.detach().cpu().numpy()
        
        # Сигналы для каждого признака (аппроксимация)
        feature_signals_plus = x.detach().cpu().numpy() * results['effective_weights_plus']
        feature_signals_minus = x.detach().cpu().numpy() * results['effective_weights_minus']
        results['approx_feature_contributions_plus'] = feature_signals_plus
        results['approx_feature_contributions_minus'] = feature_signals_minus
        
        return results
    
    def visualize_explanation(self, x: Union[torch.Tensor, np.ndarray], 
                            method: str = "shap", 
                            show_plot: bool = True) -> Optional[object]:
        """
        Визуализация объяснения (если доступна).
        
        Args:
            x: Входной вектор
            method: Метод визуализации ("shap", "lime")
            show_plot: Показывать ли график
            
        Returns:
            Объект визуализации (если доступен)
        """
        if method == "shap" and SHAP_AVAILABLE:
            # Получаем SHAP значения
            shap_vals = self.shap_explain(x)
            
            if show_plot:
                shap.plots.waterfall(shap.Explanation(values=shap_vals, data=x))
            
            return shap_vals
        elif method == "lime" and LIME_AVAILABLE:
            # Получаем LIME объяснение
            lime_exp = self.lime_explain(x)
            
            if show_plot:
                # Визуализация LIME
                lime_vis = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.array([x]) if isinstance(x, np.ndarray) else x.detach().cpu().numpy(),
                    feature_names=[f"feature_{i}" for i in range(len(x) if isinstance(x, np.ndarray) else x.shape[0])]
                ).explain_instance(
                    x if isinstance(x, np.ndarray) else x.detach().cpu().numpy(),
                    self._predict_batch
                )
                lime_vis.show_in_notebook()
            
            return lime_exp
        else:
            logger.warning(f"Visualization method '{method}' not available or not implemented")
            return None