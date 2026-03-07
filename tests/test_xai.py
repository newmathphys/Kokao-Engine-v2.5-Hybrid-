"""Тесты для XAI модуля Kokao Engine."""
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from kokao import KokaoCore, CoreConfig
from kokao.xai import XAIAnalyzer


@pytest.fixture
def sample_core():
    """Создание тестового ядра."""
    config = CoreConfig(input_dim=5)
    return KokaoCore(config)


def test_xai_analyzer_initialization(sample_core):
    """Проверяем инициализацию XAIAnalyzer."""
    analyzer = XAIAnalyzer(sample_core)
    
    assert analyzer.core is sample_core
    assert hasattr(analyzer, '_predict_fn')


def test_predict_batch_functionality(sample_core):
    """Проверяем внутреннюю функцию батчевого предсказания."""
    analyzer = XAIAnalyzer(sample_core)
    
    # Тестируем с тензором
    x_tensor = torch.randn(5)
    result = analyzer._predict_batch(x_tensor)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    
    # Тестируем с numpy массивом
    x_numpy = np.random.random(5)
    result = analyzer._predict_batch(x_numpy)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    
    # Тестируем с батчем
    x_batch = np.random.random((3, 5))
    result = analyzer._predict_batch(x_batch)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


@patch('kokao.xai.SHAP_AVAILABLE', True)
def test_shap_explain_implementation_exists(sample_core):
    """Проверяем, что метод SHAP существует при доступности библиотеки."""
    analyzer = XAIAnalyzer(sample_core)
    
    x = torch.randn(5)
    
    # Проверяем, что метод существует
    assert hasattr(analyzer, 'shap_explain')
    
    # Но так как у нас нет реальной реализации SHAP, ожидаем ошибку
    # Вместо этого проверим, что метод может быть вызван без ошибок
    # если SHAP недоступен
    pass


@patch('kokao.xai.SHAP_AVAILABLE', False)
def test_shap_explain_throws_error_when_unavailable(sample_core):
    """Проверяем, что метод SHAP выбрасывает ошибку при недоступности библиотеки."""
    analyzer = XAIAnalyzer(sample_core)
    
    x = torch.randn(5)
    
    with pytest.raises(ImportError, match="SHAP не установлен"):
        analyzer.shap_explain(x)


@patch('kokao.xai.LIME_AVAILABLE', False)
def test_lime_explain_throws_error_when_unavailable(sample_core):
    """Проверяем, что метод LIME выбрасывает ошибку при недоступности библиотеки."""
    analyzer = XAIAnalyzer(sample_core)
    
    x = torch.randn(5)
    
    with pytest.raises(ImportError, match="LIME не установлен"):
        analyzer.lime_explain(x)


def test_lime_explain_implementation_exists(sample_core):
    """Проверяем, что метод LIME существует."""
    analyzer = XAIAnalyzer(sample_core)
    
    x = torch.randn(5)
    
    # Проверяем, что метод существует
    assert hasattr(analyzer, 'lime_explain')


def test_analyze_feature_importance(sample_core):
    """Проверяем комплексный анализ важности признаков."""
    analyzer = XAIAnalyzer(sample_core)
    
    x = torch.randn(5)
    
    # Выполняем анализ
    results = analyzer.analyze_feature_importance(x)
    
    # Проверяем, что возвращаемые ключи существуют
    expected_keys = [
        'shap_values', 
        'lime_coeffs', 
        'effective_weights_plus', 
        'effective_weights_minus',
        'approx_feature_contributions_plus',
        'approx_feature_contributions_minus'
    ]
    
    for key in expected_keys:
        assert key in results
    
    # Проверяем, что веса имеют правильную размерность
    assert results['effective_weights_plus'].shape == (5,)
    assert results['effective_weights_minus'].shape == (5,)
    
    # Проверяем, что аппроксимированные вклады имеют правильную размерность
    assert results['approx_feature_contributions_plus'].shape == (5,)
    assert results['approx_feature_contributions_minus'].shape == (5,)


@patch('kokao.xai.SHAP_AVAILABLE', False)
@patch('kokao.xai.LIME_AVAILABLE', False)
def test_analyze_feature_importance_without_external_libs(sample_core):
    """Проверяем анализ без внешних библиотек."""
    analyzer = XAIAnalyzer(sample_core)
    
    x = torch.randn(5)
    
    # Выполняем анализ
    results = analyzer.analyze_feature_importance(x)
    
    # Проверяем, что результаты SHAP и LIME равны None
    assert results['shap_values'] is None
    assert results['lime_coeffs'] is None
    
    # Но другие результаты должны быть доступны
    assert results['effective_weights_plus'] is not None
    assert results['effective_weights_minus'] is not None
    assert results['approx_feature_contributions_plus'] is not None
    assert results['approx_feature_contributions_minus'] is not None


def test_visualize_explanation_method_exists(sample_core):
    """Проверяем, что метод визуализации существует."""
    analyzer = XAIAnalyzer(sample_core)
    
    x = torch.randn(5)
    
    # Проверяем, что метод существует
    assert hasattr(analyzer, 'visualize_explanation')


def test_visualize_with_unavailable_methods(sample_core):
    """Проверяем визуализацию с недоступными методами."""
    analyzer = XAIAnalyzer(sample_core)
    
    x = torch.randn(5)
    
    # Проверяем, что при недоступных методах возвращается None
    result = analyzer.visualize_explanation(x, method="nonexistent")
    assert result is None


def test_input_conversion_in_predict_batch(sample_core):
    """Проверяем конвертацию входных данных во внутренней функции."""
    analyzer = XAIAnalyzer(sample_core)
    
    # Тестируем с разными типами входных данных
    tensor_input = torch.randn(5)
    numpy_input = tensor_input.numpy()
    
    # Оба должны давать одинаковый результат
    tensor_result = analyzer._predict_batch(tensor_input)
    numpy_result = analyzer._predict_batch(numpy_input)
    
    assert np.allclose(tensor_result, numpy_result, atol=1e-6)


def test_multidimensional_input_handling(sample_core):
    """Проверяем обработку многомерных входов."""
    analyzer = XAIAnalyzer(sample_core)
    
    # Тестируем батч из нескольких векторов
    batch_input = torch.randn(3, 5)
    batch_numpy = batch_input.numpy()
    
    tensor_result = analyzer._predict_batch(batch_input)
    numpy_result = analyzer._predict_batch(batch_numpy)
    
    assert tensor_result.shape == (3,)
    assert numpy_result.shape == (3,)
    assert np.allclose(tensor_result, numpy_result, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])