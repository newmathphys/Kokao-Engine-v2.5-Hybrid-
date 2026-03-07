"""Тесты для интеграции с LangChain."""
import pytest
import json
import torch
from unittest.mock import Mock, patch

from kokao import KokaoCore, CoreConfig
from kokao.integrations.langchain import (
    LangChainKokaoAdapter, 
    KokaoSignalTool, 
    KokaoInversionTool, 
    KokaoTrainTool
)


@pytest.fixture
def sample_core():
    """Создание тестового ядра."""
    config = CoreConfig(input_dim=5)
    return KokaoCore(config)


def test_langchain_adapter_initialization(sample_core):
    """Проверяем инициализацию адаптера LangChain."""
    # Проверяем, что при отсутствии LangChain выбрасывается ошибка
    with patch('kokao.integrations.langchain.LANGCHAIN_AVAILABLE', False):
        with pytest.raises(ImportError, match="LangChain не установлен"):
            LangChainKokaoAdapter(sample_core)


def test_kokao_signal_tool_creation(sample_core):
    """Проверяем создание инструмента сигнала."""
    tool = KokaoSignalTool(core=sample_core)
    
    assert tool.name == "kokao_signal_calculator"
    assert "сигнал" in tool.description.lower()


def test_kokao_signal_tool_execution(sample_core):
    """Проверяем выполнение инструмента сигнала."""
    tool = KokaoSignalTool(core=sample_core)
    
    # Создаем тестовый вектор
    test_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    vector_json = json.dumps(test_vector)
    
    result = tool._run(vector_json)
    
    # Проверяем, что результат содержит информацию о сигнале
    assert "сигнал" in result.lower() or "signal" in result.lower()
    assert str(test_vector) in result


def test_kokao_signal_tool_invalid_json(sample_core):
    """Проверяем реакцию инструмента сигнала на невалидный JSON."""
    tool = KokaoSignalTool(core=sample_core)
    
    invalid_json = "{invalid json}"
    
    result = tool._run(invalid_json)
    
    assert "ошибка" in result.lower() or "error" in result.lower()


def test_kokao_signal_tool_wrong_dimension(sample_core):
    """Проверяем реакцию инструмента сигнала на неправильную размерность."""
    tool = KokaoSignalTool(core=sample_core)
    
    # Вектор неправильной размерности (должен быть 5)
    wrong_vector = [0.1, 0.2]  # Только 2 элемента
    vector_json = json.dumps(wrong_vector)
    
    result = tool._run(vector_json)
    
    assert "размерность" in result.lower() or "dimension" in result.lower()


def test_kokao_inversion_tool_creation(sample_core):
    """Проверяем создание инструмента инверсии."""
    tool = KokaoInversionTool(core=sample_core)
    
    assert tool.name == "kokao_signal_inverter"
    assert "генерирует" in tool.description.lower() or "generates" in tool.description.lower()


def test_kokao_inversion_tool_execution(sample_core):
    """Проверяем выполнение инструмента инверсии."""
    tool = KokaoInversionTool(core=sample_core)
    
    target_signal = "0.5"
    
    result = tool._run(target_signal)
    
    # Проверяем, что результат содержит информацию о генерации
    assert "сгенерирован" in result.lower() or "generated" in result.lower()
    assert "вектор" in result.lower() or "vector" in result.lower()


def test_kokao_inversion_tool_invalid_signal(sample_core):
    """Проверяем реакцию инструмента инверсии на невалидный сигнал."""
    tool = KokaoInversionTool(core=sample_core)
    
    invalid_signal = "not_a_number"
    
    result = tool._run(invalid_signal)
    
    assert "ошибка" in result.lower() or "error" in result.lower()


def test_kokao_train_tool_creation(sample_core):
    """Проверяем создание инструмента обучения."""
    tool = KokaoTrainTool(core=sample_core)
    
    assert tool.name == "kokao_trainer"
    assert "обучает" in tool.description.lower() or "trains" in tool.description.lower()


def test_kokao_train_tool_execution(sample_core):
    """Проверяем выполнение инструмента обучения."""
    tool = KokaoTrainTool(core=sample_core)
    
    # Подготовим данные для обучения
    train_data = {
        "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
        "target": 0.8
    }
    train_json = json.dumps(train_data)
    
    result = tool._run(train_json)
    
    # Проверяем, что результат содержит информацию об обучении
    assert "обучение" in result.lower() or "training" in result.lower()
    assert "потеря" in result.lower() or "loss" in result.lower()


def test_kokao_train_tool_missing_fields(sample_core):
    """Проверяем реакцию инструмента обучения на отсутствующие поля."""
    tool = KokaoTrainTool(core=sample_core)
    
    # Данные без обязательных полей
    incomplete_data = {
        "vector": [0.1, 0.2, 0.3, 0.4, 0.5]
        # Нет поля "target"
    }
    train_json = json.dumps(incomplete_data)
    
    result = tool._run(train_json)
    
    assert "ошибка" in result.lower() or "error" in result.lower()


def test_get_tools_list(sample_core):
    """Проверяем получение списка инструментов."""
    with patch('kokao.integrations.langchain.LANGCHAIN_AVAILABLE', True):
        adapter = LangChainKokaoAdapter(sample_core)
        
        tools = adapter.get_tools()
        
        # Проверяем, что возвращается 3 инструмента
        assert len(tools) == 3
        
        # Проверяем, что все инструменты правильного типа
        tool_names = [tool.name for tool in tools]
        expected_names = ["kokao_signal_calculator", "kokao_signal_inverter", "kokao_trainer"]
        
        for name in expected_names:
            assert name in tool_names


def test_get_tool_by_name(sample_core):
    """Проверяем получение инструмента по имени."""
    with patch('kokao.integrations.langchain.LANGCHAIN_AVAILABLE', True):
        adapter = LangChainKokaoAdapter(sample_core)
        
        # Получаем инструмент по имени
        signal_tool = adapter.get_tool_by_name("kokao_signal_calculator")
        inversion_tool = adapter.get_tool_by_name("kokao_signal_inverter")
        train_tool = adapter.get_tool_by_name("kokao_trainer")
        
        assert signal_tool is not None
        assert inversion_tool is not None
        assert train_tool is not None
        
        # Проверяем, что несуществующий инструмент возвращает None
        nonexistent_tool = adapter.get_tool_by_name("nonexistent_tool")
        assert nonexistent_tool is None


def test_async_methods_exist(sample_core):
    """Проверяем, что асинхронные методы существуют."""
    signal_tool = KokaoSignalTool(core=sample_core)
    inversion_tool = KokaoInversionTool(core=sample_core)
    train_tool = KokaoTrainTool(core=sample_core)
    
    # Проверяем, что асинхронные методы существуют
    assert hasattr(signal_tool, '_arun')
    assert hasattr(inversion_tool, '_arun')
    assert hasattr(train_tool, '_arun')


if __name__ == "__main__":
    pytest.main([__file__])