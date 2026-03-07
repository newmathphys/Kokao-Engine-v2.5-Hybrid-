"""Интеграция Kokao Engine с LangChain."""
import logging
from typing import Any, Dict, Optional, Type, List
from pydantic import BaseModel, Field, ConfigDict

try:
    from langchain.tools import BaseTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object  # Заглушка для базового класса

from ..core import KokaoCore
from ..decoder import Decoder

logger = logging.getLogger(__name__)


class KokaoSignalTool(BaseTool):
    """Инструмент LangChain для вычисления сигнала KokaoCore."""

    name: str = "kokao_signal_calculator"
    description: str = "Вычисляет скалярный сигнал для заданного вектора с помощью KokaoCore. Вход: вектор признаков в формате JSON."

    core: KokaoCore
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, vector_json: str) -> str:
        """
        Выполнение инструмента.
        
        Args:
            vector_json: JSON строка с вектором признаков
            
        Returns:
            Строка с вычисленным сигналом
        """
        import json
        import torch
        
        try:
            # Парсим JSON
            vector_data = json.loads(vector_json)
            
            if not isinstance(vector_data, list):
                raise ValueError("Вектор должен быть представлен в формате JSON-массива")
            
            # Преобразуем в тензор
            vector = torch.tensor(vector_data, dtype=torch.float32)
            
            # Проверяем размерность
            if vector.shape[0] != self.core.config.input_dim:
                raise ValueError(f"Размерность вектора {vector.shape[0]} не соответствует ожидаемой {self.core.config.input_dim}")
            
            # Вычисляем сигнал
            signal = self.core.signal(vector)
            
            return f"Сигнал для вектора {vector_data}: {signal:.6f}"
            
        except json.JSONDecodeError as e:
            return f"Ошибка парсинга JSON: {str(e)}"
        except Exception as e:
            return f"Ошибка вычисления сигнала: {str(e)}"
    
    async def _arun(self, vector_json: str) -> str:
        """Асинхронная версия run."""
        # В этой реализации синхронная и асинхронная версии одинаковы
        return self._run(vector_json)


class KokaoInversionTool(BaseTool):
    """Инструмент LangChain для генерации вектора по целевому сигналу."""

    name: str = "kokao_signal_inverter"
    description: str = "Генерирует вектор признаков, дающий заданный целевой сигнал с помощью KokaoCore. Вход: целевой сигнал (число)."

    core: KokaoCore
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, core: KokaoCore, **kwargs):
        # Передаем core в kwargs для Pydantic
        kwargs['core'] = core
        super().__init__(**kwargs)
        # decoder создаем после инициализации, не как поле Pydantic
        object.__setattr__(self, 'decoder', Decoder(core))
    
    def _run(self, target_signal: str) -> str:
        """
        Выполнение инструмента.
        
        Args:
            target_signal: Целевой сигнал в виде строки
            
        Returns:
            Строка с сгенерированным вектором
        """
        import torch
        
        try:
            # Преобразуем строку в число
            target = float(target_signal)
            
            # Генерируем вектор
            generated_vector = self.decoder.generate(target)
            
            # Вычисляем сигнал для проверки
            actual_signal = self.core.signal(generated_vector)
            
            return f"Сгенерирован вектор: {generated_vector.tolist()}, дающий сигнал: {actual_signal:.6f} (цель: {target})"
            
        except ValueError as e:
            return f"Ошибка преобразования сигнала: {str(e)}"
        except Exception as e:
            return f"Ошибка генерации вектора: {str(e)}"
    
    async def _arun(self, target_signal: str) -> str:
        """Асинхронная версия run."""
        # В этой реализации синхронная и асинхронная версии одинаковы
        return self._run(target_signal)


class KokaoTrainTool(BaseTool):
    """Инструмент LangChain для обучения ядра KokaoCore."""

    name: str = "kokao_trainer"
    description: str = "Обучает KokaoCore на заданном векторе и целевом сигналу. Вход: JSON с полями 'vector' и 'target'."

    core: KokaoCore
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, train_data_json: str) -> str:
        """
        Выполнение инструмента.
        
        Args:
            train_data_json: JSON строка с вектором и целевым значением
            
        Returns:
            Строка с информацией о результате обучения
        """
        import json
        import torch
        
        try:
            # Парсим JSON
            train_data = json.loads(train_data_json)
            
            if not isinstance(train_data, dict):
                raise ValueError("Данные обучения должны быть представлены в формате JSON-объекта")
            
            if 'vector' not in train_data or 'target' not in train_data:
                raise ValueError("JSON должен содержать поля 'vector' и 'target'")
            
            # Извлекаем вектор и цель
            vector_data = train_data['vector']
            target = float(train_data['target'])
            
            if not isinstance(vector_data, list):
                raise ValueError("Поле 'vector' должно быть JSON-массивом")
            
            # Преобразуем в тензор
            vector = torch.tensor(vector_data, dtype=torch.float32)
            
            # Проверяем размерность
            if vector.shape[0] != self.core.config.input_dim:
                raise ValueError(f"Размерность вектора {vector.shape[0]} не соответствует ожидаемой {self.core.config.input_dim}")
            
            # Обучаем
            initial_signal = self.core.signal(vector)
            loss_before = (initial_signal - target) ** 2
            loss_after = self.core.train(vector, target, lr=train_data.get('lr', 0.01), mode=train_data.get('mode', 'gradient'))
            final_signal = self.core.signal(vector)
            
            return f"Обучение завершено. Потеря до: {loss_before:.6f}, после: {loss_after:.6f}. Сигнал: {initial_signal:.6f} -> {final_signal:.6f}"
            
        except json.JSONDecodeError as e:
            return f"Ошибка парсинга JSON: {str(e)}"
        except Exception as e:
            return f"Ошибка обучения: {str(e)}"
    
    async def _arun(self, train_data_json: str) -> str:
        """Асинхронная версия run."""
        return self._run(train_data_json)


class LangChainKokaoAdapter:
    """Адаптер для создания набора инструментов LangChain для KokaoCore."""
    
    def __init__(self, core: KokaoCore):
        """
        Инициализация адаптера.
        
        Args:
            core: Экземпляр KokaoCore
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain не установлен. Установите с помощью: pip install langchain"
            )
        
        self.core = core
    
    def get_tools(self) -> List[BaseTool]:
        """
        Получение списка инструментов для LangChain.
        
        Returns:
            Список инструментов LangChain
        """
        tools = [
            KokaoSignalTool(core=self.core),
            KokaoInversionTool(core=self.core),
            KokaoTrainTool(core=self.core)
        ]
        
        logger.info(f"Created {len(tools)} LangChain tools for KokaoCore")
        return tools
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """
        Получение инструмента по имени.
        
        Args:
            name: Имя инструмента
            
        Returns:
            Инструмент или None, если не найден
        """
        tools = self.get_tools()
        for tool in tools:
            if tool.name == name:
                return tool
        return None