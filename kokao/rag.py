"""Модуль RAG для интеграции с FAISS в Kokao Engine."""
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .core import KokaoCore

logger = logging.getLogger(__name__)


class RAGModule:
    """
    Модуль RAG (Retrieval Augmented Generation) для поиска документов 
    по эмбеддингам с использованием KokaoCore сигнала как меры близости.
    """
    
    def __init__(self, core: KokaoCore, embedding_dim: int = 768):
        """
        Инициализация RAG модуля.
        
        Args:
            core: Экземпляр KokaoCore для вычисления сигналов
            embedding_dim: Размерность эмбеддингов
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS не установлен. Установите с помощью: pip install faiss-cpu"
            )
        
        self.core = core
        self.embedding_dim = embedding_dim
        
        # Инициализируем FAISS индекс
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.id_map: Dict[int, Any] = {}  # Карта ID -> метаданные
        self.doc_ids: List[int] = []      # Список ID документов
        self.doc_embeddings: List[np.ndarray] = []  # Список эмбеддингов
        
        logger.info(f"RAGModule initialized with embedding dimension {embedding_dim}")
    
    def add_document(self, doc_id: int, embedding: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """
        Добавление документа в индекс.
        
        Args:
            doc_id: Уникальный ID документа
            embedding: Эмбеддинг документа (numpy array)
            metadata: Опциональные метаданные документа
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension {embedding.shape[0]} does not match expected {self.embedding_dim}")
        
        # Сохраняем эмбеддинг и метаданные
        self.doc_ids.append(doc_id)
        self.doc_embeddings.append(embedding)
        
        # Добавляем в FAISS индекс
        embedding_float32 = embedding.astype(np.float32)
        self.index.add(embedding_float32.reshape(1, -1))
        
        # Сохраняем метаданные
        self.id_map[doc_id] = metadata or {}
        
        logger.debug(f"Added document {doc_id} to index")
    
    def build_index(self) -> None:
        """
        Построение индекса (если используется индекс с обучением).
        Для IndexFlatL2 не требуется, но добавлен для совместимости.
        """
        logger.info("Index already built (using IndexFlatL2)")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Поиск k ближайших документов к запросу.
        
        Args:
            query_embedding: Эмбеддинг запроса
            k: Количество документов для возврата
            
        Returns:
            Список пар (doc_id, distance) отсортированный по расстоянию
        """
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[0]} does not match expected {self.embedding_dim}")
        
        if k > len(self.doc_ids):
            logger.warning(f"Requested k={k} > available documents={len(self.doc_ids)}, using {len(self.doc_ids)}")
            k = len(self.doc_ids)
        
        if k == 0:
            return []
        
        # Выполняем поиск в FAISS
        query_float32 = query_embedding.astype(np.float32)
        distances, indices = self.index.search(query_float32.reshape(1, -1), k)
        
        # Преобразуем результаты
        results = []
        for i in range(k):
            idx = indices[0][i]
            if idx < len(self.doc_ids):  # Проверяем, что индекс валиден
                doc_id = self.doc_ids[idx]
                distance = float(distances[0][i])
                results.append((doc_id, distance))
        
        logger.debug(f"Found {len(results)} documents for query")
        return results
    
    def search_by_signal_similarity(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5, 
        top_docs_for_signal: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Поиск документов с использованием сигнала KokaoCore как меры близости.
        Сначала находим топ-N ближайших по L2, затем ранжируем по разнице сигналов.
        
        Args:
            query_embedding: Эмбеддинг запроса
            k: Количество документов для возврата
            top_docs_for_signal: Количество документов для рассмотрения с сигналом
            
        Returns:
            Список пар (doc_id, signal_difference_score) отсортированный по score (меньше = ближе)
        """
        if not hasattr(self.core, 'signal'):
            raise ValueError("Core must have signal method for signal-based similarity")
        
        # Сначала получаем топ документов по L2 расстоянию
        l2_results = self.search(query_embedding, k=max(k, top_docs_for_signal))
        
        # Вычисляем сигнал для эмбеддинга запроса
        query_tensor = torch.from_numpy(query_embedding).float()
        query_signal = self.core.signal(query_tensor)
        
        # Ранжируем по разнице сигналов
        signal_scores = []
        for doc_id, _ in l2_results[:top_docs_for_signal]:
            idx = self.doc_ids.index(doc_id)
            doc_embedding = self.doc_embeddings[idx]
            doc_tensor = torch.from_numpy(doc_embedding).float()
            doc_signal = self.core.signal(doc_tensor)
            
            # Используем абсолютную разницу сигналов как меру близости
            signal_diff = abs(query_signal - doc_signal)
            signal_scores.append((doc_id, signal_diff))
        
        # Сортируем по разнице сигналов (меньше = ближе)
        signal_scores.sort(key=lambda x: x[1])
        
        # Возвращаем топ k
        return signal_scores[:k]
    
    def get_document_metadata(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Получение метаданных документа по ID.
        
        Args:
            doc_id: ID документа
            
        Returns:
            Метаданные документа или None, если не найден
        """
        return self.id_map.get(doc_id)
    
    def get_total_documents(self) -> int:
        """
        Получение общего количества документов в индексе.
        
        Returns:
            Количество документов
        """
        return len(self.doc_ids)
    
    def reset_index(self) -> None:
        """
        Сброс индекса и очистка всех документов.
        """
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.id_map.clear()
        self.doc_ids.clear()
        self.doc_embeddings.clear()
        
        logger.info("RAG index reset")