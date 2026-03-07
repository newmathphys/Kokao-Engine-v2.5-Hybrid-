"""Тесты для RAG модуля Kokao Engine."""
import pytest
import numpy as np
import torch
from unittest.mock import patch

from kokao import KokaoCore, CoreConfig
from kokao.rag import RAGModule


@pytest.fixture
def sample_core():
    """Создание тестового ядра."""
    config = CoreConfig(input_dim=10)
    return KokaoCore(config)


def test_rag_initialization():
    """Проверяем инициализацию RAG модуля."""
    config = CoreConfig(input_dim=10)
    core = KokaoCore(config)
    
    rag = RAGModule(core, embedding_dim=5)
    
    assert rag.core is core
    assert rag.embedding_dim == 5
    assert rag.get_total_documents() == 0


def test_add_document():
    """Проверяем добавление документа."""
    config = CoreConfig(input_dim=10)
    core = KokaoCore(config)
    rag = RAGModule(core, embedding_dim=4)
    
    doc_id = 1
    embedding = np.array([0.1, 0.2, 0.3, 0.4])
    metadata = {"title": "Test Doc"}
    
    rag.add_document(doc_id, embedding, metadata)
    
    assert rag.get_total_documents() == 1
    assert rag.get_document_metadata(doc_id) == metadata


def test_add_document_wrong_dimension():
    """Проверяем реакцию на эмбеддинг неправильной размерности."""
    config = CoreConfig(input_dim=10)
    core = KokaoCore(config)
    rag = RAGModule(core, embedding_dim=4)
    
    doc_id = 1
    wrong_embedding = np.array([0.1, 0.2])  # Только 2 элемента, а нужно 4
    
    with pytest.raises(ValueError, match="dimension.*does not match"):
        rag.add_document(doc_id, wrong_embedding)


def test_search_functionality():
    """Проверяем функцию поиска."""
    config = CoreConfig(input_dim=10)
    core = KokaoCore(config)
    rag = RAGModule(core, embedding_dim=3)
    
    # Добавляем несколько документов
    embeddings = [
        np.array([0.1, 0.2, 0.3]),
        np.array([0.4, 0.5, 0.6]),
        np.array([0.7, 0.8, 0.9])
    ]
    
    for i, emb in enumerate(embeddings):
        rag.add_document(i, emb)
    
    # Выполняем поиск
    query = np.array([0.3, 0.4, 0.5])
    results = rag.search(query, k=2)
    
    assert len(results) == 2
    assert all(isinstance(doc_id, int) for doc_id, _ in results)
    assert all(isinstance(distance, float) for _, distance in results)
    
    # Проверяем, что возвращаемые ID существуют
    returned_ids = [doc_id for doc_id, _ in results]
    assert all(doc_id in [0, 1, 2] for doc_id in returned_ids)


def test_search_returns_correct_k():
    """Проверяем, что поиск возвращает правильное количество результатов."""
    config = CoreConfig(input_dim=10)
    core = KokaoCore(config)
    rag = RAGModule(core, embedding_dim=2)
    
    # Добавляем 5 документов
    for i in range(5):
        emb = np.random.random(2)
        rag.add_document(i, emb)
    
    # Запрашиваем 3
    query = np.random.random(2)
    results = rag.search(query, k=3)
    
    assert len(results) == 3
    
    # Запрашиваем больше, чем есть
    results_all = rag.search(query, k=10)
    assert len(results_all) == 5  # Только столько, сколько есть


def test_signal_based_search():
    """Проверяем поиск на основе сигнатур."""
    config = CoreConfig(input_dim=3)
    core = KokaoCore(config)
    rag = RAGModule(core, embedding_dim=3)
    
    # Добавляем документы
    for i in range(3):
        emb = np.full(3, 0.1 * (i + 1))  # [0.1, 0.1, 0.1], [0.2, 0.2, 0.2], ...
        rag.add_document(i, emb)
    
    query = np.full(3, 0.15)  # Промежуточное значение
    results = rag.search_by_signal_similarity(query, k=2, top_docs_for_signal=3)
    
    assert len(results) == 2
    assert all(isinstance(doc_id, int) for doc_id, _ in results)
    assert all(isinstance(score, float) for _, score in results)


def test_get_document_metadata():
    """Проверяем получение метаданных документа."""
    config = CoreConfig(input_dim=10)
    core = KokaoCore(config)
    rag = RAGModule(core, embedding_dim=2)
    
    doc_id = 100
    metadata = {"author": "Test Author", "year": 2023}
    rag.add_document(doc_id, np.array([0.1, 0.2]), metadata)
    
    retrieved_metadata = rag.get_document_metadata(doc_id)
    assert retrieved_metadata == metadata
    
    # Проверяем несуществующий ID
    non_existent = rag.get_document_metadata(999)
    assert non_existent is None


def test_get_total_documents():
    """Проверяем подсчет общего количества документов."""
    config = CoreConfig(input_dim=10)
    core = KokaoCore(config)
    rag = RAGModule(core, embedding_dim=2)
    
    assert rag.get_total_documents() == 0
    
    for i in range(5):
        rag.add_document(i, np.random.random(2))
    
    assert rag.get_total_documents() == 5
    
    # После сброса
    rag.reset_index()
    assert rag.get_total_documents() == 0


def test_reset_index():
    """Проверяем сброс индекса."""
    config = CoreConfig(input_dim=10)
    core = KokaoCore(config)
    rag = RAGModule(core, embedding_dim=2)
    
    # Добавляем документы
    for i in range(3):
        rag.add_document(i, np.random.random(2), {"idx": i})
    
    assert rag.get_total_documents() == 3
    
    # Сбрасываем
    rag.reset_index()
    
    assert rag.get_total_documents() == 0
    
    # Проверяем, что старые ID больше не работают
    query = np.random.random(2)
    results = rag.search(query, k=1)
    assert len(results) == 0


@patch('kokao.rag.FAISS_AVAILABLE', False)
def test_faiss_not_available():
    """Проверяем поведение при отсутствии FAISS."""
    config = CoreConfig(input_dim=10)
    core = KokaoCore(config)
    
    with pytest.raises(ImportError, match="FAISS не установлен"):
        RAGModule(core, embedding_dim=5)


def test_empty_search():
    """Проверяем поиск в пустом индексе."""
    config = CoreConfig(input_dim=10)
    core = KokaoCore(config)
    rag = RAGModule(core, embedding_dim=3)
    
    query = np.array([0.1, 0.2, 0.3])
    results = rag.search(query, k=5)
    
    assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__])