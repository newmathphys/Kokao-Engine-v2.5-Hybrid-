"""
Integration tests: RAG + Core.
Level 2 - Integration Tests
"""
import torch
import pytest
from kokao import KokaoCore, CoreConfig, Decoder, RAGModule


class TestRAGCoreIntegration:
    """Tests for RAG and Core integration"""
    
    def test_search_by_signal_similarity(self):
        """Search documents by signal similarity"""
        core = KokaoCore(CoreConfig(input_dim=10))
        rag = RAGModule(core, embedding_dim=10)
        
        # Add documents
        for i in range(5):
            emb = torch.randn(10).numpy()
            rag.add_document(i, emb, metadata={"class": i})
        
        # Search
        query = torch.randn(10).numpy()
        results = rag.search_by_signal_similarity(query, k=3)
        
        assert len(results) == 3, f"Should return 3 results, got {len(results)}"
        assert all(isinstance(doc_id, int) for doc_id, _ in results), \
            "All results should be integers"
    
    def test_rag_with_trained_core(self):
        """RAG with trained core"""
        core = KokaoCore(CoreConfig(input_dim=10))
        
        # Train core
        core.train_batch(torch.randn(32, 10), torch.full((32,), 0.8))
        
        rag = RAGModule(core, embedding_dim=10)
        
        # Add documents with different signals
        decoder = Decoder(core)
        for target in [0.2, 0.5, 0.8]:
            emb = decoder.generate(S_target=target).numpy()
            rag.add_document(int(target * 10), emb)
        
        # Search should find closest by signal
        query = decoder.generate(S_target=0.75).numpy()
        results = rag.search_by_signal_similarity(query, k=1)
        
        assert len(results) == 1, f"Should return 1 result, got {len(results)}"
    
    def test_rag_signal_ranking(self):
        """RAG signal-based ranking"""
        core = KokaoCore(CoreConfig(input_dim=10))
        rag = RAGModule(core, embedding_dim=10)
        
        # Add documents with known signals
        decoder = Decoder(core)
        for target in [0.3, 0.6, 0.9]:
            emb = decoder.generate(S_target=target).numpy()
            rag.add_document(int(target * 10), emb)
        
        # Query with known signal
        query = decoder.generate(S_target=0.6).numpy()
        results = rag.search_by_signal_similarity(query, k=3)
        
        # Closest signal should be first
        assert len(results) == 3, f"Should return 3 results"
        # Document with signal 0.6 should be closest to query with signal 0.6
        closest_id = results[0][0]
        assert closest_id == 6, f"Closest should be doc 6, got {closest_id}"
    
    def test_rag_empty_index(self):
        """RAG with empty index"""
        core = KokaoCore(CoreConfig(input_dim=10))
        rag = RAGModule(core, embedding_dim=10)
        
        # Search empty index
        query = torch.randn(10).numpy()
        results = rag.search(query, k=5)
        
        assert len(results) == 0, f"Should return 0 results for empty index"
    
    def test_rag_add_and_search(self):
        """RAG add and search workflow"""
        core = KokaoCore(CoreConfig(input_dim=10))
        rag = RAGModule(core, embedding_dim=10)
        
        # Add multiple documents
        for i in range(10):
            emb = torch.randn(10).numpy()
            rag.add_document(i, emb, metadata={"id": i})
        
        assert rag.get_total_documents() == 10, \
            f"Should have 10 documents, got {rag.get_total_documents()}"
        
        # Search
        query = torch.randn(10).numpy()
        results = rag.search(query, k=5)
        
        assert len(results) == 5, f"Should return 5 results"
        
        # Get metadata
        for doc_id, _ in results:
            metadata = rag.get_document_metadata(doc_id)
            assert "id" in metadata, "Metadata should contain 'id'"
    
    def test_rag_reset_index(self):
        """RAG reset index"""
        core = KokaoCore(CoreConfig(input_dim=10))
        rag = RAGModule(core, embedding_dim=10)
        
        # Add documents
        for i in range(5):
            rag.add_document(i, torch.randn(10).numpy())
        
        assert rag.get_total_documents() == 5
        
        # Reset
        rag.reset_index()
        
        assert rag.get_total_documents() == 0, "Should have 0 documents after reset"
    
    def test_rag_search_k_larger_than_documents(self):
        """RAG search with k > num_documents"""
        core = KokaoCore(CoreConfig(input_dim=10))
        rag = RAGModule(core, embedding_dim=10)
        
        # Add 3 documents
        for i in range(3):
            rag.add_document(i, torch.randn(10).numpy())
        
        # Search for 10
        query = torch.randn(10).numpy()
        results = rag.search(query, k=10)
        
        # Should return all 3
        assert len(results) == 3, f"Should return 3 results, got {len(results)}"
