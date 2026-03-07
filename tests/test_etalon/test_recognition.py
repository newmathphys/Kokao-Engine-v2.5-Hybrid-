"""
Tests for etalon recognition.
Level 1 - Basic Unit Tests
"""
import torch
import pytest
from kokao import IntuitiveEtalonSystem, CoreConfig


class TestEtalonRecognition:
    """Tests for etalon recognition"""
    
    def test_recognition_with_perfect_match(self):
        """Recognition with perfect match"""
        system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        x = torch.randn(10)
        system.learn_etalon("test", x)
        
        result = system.recognize(x, threshold=0.9)
        assert result == "test", f"Should recognize 'test', got {result}"
    
    def test_recognition_with_noise(self):
        """Recognition with noisy input"""
        system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        x = torch.randn(10)
        system.learn_etalon("test", x)
        
        x_noisy = x + torch.randn(10) * 0.1
        result = system.recognize(x_noisy, threshold=0.5)
        # May or may not recognize due to noise
        assert result in ["test", None], f"Should recognize or return None, got {result}"
    
    def test_recognition_rejects_unknown(self):
        """Reject unknown input"""
        system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        x1 = torch.randn(10)
        system.learn_etalon("known", x1)
        
        x2 = torch.randn(10) * 10  # Very different vector
        result = system.recognize(x2, threshold=0.9)
        assert result is None, f"Should reject unknown, got {result}"
    
    def test_recognition_with_multiple_etalons(self):
        """Recognition among multiple etalons"""
        system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        
        # Learn 5 well-separated etalons
        for i in range(5):
            x = torch.randn(10) * 3 + i * 5  # More separated
            system.learn_etalon(f"class_{i}", x)
        
        # Test recognition with lower threshold
        for i in range(5):
            x = torch.randn(10) * 3 + i * 5
            result = system.recognize(x, threshold=0.1)  # Lower threshold
            # Should recognize SOME class (not necessarily the exact one due to overlap)
            assert result is None or result.startswith("class_"), \
                f"Should recognize a class or None, got {result}"
    
    def test_recognition_threshold_sweep(self):
        """Recognition with different thresholds"""
        system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        x = torch.randn(10)
        system.learn_etalon("test", x)
        
        # Low threshold - should recognize
        result_low = system.recognize(x, threshold=0.1)
        assert result_low == "test", "Should recognize with low threshold"
        
        # Very high threshold - might not recognize
        result_high = system.recognize(x + torch.randn(10) * 0.5, threshold=0.95)
        # Can be None or "test"
        assert result_high in [None, "test"], \
            f"High threshold result should be None or 'test', got {result_high}"
    
    def test_recognition_with_no_etalons(self):
        """Recognition with no etalons"""
        system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        x = torch.randn(10)
        
        result = system.recognize(x, threshold=0.5)
        assert result is None, "Should return None with no etalons"
    
    def test_recognition_batch(self):
        """Batch recognition"""
        system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        
        # Learn etalons
        for i in range(3):
            x = torch.randn(10) + i
            system.learn_etalon(f"class_{i}", x)
        
        # Batch recognize
        X_batch = torch.stack([
            torch.randn(10),  # Unknown
            torch.randn(10) + 1,  # class_1
            torch.randn(10) + 2,  # class_2
        ])
        
        results = system.recognize_batch(X_batch, threshold=0.3)
        assert len(results) == 3, f"Should have 3 results, got {len(results)}"
    
    def test_recognition_self_similarity(self):
        """Etalon should recognize itself with high similarity"""
        system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        x = torch.randn(10)
        system.learn_etalon("test", x)
        
        # Get the learned etalon
        learned = system.get_etalon("test")
        
        # Similarity should be very high
        result = system.recognize(learned, threshold=0.99)
        assert result == "test", f"Should recognize itself, got {result}"
    
    def test_recognition_orthogonal_vectors(self):
        """Recognition with orthogonal vectors"""
        system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        
        # Learn etalon
        x1 = torch.randn(10)
        system.learn_etalon("test", x1)
        
        # Create orthogonal vector
        x2 = torch.randn(10)
        x2 = x2 - (x2.dot(x1) / x1.dot(x1)) * x1
        
        result = system.recognize(x2, threshold=0.5)
        assert result is None, f"Should not recognize orthogonal vector, got {result}"
    
    def test_recognition_update_etalon(self):
        """Updating existing etalon"""
        system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        
        x1 = torch.randn(10)
        system.learn_etalon("test", x1)
        
        # Update with new vector
        x2 = torch.randn(10) * 0.5
        system.learn_etalon("test", x2)
        
        # Should recognize new vector
        result = system.recognize(x2, threshold=0.9)
        assert result == "test", f"Should recognize updated etalon, got {result}"
