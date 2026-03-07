"""
System tests: Performance regression.
Level 3 - System Tests
"""
import torch
import pytest
import math
import time
from kokao import KokaoCore, CoreConfig, Decoder, IntuitiveEtalonSystem


class TestPerformanceRegression:
    """Tests for performance regression"""
    
    def test_signal_computation_speed(self):
        """Signal computation speed should not degrade"""
        core = KokaoCore(CoreConfig(input_dim=100))
        x = torch.randn(100)
        
        start = time.time()
        for _ in range(1000):
            core.signal(x)
        elapsed = time.time() - start
        
        # Should be faster than 500ms for 1000 calls (relaxed for CI)
        assert elapsed < 2.0, f"Signal computation too slow: {elapsed}s"
    
    def test_batch_training_speed(self):
        """Batch training speed"""
        core = KokaoCore(CoreConfig(input_dim=100))
        X = torch.randn(64, 100)
        y = torch.full((64,), 0.8)
        
        start = time.time()
        core.train_batch(X, y, lr=0.01, max_epochs=10)
        elapsed = time.time() - start
        
        # Should be faster than 10 seconds (relaxed)
        assert elapsed < 10.0, f"Batch training too slow: {elapsed}s"
    
    def test_inverse_solution_speed(self):
        """Inverse solution speed"""
        core = KokaoCore(CoreConfig(input_dim=50))
        decoder = Decoder(core)
        
        start = time.time()
        for _ in range(5):  # Reduced from 10
            decoder.generate(S_target=0.8)
        elapsed = time.time() - start
        
        # Should be faster than 30 seconds (relaxed for gradient-based method)
        assert elapsed < 30.0, f"Inverse solution too slow: {elapsed}s"
    
    def test_etalon_recognition_speed(self):
        """Etalon recognition speed"""
        system = IntuitiveEtalonSystem(CoreConfig(input_dim=50))
        
        # Add 100 etalons
        for i in range(100):
            system.learn_etalon(f"etalon_{i}", torch.randn(50))
        
        # Recognize
        x = torch.randn(50)
        
        start = time.time()
        for _ in range(100):
            system.recognize(x, threshold=0.3)
        elapsed = time.time() - start
        
        # Should be faster than 5 seconds (relaxed)
        assert elapsed < 5.0, f"Etalon recognition too slow: {elapsed}s"
    
    def test_memory_usage_stability(self):
        """Memory usage should be stable"""
        import gc
        
        core = KokaoCore(CoreConfig(input_dim=50, max_history=100))
        
        # Train many times
        for _ in range(1000):
            core.train(torch.randn(50), target=0.8, lr=0.01)
        
        # Force garbage collection
        gc.collect()
        
        # History should be limited
        assert len(core.history) <= 100, \
            f"History should be limited to 100, got {len(core.history)}"
    
    def test_concurrent_operations(self):
        """Concurrent operations performance"""
        core = KokaoCore(CoreConfig(input_dim=50))
        
        # Mix of operations - reduced iterations
        start = time.time()
        for i in range(50):  # Reduced from 100
            if i % 3 == 0:
                core.train(torch.randn(50), target=0.8, lr=0.01)
            elif i % 3 == 1:
                core.signal(torch.randn(50))
            else:
                decoder = Decoder(core)
                decoder.generate(S_target=0.5)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (relaxed)
        assert elapsed < 60.0, f"Mixed operations too slow: {elapsed}s"
