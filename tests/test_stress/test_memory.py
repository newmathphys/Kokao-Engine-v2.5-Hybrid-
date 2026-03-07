"""
Stress tests: Memory and numerical stability.
Level 4 - Stress Tests
"""
import torch
import pytest
import math
from kokao import KokaoCore, CoreConfig, IntuitiveEtalonSystem


class TestMemoryStress:
    """Memory stress tests"""
    
    def test_many_etalons(self):
        """Work with many etalons"""
        system = IntuitiveEtalonSystem(CoreConfig(input_dim=50))
        
        # Add 1000 etalons
        for i in range(1000):
            system.learn_etalon(f"etalon_{i}", torch.randn(50))
        
        assert system.get_etalon_count() == 1000, \
            f"Should have 1000 etalons, got {system.get_etalon_count()}"
        
        # Recognition should work
        result = system.recognize(torch.randn(50), threshold=0.3)
        assert result is None or isinstance(result, str), \
            "Recognition should return None or string"
    
    def test_long_training_session(self):
        """Long training session"""
        core = KokaoCore(CoreConfig(input_dim=50))
        
        # 10000 training steps
        for i in range(10000):
            x = torch.randn(50)
            target = torch.rand(1).item() * 0.8 + 0.1
            core.train(x, target=target, lr=0.001)
            
            # Check every 1000 steps
            if i % 1000 == 0:
                assert not torch.isnan(core.w_plus).any(), \
                    f"w_plus has NaN at step {i}"
                assert not torch.isnan(core.w_minus).any(), \
                    f"w_minus has NaN at step {i}"
        
        # Final check
        w_plus, w_minus = core._get_effective_weights()
        assert torch.isfinite(w_plus).all(), "w_plus should be finite"
        assert torch.isfinite(w_minus).all(), "w_minus should be finite"
    
    def test_history_growth_limit(self):
        """History growth should be limited"""
        core = KokaoCore(CoreConfig(input_dim=10, max_history=100))
        
        # 1000 training steps
        for _ in range(1000):
            core.train(torch.randn(10), target=0.8, lr=0.01)
        
        # History should be limited
        assert len(core.history) <= 100, \
            f"History should be <= 100, got {len(core.history)}"
    
    def test_many_decoder_generations(self):
        """Many decoder generations"""
        from kokao import Decoder
        
        core = KokaoCore(CoreConfig(input_dim=20))
        decoder = Decoder(core)
        
        # 100 generations
        for _ in range(100):
            x = decoder.generate(S_target=0.5)
            assert x.shape == (20,), f"Expected shape (20,), got {x.shape}"
            assert torch.isfinite(x).all(), "Generated vector should be finite"
    
    def test_concurrent_core_operations(self):
        """Concurrent operations on multiple cores"""
        cores = [KokaoCore(CoreConfig(input_dim=20)) for _ in range(10)]
        
        # Train all cores
        for core in cores:
            for _ in range(100):
                core.train(torch.randn(20), target=0.5, lr=0.01)
        
        # All should work
        for i, core in enumerate(cores):
            s = core.signal(torch.randn(20))
            assert math.isfinite(s), f"Core {i} signal {s} should be finite"
