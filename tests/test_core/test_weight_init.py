"""
Tests for weight initialization in KokaoCore.
Level 1 - Basic Unit Tests
"""
import torch
import pytest
from kokao import KokaoCore, CoreConfig


class TestWeightInitialization:
    """Tests for weight initialization"""
    
    def test_weights_initialized_positive(self):
        """Weights should be positive after softplus"""
        core = KokaoCore(CoreConfig(input_dim=10))
        w_plus, w_minus = core._get_effective_weights()
        assert (w_plus > 0).all(), "w_plus should be positive"
        assert (w_minus > 0).all(), "w_minus should be positive"
    
    def test_weights_sum_equals_target(self):
        """Weight sums should equal target_sum"""
        config = CoreConfig(input_dim=10, target_sum=50.0)
        core = KokaoCore(config)
        w_plus, w_minus = core._get_effective_weights()
        assert torch.isclose(w_plus.sum(), torch.tensor(50.0), atol=1e-5), \
            f"w_plus sum {w_plus.sum().item()} != 50.0"
        assert torch.isclose(w_minus.sum(), torch.tensor(50.0), atol=1e-5), \
            f"w_minus sum {w_minus.sum().item()} != 50.0"
    
    def test_weights_reproducible_with_seed(self):
        """Same seed should produce same weights"""
        config1 = CoreConfig(input_dim=10, seed=42)
        config2 = CoreConfig(input_dim=10, seed=42)
        core1 = KokaoCore(config1)
        core2 = KokaoCore(config2)
        assert torch.allclose(core1.w_plus, core2.w_plus), \
            "w_plus should be same with same seed"
        assert torch.allclose(core1.w_minus, core2.w_minus), \
            "w_minus should be same with same seed"
    
    def test_different_seeds_different_weights(self):
        """Different seeds should produce different weights"""
        config1 = CoreConfig(input_dim=10, seed=42)
        config2 = CoreConfig(input_dim=10, seed=123)
        core1 = KokaoCore(config1)
        core2 = KokaoCore(config2)
        assert not torch.allclose(core1.w_plus, core2.w_plus), \
            "w_plus should differ with different seeds"
        assert not torch.allclose(core1.w_minus, core2.w_minus), \
            "w_minus should differ with different seeds"
    
    def test_weight_distribution_statistics(self):
        """Weight distribution statistics"""
        core = KokaoCore(CoreConfig(input_dim=100))
        w_plus, w_minus = core._get_effective_weights()
        assert w_plus.mean() > 0, "w_plus mean should be positive"
        assert w_plus.std() > 0, "w_plus should have variance"
        assert torch.isfinite(w_plus).all(), "w_plus should be finite"
        assert torch.isfinite(w_minus).all(), "w_minus should be finite"
    
    def test_target_sum_100_default(self):
        """Default target_sum should be 100"""
        core = KokaoCore(CoreConfig(input_dim=10))
        w_plus, w_minus = core._get_effective_weights()
        assert torch.isclose(w_plus.sum(), torch.tensor(100.0), atol=1e-5)
        assert torch.isclose(w_minus.sum(), torch.tensor(100.0), atol=1e-5)
    
    def test_weights_after_normalization(self):
        """Weights should maintain target_sum after multiple normalizations"""
        core = KokaoCore(CoreConfig(input_dim=10, target_sum=75.0))
        
        # Multiple normalizations
        for _ in range(10):
            core._normalize()
        
        w_plus, w_minus = core._get_effective_weights()
        assert torch.isclose(w_plus.sum(), torch.tensor(75.0), atol=1e-5)
        assert torch.isclose(w_minus.sum(), torch.tensor(75.0), atol=1e-5)
