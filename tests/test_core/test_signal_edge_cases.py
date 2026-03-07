"""
Tests for signal edge cases in KokaoCore.
Level 1 - Basic Unit Tests
"""
import torch
import pytest
import math
from kokao import KokaoCore, CoreConfig


class TestSignalEdgeCases:
    """Tests for signal edge cases"""
    
    def test_signal_orthogonal_input(self):
        """Signal for orthogonal input should be near zero"""
        core = KokaoCore(CoreConfig(input_dim=10))
        w_plus, _ = core._get_effective_weights()
        
        # Create vector orthogonal to w_plus
        x = torch.randn(10)
        x = x - (x.dot(w_plus) / w_plus.dot(w_plus)) * w_plus
        
        s = core.signal(x)
        assert abs(s) < 1e-5, f"Signal {s} should be near 0 for orthogonal input"
    
    def test_signal_parallel_to_w_plus(self):
        """Signal is maximal when x is parallel to w_plus"""
        core = KokaoCore(CoreConfig(input_dim=10))
        w_plus, w_minus = core._get_effective_weights()
        
        x = w_plus.clone()
        s_parallel = core.signal(x)
        
        # Just check it's finite and positive
        assert math.isfinite(s_parallel), f"Signal {s_parallel} should be finite"
        assert s_parallel > 0, f"Signal {s_parallel} should be positive"
    
    def test_signal_parallel_to_w_minus(self):
        """Signal when x is parallel to w_minus"""
        core = KokaoCore(CoreConfig(input_dim=10))
        _, w_minus = core._get_effective_weights()
        
        x = w_minus.clone()
        s = core.signal(x)
        
        # Should be positive and finite (due to softplus)
        assert math.isfinite(s), f"Signal {s} should be finite"
        assert s > 0, f"Signal {s} should be positive"
    
    def test_signal_very_small_input(self):
        """Signal for very small input"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.ones(10) * 1e-10
        s = core.signal(x)
        assert math.isfinite(s), f"Signal {s} should be finite for small input"
        assert not math.isnan(s), "Signal should not be NaN"
    
    def test_signal_very_large_input(self):
        """Signal for very large input (scale invariance)"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x_small = torch.randn(10)
        x_large = x_small * 1e10
        
        s_small = core.signal(x_small)
        s_large = core.signal(x_large)
        
        assert math.isclose(s_small, s_large, rel_tol=1e-5), \
            f"Scale invariance failed: {s_small} != {s_large}"
    
    def test_signal_mixed_sign_input(self):
        """Signal for input with mixed signs"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.randn(10)
        x[:5] = -x[:5]  # Make half negative
        
        s = core.signal(x)
        assert math.isfinite(s), f"Signal {s} should be finite for mixed signs"
        assert not math.isnan(s), "Signal should not be NaN"
    
    def test_signal_zero_vector(self):
        """Signal for zero vector should be handled"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.zeros(10)
        s = core.signal(x)
        # Should be finite (implementation dependent)
        assert math.isfinite(s), f"Signal {s} should be finite for zero input"
    
    def test_signal_with_different_targets(self):
        """Signal after training towards different targets"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.randn(10)
        
        # Train towards different targets (using train_batch for epochs)
        for target in [0.2, 0.5, 0.8]:
            X = x.unsqueeze(0).repeat(10, 1)
            y = torch.full((10,), target)
            core.train_batch(X, y, lr=0.01, max_epochs=5)
            s = core.signal(x)
            assert math.isfinite(s), f"Signal {s} should be finite for target {target}"
    
    def test_signal_batch_preserves_shape(self):
        """Batch signal should preserve batch dimension"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x_batch = torch.randn(32, 10)
        
        s_batch = core.forward(x_batch)
        assert s_batch.shape == (32,), f"Expected shape (32,), got {s_batch.shape}"
    
    def test_signal_3d_batch(self):
        """3D batch signal"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x_batch = torch.randn(4, 8, 10)
        
        s_batch = core.forward(x_batch)
        assert s_batch.shape == (4, 8), f"Expected shape (4, 8), got {s_batch.shape}"
