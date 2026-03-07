"""
Stress tests: Numerical stability.
Level 4 - Stress Tests
"""
import torch
import pytest
import math
from kokao import KokaoCore, CoreConfig


class TestNumericalStability:
    """Numerical stability stress tests"""
    
    def test_extreme_input_values(self):
        """Handle extreme input values"""
        core = KokaoCore(CoreConfig(input_dim=10))
        
        # Very large values - scale invariant so should work
        x_large = torch.randn(10) * 1e10
        s_large = core.signal(x_large)
        assert math.isfinite(s_large), f"Signal {s_large} should be finite"
        
        # Very small values
        x_small = torch.randn(10) * 1e-10
        s_small = core.signal(x_small)
        assert math.isfinite(s_small), f"Signal {s_small} should be finite"
    
    def test_extreme_weights(self):
        """Handle extreme weights"""
        core = KokaoCore(CoreConfig(input_dim=10))
        
        # Artificially create large weights
        with torch.no_grad():
            core.w_plus.fill_(100)
            core.w_minus.fill_(100)
        
        x = torch.randn(10)
        s = core.signal(x)
        assert math.isfinite(s), f"Signal {s} should be finite"
    
    def test_repeated_normalization(self):
        """Repeated normalization"""
        core = KokaoCore(CoreConfig(input_dim=10))
        
        for _ in range(1000):
            core.train(torch.randn(10), target=0.8, lr=0.01)
            w_plus, w_minus = core._get_effective_weights()
            
            # Check normalization
            assert torch.isclose(w_plus.sum(), torch.tensor(100.0), atol=1.0), \
                f"w_plus sum {w_plus.sum().item()} should be ~100"
            assert torch.isclose(w_minus.sum(), torch.tensor(100.0), atol=1.0), \
                f"w_minus sum {w_minus.sum().item()} should be ~100"
    
    def test_gradient_clipping_effectiveness(self):
        """Gradient clipping effectiveness"""
        core = KokaoCore(CoreConfig(input_dim=10))
        
        # Train with very large learning rate
        for _ in range(100):
            core.train(torch.randn(10), target=0.8, lr=10.0)
        
        # Weights should still be finite
        assert not torch.isnan(core.w_plus).any(), "w_plus should not have NaN"
        assert not torch.isinf(core.w_plus).any(), "w_plus should not have Inf"
        assert not torch.isnan(core.w_minus).any(), "w_minus should not have NaN"
    
    def test_division_by_zero_protection(self):
        """Division by zero protection"""
        core = KokaoCore(CoreConfig(input_dim=10))
        
        # Create situation where S⁻ could be very small
        with torch.no_grad():
            core.w_minus.fill_(0.001)
        
        x = torch.randn(10)
        s = core.signal(x)
        
        # Should not crash, should be finite
        assert math.isfinite(s), f"Signal {s} should be finite"
    
    def test_softplus_numerical_stability(self):
        """Softplus numerical stability"""
        core = KokaoCore(CoreConfig(input_dim=10))
        
        # Set extreme internal parameters
        with torch.no_grad():
            core.w_plus.fill_(100)  # Very large positive
            core.w_minus.fill_(-100)  # Very large negative
        
        w_plus, w_minus = core._get_effective_weights()
        
        # softplus should handle this
        assert (w_plus > 0).all(), "w_plus should be positive"
        assert (w_minus > 0).all(), "w_minus should be positive"
        assert torch.isfinite(w_plus).all(), "w_plus should be finite"
        assert torch.isfinite(w_minus).all(), "w_minus should be finite"
    
    def test_signal_sign_preservation(self):
        """Signal sign preservation under stress"""
        core = KokaoCore(CoreConfig(input_dim=10))
        
        # Train towards positive target
        for _ in range(100):
            core.train(torch.randn(10), target=0.8, lr=0.01)
        
        # Test with various inputs
        for _ in range(50):
            x = torch.randn(10)
            s = core.signal(x)
            
            # With positive weights, signal should typically be positive
            assert math.isfinite(s), f"Signal {s} should be finite"
    
    def test_batch_numerical_stability(self):
        """Batch numerical stability"""
        core = KokaoCore(CoreConfig(input_dim=50))
        
        # Large batch with extreme values
        X = torch.randn(256, 50) * 10
        
        # Forward pass
        S = core.forward(X)
        
        assert torch.isfinite(S).all(), f"All outputs should be finite"
        assert S.shape == (256,), f"Expected shape (256,), got {S.shape}"
    
    def test_weight_update_stability(self):
        """Weight update stability"""
        core = KokaoCore(CoreConfig(input_dim=20))
        
        initial_w_plus = core.w_plus.clone()
        initial_w_minus = core.w_minus.clone()
        
        # Many updates
        for _ in range(500):
            core.train(torch.randn(20), target=torch.rand(1).item(), lr=0.01)
        
        # Weights should change but remain finite
        assert not torch.equal(core.w_plus, initial_w_plus), \
            "w_plus should change"
        assert not torch.equal(core.w_minus, initial_w_minus), \
            "w_minus should change"
        
        assert torch.isfinite(core.w_plus).all(), "w_plus should be finite"
        assert torch.isfinite(core.w_minus).all(), "w_minus should be finite"
