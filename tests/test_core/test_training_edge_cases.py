"""
Tests for training edge cases in KokaoCore.
Level 1 - Basic Unit Tests
"""
import torch
import pytest
import math
from kokao import KokaoCore, CoreConfig


class TestTrainingEdgeCases:
    """Tests for training edge cases"""
    
    def test_train_with_zero_target(self):
        """Training with target=0"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.randn(10)
        loss = core.train(x, target=0.0, lr=0.01)
        assert math.isfinite(loss), f"Loss {loss} should be finite"
    
    def test_train_with_negative_target(self):
        """Training with negative target"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.randn(10)
        loss = core.train(x, target=-0.5, lr=0.01)
        assert math.isfinite(loss), f"Loss {loss} should be finite"
    
    def test_train_with_very_large_target(self):
        """Training with very large target"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.randn(10)
        loss = core.train(x, target=100.0, lr=0.01)
        assert math.isfinite(loss), f"Loss {loss} should be finite"
    
    def test_train_zero_learning_rate(self):
        """Training with lr=0 should not change weights"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.randn(10)
        w_plus_before = core.w_plus.clone()
        w_minus_before = core.w_minus.clone()
        
        core.train(x, target=0.8, lr=0.0)
        
        assert torch.allclose(core.w_plus, w_plus_before), \
            "w_plus should not change with lr=0"
        assert torch.allclose(core.w_minus, w_minus_before), \
            "w_minus should not change with lr=0"
    
    def test_train_very_large_learning_rate(self):
        """Training with very large lr (gradient clipping)"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.randn(10)
        loss = core.train(x, target=0.8, lr=10.0)
        
        assert math.isfinite(loss), f"Loss {loss} should be finite"
        assert not torch.isnan(core.w_plus).any(), "w_plus should not have NaN"
        assert not torch.isnan(core.w_minus).any(), "w_minus should not have NaN"
    
    def test_train_same_sample_multiple_times(self):
        """Training multiple times on same sample"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.randn(10)
        losses = []
        
        for _ in range(10):
            loss = core.train(x, target=0.8, lr=0.01)
            losses.append(loss)
        
        # Loss should generally decrease (with some variance)
        assert losses[-1] < losses[0], \
            f"Final loss {losses[-1]} should be < initial {losses[0]}"
    
    def test_train_alternating_targets(self):
        """Training with alternating targets"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.randn(10)
        
        for target in [0.2, 0.8, 0.2, 0.8, 0.2]:
            loss = core.train(x, target=target, lr=0.01)
            assert math.isfinite(loss), f"Loss {loss} should be finite for target {target}"
    
    def test_train_with_zero_input(self):
        """Training with zero input"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.zeros(10)
        loss = core.train(x, target=0.8, lr=0.01)
        assert math.isfinite(loss), f"Loss {loss} should be finite"
    
    def test_train_kosyakov_mode_zero_input(self):
        """Kosyakov mode with zero input should handle gracefully"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.zeros(10)
        loss = core.train(x, target=0.8, lr=0.01, mode="kosyakov")
        # Should not crash, loss should be finite
        assert math.isfinite(loss), f"Loss {loss} should be finite"
    
    def test_train_with_extreme_target(self):
        """Training with extreme target values"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.randn(10)
        
        for target in [-100, -10, 0, 10, 100]:
            core_copy = KokaoCore(CoreConfig(input_dim=10, seed=42))
            loss = core_copy.train(x, target=target, lr=0.01)
            assert math.isfinite(loss), f"Loss should be finite for target {target}"
    
    def test_train_batch_with_single_sample(self):
        """Batch training with batch_size=1"""
        core = KokaoCore(CoreConfig(input_dim=10))
        X = torch.randn(1, 10)
        y = torch.full((1,), 0.8)
        
        loss = core.train_batch(X, y, lr=0.01)
        assert math.isfinite(loss), f"Loss {loss} should be finite"
    
    def test_train_batch_with_large_batch(self):
        """Batch training with large batch"""
        core = KokaoCore(CoreConfig(input_dim=10))
        X = torch.randn(512, 10)
        y = torch.full((512,), 0.8)
        
        loss = core.train_batch(X, y, lr=0.01)
        assert math.isfinite(loss), f"Loss {loss} should be finite"
    
    def test_train_preserves_weight_positivity(self):
        """Training should preserve weight positivity"""
        core = KokaoCore(CoreConfig(input_dim=10))
        x = torch.randn(10)
        
        for _ in range(100):
            core.train(x, target=0.8, lr=0.01)
        
        w_plus, w_minus = core._get_effective_weights()
        assert (w_plus > 0).all(), "w_plus should remain positive"
        assert (w_minus > 0).all(), "w_minus should remain positive"
    
    def test_train_version_increments_each_step(self):
        """Version should increment on each training step"""
        core = KokaoCore(CoreConfig(input_dim=10))
        initial_version = core.version
        
        for i in range(5):
            core.train(torch.randn(10), target=0.8, lr=0.01)
            assert core.version == initial_version + i + 1, \
                f"Version should be {initial_version + i + 1}"
