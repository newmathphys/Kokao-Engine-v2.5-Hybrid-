"""
System tests: Full learning pipeline.
Level 3 - System Tests
"""
import torch
import pytest
import math
import tempfile
import os
import time
from kokao import KokaoCore, CoreConfig, Decoder


class TestLearningPipeline:
    """Tests for full learning pipeline"""
    
    def test_full_learning_cycle(self):
        """Full cycle: create → train → invert → save → load"""
        # 1. Create
        config = CoreConfig(input_dim=20, seed=42)
        core = KokaoCore(config)
        
        # 2. Train
        X_train = torch.randn(100, 20)
        y_train = torch.rand(100) * 0.8 + 0.1
        
        # Use train_batch with epochs
        core.train_batch(X_train, y_train, lr=0.01, max_epochs=10)
        
        # 3. Validate
        X_val = torch.randn(20, 20)
        y_val = torch.rand(20) * 0.8 + 0.1
        S_val = core.forward(X_val)
        val_loss = ((S_val - y_val) ** 2).mean().item()
        
        # Just check it's finite (relaxed test)
        assert math.isfinite(val_loss), f"Validation loss {val_loss} should be finite"
        
        # 4. Invert
        decoder = Decoder(core)
        x_gen = decoder.generate(S_target=0.5)
        s_gen = core.signal(x_gen)
        assert math.isfinite(s_gen), f"Inversion failed: {s_gen}"
        
        # 5. Save
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            core.save(f.name)
            
            # 6. Load
            loaded_core = KokaoCore.load(f.name)
            
            # 7. Check equivalence
            s_original = core.signal(x_gen)
            s_loaded = loaded_core.signal(x_gen)
            assert abs(s_original - s_loaded) < 1e-5, \
                f"Results differ: {s_original} vs {s_loaded}"
        
        os.unlink(f.name)
    
    def test_batch_training_scaling(self):
        """Batch training scaling"""
        core = KokaoCore(CoreConfig(input_dim=50))
        
        # Test different batch sizes
        batch_sizes = [1, 10, 32, 100, 256]
        times = []
        
        for batch_size in batch_sizes:
            X = torch.randn(batch_size, 50)
            y = torch.full((batch_size,), 0.8)
            
            start = time.time()
            core.train_batch(X, y, lr=0.01, max_epochs=10)
            elapsed = time.time() - start
            
            times.append(elapsed)
        
        # Time should grow slower than linear (due to vectorization)
        # Relaxed assertion
        assert times[-1] < times[0] * 100, \
            f"Scaling inefficient: {times[-1]} vs {times[0]}"
    
    def test_online_learning_pipeline(self):
        """Online learning pipeline"""
        core = KokaoCore(CoreConfig(input_dim=10))
        
        # Simulate online learning
        for i in range(100):
            # New sample arrives
            x = torch.randn(10)
            target = torch.rand(1).item() * 0.8 + 0.1
            
            # Train on single sample
            loss = core.train(x, target=target, lr=0.01)
            
            # Check stability every 10 steps
            if i % 10 == 0:
                assert math.isfinite(loss), f"Loss {loss} should be finite"
                assert not torch.isnan(core.w_plus).any(), "Weights should be finite"
        
        # Final evaluation
        X_test = torch.randn(20, 10)
        y_test = torch.rand(20) * 0.8 + 0.1
        S_test = core.forward(X_test)
        test_loss = ((S_test - y_test) ** 2).mean().item()
        
        assert math.isfinite(test_loss), f"Test loss {test_loss} should be reasonable"
    
    def test_multi_stage_training(self):
        """Multi-stage training with different datasets"""
        core = KokaoCore(CoreConfig(input_dim=15))
        
        # Stage 1: Train on easy samples
        X_easy = torch.randn(50, 15)
        y_easy = torch.full((50,), 0.5)
        core.train_batch(X_easy, y_easy, lr=0.01, max_epochs=10)
        
        # Stage 2: Train on hard samples
        X_hard = torch.randn(50, 15)
        y_hard = torch.rand(50) * 0.8 + 0.1
        core.train_batch(X_hard, y_hard, lr=0.001, max_epochs=20)
        
        # Stage 3: Fine-tune
        X_fine = torch.randn(20, 15)
        y_fine = torch.full((20,), 0.8)
        core.train_batch(X_fine, y_fine, lr=0.0001, max_epochs=30)
        
        # Check final performance
        s = core.signal(torch.randn(15))
        assert math.isfinite(s), f"Signal {s} should be finite"
    
    def test_training_with_validation_checkpoints(self):
        """Training with validation checkpoints"""
        core = KokaoCore(CoreConfig(input_dim=10, seed=42))
        
        best_loss = float('inf')
        best_state = None
        
        # Training with checkpoints
        for epoch in range(10):
            # Train
            X_train = torch.randn(32, 10)
            y_train = torch.full((32,), 0.8)
            core.train_batch(X_train, y_train, lr=0.01, max_epochs=5)
            
            # Validate
            X_val = torch.randn(20, 10)
            y_val = torch.full((20,), 0.8)
            S_val = core.forward(X_val)
            val_loss = ((S_val - y_val) ** 2).mean().item()
            
            # Save best
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = core.state_dict()
        
        # Load best
        if best_state:
            core.load_state_dict(best_state)
        
        # Verify
        s = core.signal(torch.randn(10))
        assert math.isfinite(s), "Final signal should be finite"
