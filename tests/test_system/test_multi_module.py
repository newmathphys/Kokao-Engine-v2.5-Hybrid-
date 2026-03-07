"""
System tests: Multi-module integration.
Level 3 - System Tests
"""
import torch
import pytest
import math
from kokao import (
    KokaoCore, CoreConfig, Decoder,
    IntuitiveEtalonSystem, NormalIntuitiveEtalonSystem,
    SelfPlanningSystem
)


class TestMultiModuleSystem:
    """Tests for multi-module integration"""
    
    def test_cognitive_pipeline(self):
        """Cognitive pipeline: perception → recognition → decision → action"""
        config = CoreConfig(input_dim=20)
        
        # Modules
        etalon = IntuitiveEtalonSystem(config)
        goal_system = SelfPlanningSystem(config)
        core = KokaoCore(config)
        
        # 1. Learn etalons
        etalon.learn_etalon("danger", torch.randn(20))
        etalon.learn_etalon("safety", torch.randn(20))
        etalon.learn_etalon("food", torch.randn(20))
        
        # 2. Stimulus
        stimulus = torch.randn(20)
        
        # 3. Recognize
        recognized = etalon.recognize(stimulus, threshold=0.3)
        
        # 4. Goal reaction
        if recognized:
            goal_system.satisfy_goal(recognized, amount=0.1)
        
        # 5. Compute signal
        signal = core.signal(stimulus)
        
        # Checks
        assert recognized is None or recognized in ["danger", "safety", "food"]
        assert math.isfinite(signal)
    
    def test_normal_etalon_associative_learning(self):
        """Associative learning in normal etalon system"""
        config = CoreConfig(input_dim=10)
        system = NormalIntuitiveEtalonSystem(config)
        
        # Learn pairs
        for i in range(10):
            image = torch.randn(10)
            action = torch.randn(10)
            system.learn_image_action_pair(
                image, action,
                target_image=0.8,
                target_action=0.8,
                reward=1.0
            )
        
        # Predict
        test_image = torch.randn(10)
        predicted_action = system.predict_action(test_image)
        
        assert predicted_action.shape == (10,), \
            f"Expected shape (10,), got {predicted_action.shape}"
        assert torch.isfinite(predicted_action).all(), \
            "Action should be finite"
        
        # Fantasy
        refined = system.imagine_and_refine(predicted_action, iterations=5)
        assert refined.shape == (10,), "Refined action should have same shape"
    
    def test_full_cognitive_architecture(self):
        """Full cognitive architecture test"""
        config = CoreConfig(input_dim=15)
        
        # All modules
        core = KokaoCore(config)
        etalon = IntuitiveEtalonSystem(config)
        normal_etalon = NormalIntuitiveEtalonSystem(config)
        goal_system = SelfPlanningSystem(config)
        decoder = Decoder(core)
        
        # 1. Training phase
        for _ in range(50):
            x = torch.randn(15)
            target = torch.rand(1).item() * 0.8 + 0.1
            core.train(x, target=target, lr=0.01)
        
        # 2. Etalon learning
        for i in range(5):
            etalon.learn_etalon(f"class_{i}", torch.randn(15) + i)
        
        # 3. Associative learning
        for _ in range(10):
            image = torch.randn(15)
            action = torch.randn(15)
            normal_etalon.learn_image_action_pair(image, action, reward=0.5)
        
        # 4. Recognition
        stimulus = torch.randn(15)
        recognized = etalon.recognize(stimulus, threshold=0.2)
        
        # 5. Action prediction
        predicted_action = normal_etalon.predict_action(stimulus)
        
        # 6. Signal computation
        signal = core.signal(stimulus)
        
        # 7. Inversion
        x_generated = decoder.generate(S_target=0.5)
        
        # Checks
        assert math.isfinite(signal), f"Signal {signal} should be finite"
        assert predicted_action.shape == (15,), "Action should have correct shape"
        assert x_generated.shape == (15,), "Generated vector should have correct shape"
    
    def test_module_isolation(self):
        """Test that modules don't interfere with each other"""
        config = CoreConfig(input_dim=10)
        
        # Create independent modules
        core1 = KokaoCore(config)
        core2 = KokaoCore(config)
        etalon1 = IntuitiveEtalonSystem(config)
        etalon2 = IntuitiveEtalonSystem(config)
        
        # Train core1
        for _ in range(50):
            core1.train(torch.randn(10), target=0.8, lr=0.01)
        
        # Train core2 differently
        for _ in range(50):
            core2.train(torch.randn(10), target=0.2, lr=0.01)
        
        # Signals should be different
        x_test = torch.randn(10)
        s1 = core1.signal(x_test)
        s2 = core2.signal(x_test)
        
        assert abs(s1 - s2) > 0.1, \
            f"Independent cores should have different signals: {s1} vs {s2}"
    
    def test_resource_sharing(self):
        """Test resource sharing between modules"""
        config = CoreConfig(input_dim=20, target_sum=100.0)
        
        # Multiple modules sharing config
        core1 = KokaoCore(config)
        core2 = KokaoCore(config)
        etalon = IntuitiveEtalonSystem(config)
        
        # All should have same target_sum
        w1_plus, _ = core1._get_effective_weights()
        w2_plus, _ = core2._get_effective_weights()
        
        assert torch.isclose(w1_plus.sum(), torch.tensor(100.0), atol=1e-5)
        assert torch.isclose(w2_plus.sum(), torch.tensor(100.0), atol=1e-5)
        
        # Train both
        core1.train(torch.randn(20), target=0.5, lr=0.01)
        core2.train(torch.randn(20), target=0.9, lr=0.01)
        
        # Both should maintain target_sum
        w1_plus, _ = core1._get_effective_weights()
        w2_plus, _ = core2._get_effective_weights()
        
        assert torch.isclose(w1_plus.sum(), torch.tensor(100.0), atol=1e-5)
        assert torch.isclose(w2_plus.sum(), torch.tensor(100.0), atol=1e-5)
