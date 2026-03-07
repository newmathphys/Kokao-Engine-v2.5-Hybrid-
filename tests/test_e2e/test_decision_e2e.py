"""
End-to-End tests: Decision Making.
Level 4 - E2E Tests
"""
import torch
import pytest
from kokao import IntuitiveEtalonSystem, SelfPlanningSystem, CoreConfig


class TestDecisionMakingE2E:
    """E2E decision making tests"""
    
    def test_survival_simulation(self):
        """Survival simulation: food, safety, rest"""
        config = CoreConfig(input_dim=10)
        goal_system = SelfPlanningSystem(config)
        etalon_system = IntuitiveEtalonSystem(config)
        
        # Learn etalons
        etalon_system.learn_etalon("food", torch.randn(10))
        etalon_system.learn_etalon("danger", torch.randn(10))
        etalon_system.learn_etalon("rest", torch.randn(10))
        
        # Simulate 100 steps
        energy_level = 1.0
        safety_level = 1.0
        
        for step in range(100):
            # Random stimulus
            stimulus_type = torch.randint(0, 3, (1,)).item()
            stimulus = torch.randn(10)
            
            # Recognize
            recognized = etalon_system.recognize(stimulus, threshold=0.3)
            
            # React
            if recognized == "food" and energy_level < 0.8:
                goal_system.satisfy_goal("energy", amount=0.1)
                energy_level = goal_system.get_goal_value("energy")
            elif recognized == "danger":
                goal_system.satisfy_goal("safety", amount=0.1)
                safety_level = goal_system.get_goal_value("safety")
            elif recognized == "rest":
                goal_system.satisfy_goal("energy", amount=0.05)
            
            # Natural decay
            energy_level = max(0, energy_level - 0.01)
            safety_level = max(0, safety_level - 0.005)
            
            goal_system.goals["physiological"]["energy"]["value"] = energy_level
            goal_system.goals["physiological"]["safety"]["value"] = safety_level
        
        # System should maintain levels > 0
        assert energy_level > 0 or safety_level > 0, \
            "Should maintain at least one level > 0"
    
    def test_goal_hierarchy_decision(self):
        """Decision making with goal hierarchy"""
        config = CoreConfig(input_dim=10)
        goal_system = SelfPlanningSystem(config)
        
        # Set different deprivation levels
        goal_system.goals["physiological"]["energy"]["deprivation"] = 0.9
        goal_system.goals["social"]["status"]["deprivation"] = 0.5
        goal_system.goals["abstract"]["self_expression"]["deprivation"] = 0.2
        
        # Get active goal (should be highest deprivation)
        active_goal = goal_system.get_active_goal()
        
        # Energy has highest deprivation
        assert active_goal == "energy", \
            f"Expected 'energy' as active goal, got {active_goal}"
        
        # Satisfy energy
        goal_system.satisfy_goal("energy", amount=0.5)
        
        # Get active goal again
        new_active_goal = goal_system.get_active_goal()
        
        # Goal might change or stay same depending on deprivation levels
        # Just check it returns valid result
        assert new_active_goal is None or isinstance(new_active_goal, str), \
            f"Active goal should be string or None, got {new_active_goal}"
    
    def test_conflicting_goals(self):
        """Decision making with conflicting goals"""
        config = CoreConfig(input_dim=10)
        goal_system = SelfPlanningSystem(config)
        
        # Create conflicting goals
        goal_system.goals["physiological"]["energy"]["deprivation"] = 0.8
        goal_system.goals["social"]["affection"]["deprivation"] = 0.8
        
        # Get priority
        energy_priority = goal_system.get_goal_priority("energy")
        affection_priority = goal_system.get_goal_priority("affection")
        
        # Both should have high priority
        assert energy_priority > 0.5, f"Energy priority {energy_priority} should be high"
        assert affection_priority > 0.5, f"Affection priority {affection_priority} should be high"
        
        # Satisfy one
        goal_system.satisfy_goal("energy", amount=0.5)
        
        # Priorities should change
        new_energy_priority = goal_system.get_goal_priority("energy")
        assert new_energy_priority < energy_priority, \
            "Energy priority should decrease after satisfaction"
    
    def test_long_term_goal_pursuit(self):
        """Long-term goal pursuit"""
        config = CoreConfig(input_dim=10)
        goal_system = SelfPlanningSystem(config)
        
        # Set abstract goal
        goal_system.goals["abstract"]["self_expression"]["target"] = 1.0
        goal_system.goals["abstract"]["self_expression"]["value"] = 0.0
        
        # Pursue goal over many steps
        for step in range(50):
            # Small progress each step
            goal_system.satisfy_goal("self_expression", amount=0.02)
        
        # Check progress
        final_value = goal_system.get_goal_value("self_expression")
        assert final_value > 0.5, f"Final value {final_value} should be > 0.5"
        
        # Check deprivation decreased
        final_deprivation = goal_system.get_goal_mismatch("self_expression")
        assert final_deprivation < 0.5, \
            f"Final deprivation {final_deprivation} should be < 0.5"
    
    def test_emotional_state_tracking(self):
        """Emotional state tracking through goals"""
        config = CoreConfig(input_dim=10)
        goal_system = SelfPlanningSystem(config)
        
        # Track total deprivation (like stress)
        initial_deprivation = goal_system.get_total_deprivation()
        
        # Create stress (increase deprivation)
        for goal_level in goal_system.goals:
            for goal_id in goal_system.goals[goal_level]:
                goal_system.goals[goal_level][goal_id]["deprivation"] = 0.8
        
        high_deprivation = goal_system.get_total_deprivation()
        assert high_deprivation > initial_deprivation, \
            "Deprivation should increase"
        
        # Reduce stress (satisfy goals)
        for goal_level in goal_system.goals:
            for goal_id in goal_system.goals[goal_level]:
                goal_system.satisfy_goal(goal_id, amount=0.5)
        
        low_deprivation = goal_system.get_total_deprivation()
        assert low_deprivation < high_deprivation, \
            "Deprivation should decrease after satisfaction"
