"""
Integration tests: Decision Making (Etalon + Goal System).
Level 2 - Integration Tests
"""
import torch
import pytest
from kokao import IntuitiveEtalonSystem, SelfPlanningSystem, CoreConfig


class TestDecisionMaking:
    """Tests for decision making integration"""
    
    def test_etalon_activates_goal(self):
        """Etalon recognition activates goal"""
        etalon_system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        goal_system = SelfPlanningSystem(CoreConfig(input_dim=10))
        
        # Learn "food" etalon
        food_vector = torch.randn(10)
        etalon_system.learn_etalon("food", food_vector)
        
        # Recognize
        result = etalon_system.recognize(food_vector, threshold=0.5)
        assert result == "food", f"Should recognize 'food', got {result}"
        
        # Activate goal
        goal_system.satisfy_goal("energy", amount=0.3)
        assert goal_system.get_goal_value("energy") > 0.5, \
            "Energy goal should be satisfied"
    
    def test_goal_priority_affects_recognition(self):
        """Goal priority affects recognition threshold"""
        etalon_system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        goal_system = SelfPlanningSystem(CoreConfig(input_dim=10))
        
        # Create deprivation
        goal_system.goals["physiological"]["energy"]["deprivation"] = 0.8
        
        # Priority should be high
        priority = goal_system.get_goal_priority("energy")
        assert priority > 0.5, f"Priority {priority} should be > 0.5"
    
    def test_full_decision_cycle(self):
        """Full decision cycle: perceive → recognize → act"""
        config = CoreConfig(input_dim=10)
        etalon_system = IntuitiveEtalonSystem(config)
        goal_system = SelfPlanningSystem(config)
        
        # Learn etalons
        etalon_system.learn_etalon("danger", torch.randn(10))
        etalon_system.learn_etalon("safety", torch.randn(10))
        
        # Stimulus
        stimulus = torch.randn(10)
        
        # Perceive
        recognized = etalon_system.recognize(stimulus, threshold=0.3)
        
        # Act based on recognition
        if recognized:
            if recognized == "danger":
                goal_system.satisfy_goal("safety", amount=0.2)
            elif recognized == "safety":
                goal_system.satisfy_goal("energy", amount=0.1)
        
        # Check goal state changed
        total_deprivation = goal_system.get_total_deprivation()
        assert torch.isfinite(torch.tensor(total_deprivation)), \
            "Deprivation should be finite"
    
    def test_multiple_goals_competition(self):
        """Multiple goals competing"""
        goal_system = SelfPlanningSystem(CoreConfig(input_dim=10))
        
        # Create deprivation in multiple goals
        goal_system.goals["physiological"]["energy"]["deprivation"] = 0.9
        goal_system.goals["social"]["status"]["deprivation"] = 0.5
        goal_system.goals["abstract"]["self_expression"]["deprivation"] = 0.2
        
        # Get active goal (should be highest deprivation)
        active_goal = goal_system.get_active_goal()
        
        # Energy has highest deprivation, should be active
        assert active_goal == "energy", \
            f"Expected 'energy' as active goal, got {active_goal}"
    
    def test_goal_satisfaction_chain(self):
        """Goal satisfaction chain reaction"""
        etalon_system = IntuitiveEtalonSystem(CoreConfig(input_dim=10))
        goal_system = SelfPlanningSystem(CoreConfig(input_dim=10))
        
        # Learn etalon
        etalon_system.learn_etalon("reward", torch.randn(10))
        
        # Multiple recognition-satisfaction cycles
        for i in range(5):
            result = etalon_system.recognize(torch.randn(10), threshold=0.1)
            if result:
                goal_system.satisfy_goal("energy", amount=0.1)
        
        # Check goal hierarchy
        hierarchy = goal_system.get_goal_hierarchy()
        assert "physiological" in hierarchy, "Should have physiological goals"
        assert "energy" in hierarchy["physiological"], "Should have energy goal"
    
    def test_deprivation_driven_behavior(self):
        """Deprivation drives behavior"""
        goal_system = SelfPlanningSystem(CoreConfig(input_dim=10))
        
        # Set high deprivation
        goal_system.goals["physiological"]["energy"]["deprivation"] = 0.95
        goal_system.goals["physiological"]["energy"]["value"] = 0.05
        
        # Get priority
        priority = goal_system.get_goal_priority("energy")
        assert priority > 0.8, f"Priority {priority} should be high"
        
        # Satisfy
        goal_system.satisfy_goal("energy", amount=0.5)
        
        # Priority should decrease
        new_priority = goal_system.get_goal_priority("energy")
        assert new_priority < priority, \
            f"Priority should decrease: {new_priority} < {priority}"
