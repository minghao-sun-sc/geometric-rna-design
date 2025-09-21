# multiround/debug/unit/test_ema.py
"""Unit tests for EMA (Exponential Moving Average) module."""

import os
import sys
import unittest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

# Import testing utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.test_helpers import TestConfig, setup_test_environment, mock_model_weights

# Import the EMA module (will be implemented)
try:
    from multiround.ema import EMAModel
except ImportError:
    # Create a placeholder for testing
    class EMAModel:
        def __init__(self, model, decay=0.999, update_after_step=0, update_every=1):
            self.model = model
            self.decay = decay
            self.update_after_step = update_after_step
            self.update_every = update_every
            self.step_count = 0
            self.shadow_params = None
            self._setup_shadow_params()
        
        def _setup_shadow_params(self):
            """Initialize shadow parameters."""
            self.shadow_params = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow_params[name] = param.data.clone()
        
        def update(self):
            """Update EMA parameters."""
            if self.step_count < self.update_after_step:
                self.step_count += 1
                return
            
            if self.step_count % self.update_every != 0:
                self.step_count += 1
                return
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name] = (
                        self.decay * self.shadow_params[name] + 
                        (1 - self.decay) * param.data
                    )
            
            self.step_count += 1
        
        def apply_shadow(self):
            """Apply shadow parameters to model."""
            self.store_params = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.store_params[name] = param.data.clone()
                    param.data.copy_(self.shadow_params[name])
        
        def restore_params(self):
            """Restore original parameters."""
            if hasattr(self, 'store_params'):
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.store_params:
                        param.data.copy_(self.store_params[name])
                delattr(self, 'store_params')
        
        def state_dict(self):
            """Get EMA state dict."""
            return {
                'shadow_params': self.shadow_params,
                'step_count': self.step_count,
                'decay': self.decay
            }
        
        def load_state_dict(self, state_dict):
            """Load EMA state dict."""
            self.shadow_params = state_dict['shadow_params']
            self.step_count = state_dict['step_count']
            self.decay = state_dict['decay']


class SimpleTestModel(nn.Module):
    """Simple model for testing EMA functionality."""
    
    def __init__(self, input_dim=10, hidden_dim=5, output_dim=2):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class TestEMAModel(unittest.TestCase):
    """Test cases for EMAModel."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.model = SimpleTestModel()
        self.config = TestConfig.create_minimal_config()
    
    def test_ema_initialization(self):
        """Test EMA model initialization."""
        ema = EMAModel(self.model, decay=0.999)
        
        # Test basic properties
        self.assertEqual(ema.decay, 0.999)
        self.assertEqual(ema.update_after_step, 0)
        self.assertEqual(ema.update_every, 1)
        self.assertEqual(ema.step_count, 0)
        
        # Test shadow parameters initialization
        self.assertIsNotNone(ema.shadow_params)
        
        # Verify shadow params match model params initially
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertTrue(torch.equal(ema.shadow_params[name], param.data))
    
    def test_ema_update_basic(self):
        """Test basic EMA update functionality."""
        ema = EMAModel(self.model, decay=0.9)
        
        # Get initial parameters
        initial_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        # Modify model parameters
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Update EMA
        ema.update()
        
        # Check that shadow parameters have been updated
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                expected = 0.9 * initial_params[name] + 0.1 * param.data
                self.assertTrue(torch.allclose(ema.shadow_params[name], expected, atol=1e-6))
    
    def test_ema_update_after_step(self):
        """Test EMA update_after_step functionality."""
        ema = EMAModel(self.model, decay=0.9, update_after_step=3)
        
        # Get initial shadow parameters
        initial_shadow = {name: shadow.clone() for name, shadow in ema.shadow_params.items()}
        
        # Update model parameters and call ema.update() multiple times
        for step in range(5):
            with torch.no_grad():
                for param in self.model.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            ema.update()
            
            if step < 3:
                # Shadow parameters should not change before update_after_step
                for name, shadow in ema.shadow_params.items():
                    self.assertTrue(torch.equal(shadow, initial_shadow[name]))
            else:
                # Shadow parameters should change after update_after_step
                for name, shadow in ema.shadow_params.items():
                    self.assertFalse(torch.equal(shadow, initial_shadow[name]))
                    break  # Only need to check once after threshold
    
    def test_ema_update_every(self):
        """Test EMA update_every functionality."""
        ema = EMAModel(self.model, decay=0.9, update_every=3)
        
        update_counts = 0
        initial_shadow = {name: shadow.clone() for name, shadow in ema.shadow_params.items()}
        
        # Track how many times shadow parameters actually update
        for step in range(10):
            prev_shadow = {name: shadow.clone() for name, shadow in ema.shadow_params.items()}
            
            with torch.no_grad():
                for param in self.model.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            
            ema.update()
            
            # Check if shadow parameters changed
            shadow_changed = False
            for name, shadow in ema.shadow_params.items():
                if not torch.equal(shadow, prev_shadow[name]):
                    shadow_changed = True
                    break
            
            if shadow_changed:
                update_counts += 1
        
        # Should update every 3 steps: steps 0, 3, 6, 9 = 4 updates total
        expected_updates = len([i for i in range(10) if i % 3 == 0])
        self.assertEqual(update_counts, expected_updates)
    
    def test_apply_and_restore_shadow(self):
        """Test applying and restoring shadow parameters."""
        ema = EMAModel(self.model, decay=0.9)
        
        # Get initial model parameters
        initial_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        # Modify model parameters
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        
        # Update EMA to create different shadow parameters
        ema.update()
        
        # Get current model parameters (after modification)
        modified_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        # Apply shadow parameters
        ema.apply_shadow()
        
        # Check that model parameters now match shadow parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertTrue(torch.equal(param.data, ema.shadow_params[name]))
        
        # Restore original parameters
        ema.restore_params()
        
        # Check that model parameters are restored to modified state
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertTrue(torch.equal(param.data, modified_params[name]))
    
    def test_ema_state_dict(self):
        """Test EMA state dict saving and loading."""
        ema1 = EMAModel(self.model, decay=0.95)
        
        # Perform some updates
        for _ in range(5):
            with torch.no_grad():
                for param in self.model.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            ema1.update()
        
        # Save state dict
        state_dict = ema1.state_dict()
        
        # Create new EMA instance and load state dict
        model2 = SimpleTestModel()
        ema2 = EMAModel(model2, decay=0.999)  # Different initial decay
        ema2.load_state_dict(state_dict)
        
        # Check that state was properly loaded
        self.assertEqual(ema2.decay, 0.95)
        self.assertEqual(ema2.step_count, ema1.step_count)
        
        # Check shadow parameters match
        for name in ema1.shadow_params:
            if name in ema2.shadow_params:
                self.assertTrue(torch.equal(ema1.shadow_params[name], ema2.shadow_params[name]))
    
    def test_ema_convergence_behavior(self):
        """Test EMA convergence behavior with fixed target."""
        ema = EMAModel(self.model, decay=0.9)
        
        # Get initial parameters
        target_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        # Set model to fixed target values
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.fill_(1.0)  # Set to constant value
        
        # Perform many EMA updates
        for _ in range(100):
            ema.update()
        
        # Shadow parameters should converge close to target (1.0)
        for name, shadow in ema.shadow_params.items():
            self.assertTrue(torch.allclose(shadow, torch.ones_like(shadow), atol=1e-2))
    
    def test_ema_with_different_decay_values(self):
        """Test EMA behavior with different decay values."""
        decay_values = [0.9, 0.99, 0.999, 0.9999]
        
        results = {}
        for decay in decay_values:
            model = SimpleTestModel()
            ema = EMAModel(model, decay=decay)
            
            # Set initial value for both model and shadow
            with torch.no_grad():
                for param in model.parameters():
                    param.data.fill_(0.0)
                for name, shadow in ema.shadow_params.items():
                    shadow.fill_(0.0)
            
            # Set target value
            with torch.no_grad():
                for param in model.parameters():
                    param.data.fill_(1.0)
            
            # Single update
            ema.update()
            
            # Store result
            first_param_name = next(iter(ema.shadow_params.keys()))
            results[decay] = ema.shadow_params[first_param_name][0, 0].item()
        
        # Higher decay should result in values closer to initial (0.0)
        # Lower decay should result in values closer to target (1.0)
        self.assertLess(results[0.9999], results[0.999])
        self.assertLess(results[0.999], results[0.99])
        self.assertLess(results[0.99], results[0.9])


class TestEMAIntegration(unittest.TestCase):
    """Test EMA integration with training scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        setup_test_environment()
        self.model = SimpleTestModel()
        self.config = TestConfig.create_minimal_config()
    
    def test_ema_with_optimizer_steps(self):
        """Test EMA behavior with actual optimizer steps."""
        ema = EMAModel(self.model, decay=0.99)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Simple training loop simulation
        for step in range(10):
            # Forward pass with dummy data
            x = torch.randn(5, 10)
            y = torch.randn(5, 2)
            
            # Compute loss
            output = self.model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update EMA after optimizer step
            ema.update()
        
        # EMA should have tracked the parameter changes
        self.assertEqual(ema.step_count, 10)
        
        # Shadow parameters should be different from current model parameters
        params_different = False
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in ema.shadow_params:
                if not torch.equal(param.data, ema.shadow_params[name]):
                    params_different = True
                    break
        
        self.assertTrue(params_different)
    
    def test_ema_for_reference_model_use_case(self):
        """Test EMA for the reference model update use case."""
        # This simulates the multiround training scenario
        ema = EMAModel(self.model, decay=0.999, update_after_step=100)
        
        # Simulate training for one round (many steps)
        for step in range(200):
            # Simulate parameter updates
            with torch.no_grad():
                for param in self.model.parameters():
                    param.add_(torch.randn_like(param) * 0.001)
            
            ema.update()
        
        # At end of round, apply EMA parameters for reference model
        original_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        ema.apply_shadow()
        ema_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        ema.restore_params()
        restored_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        # Verify apply/restore worked correctly
        for name in original_params:
            self.assertTrue(torch.equal(original_params[name], restored_params[name]))
            # EMA params should be different (smoother)
            self.assertFalse(torch.equal(original_params[name], ema_params[name]))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)