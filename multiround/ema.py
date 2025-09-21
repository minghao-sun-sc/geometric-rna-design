# multiround/ema.py
from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, Iterable, Optional
import torch
import torch.nn as nn

def _unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m

@dataclass
class EMAConfig:
    decay: float = 0.999
    warmup_updates: int = 0          # optional warmup for decay scheduling
    device: Optional[str] = None     # None = same device as model; "cpu" to offload
    include_buffers: bool = False    # rarely needed; usually False
    skip_keys: Iterable[str] = ()    # substrings to skip (e.g., ["bias", "bn"])

class ModelEMA:
    """
    Exponential Moving Average of model parameters (float32 shadows).
    Call `update(model)` AFTER each optimizer step.
    """
    def __init__(self, model: nn.Module, cfg: EMAConfig):
        self.cfg = cfg
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}  # for temporary swap

        model = _unwrap_model(model)
        device = cfg.device
        for n, p in model.named_parameters():
            if (not p.requires_grad) or (not p.dtype.is_floating_point):
                continue
            if any(s in n for s in cfg.skip_keys):
                continue
            t = p.detach().float().clone()
            if device is not None:
                t = t.to(device)
            self.shadow[n] = t

        if cfg.include_buffers:
            for n, b in model.named_buffers():
                if b.dtype.is_floating_point:
                    t = b.detach().float().clone()
                    if device is not None:
                        t = t.to(device)
                    self.shadow[n] = t

    def _decay(self) -> float:
        # Optional warmup: ramps the *effective* decay up with updates
        if self.cfg.warmup_updates and self.num_updates < self.cfg.warmup_updates:
            # keep EMA closer to current weights during early steps
            warm = float(self.num_updates) / float(self.cfg.warmup_updates)
            return 1.0 - (1.0 - self.cfg.decay) * warm
        return self.cfg.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Call after each optimizer.step()."""
        self.num_updates += 1
        d = self._decay()
        model = _unwrap_model(model)
        for n, p in model.named_parameters():
            if n not in self.shadow:
                continue
            if (not p.requires_grad) or (not p.dtype.is_floating_point):
                continue
            s = self.shadow[n]
            # cast to float32 for numerically stable EMA
            s.mul_(d).add_(p.detach().float().to(s.device), alpha=(1.0 - d))

        if self.cfg.include_buffers:
            for n, b in model.named_buffers():
                if n in self.shadow and b.dtype.is_floating_point:
                    s = self.shadow[n]
                    s.mul_(d).add_(b.detach().float().to(s.device), alpha=(1.0 - d))

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite `model` params with EMA weights (in-place)."""
        model = _unwrap_model(model)
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.device, dtype=p.dtype))
        if self.cfg.include_buffers:
            for n, b in model.named_buffers():
                if n in self.shadow:
                    b.data.copy_(self.shadow[n].to(b.device, dtype=b.dtype))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Save current model params so we can restore after temporary EMA swap."""
        self.backup.clear()
        model = _unwrap_model(model)
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.detach().clone()
        if self.cfg.include_buffers:
            for n, b in model.named_buffers():
                if n in self.shadow:
                    self.backup[n] = b.data.detach().clone()

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore params saved by `store()`."""
        model = _unwrap_model(model)
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        if self.cfg.include_buffers:
            for n, b in model.named_buffers():
                if n in self.backup:
                    b.data.copy_(self.backup[n])
        self.backup.clear()

    @contextmanager
    def apply_to(self, model: nn.Module):
        """
        Context manager for "evaluate with EMA":
            with ema.apply_to(model):
                evaluate(...)
        """
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> Dict:
        return {
            "decay": self.cfg.decay,
            "warmup_updates": self.cfg.warmup_updates,
            "num_updates": self.num_updates,
            "device": self.cfg.device,
            "include_buffers": self.cfg.include_buffers,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict) -> None:
        self.num_updates = int(state.get("num_updates", 0))
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}


# Enhanced EMA integration for multi-round training
class MultiRoundEMA:
    """
    Enhanced EMA wrapper for multi-round training with advanced features:
    - Integration with training loops
    - Checkpoint management
    - Reference model updates using EMA
    - Comprehensive logging
    """
    
    def __init__(self, model: nn.Module, cfg):
        """
        Initialize MultiRoundEMA with configuration.
        
        Args:
            model: The model to apply EMA to
            cfg: Configuration object with EMA settings
        """
        self.model = model
        self.cfg = cfg
        self.enabled = self._check_ema_enabled()
        
        if self.enabled:
            # Create EMA configuration
            ema_cfg = EMAConfig(
                decay=getattr(cfg.ema, 'decay', 0.999),
                warmup_updates=getattr(cfg.ema, 'update_after_step', 0),
                device=getattr(cfg.ema, 'device', None),
                include_buffers=getattr(cfg.ema, 'include_buffers', False),
                skip_keys=getattr(cfg.ema, 'skip_keys', [])
            )
            
            # Initialize EMA
            self.ema = ModelEMA(model, ema_cfg)
            self.update_every = getattr(cfg.ema, 'update_every', 1)
            self.step_count = 0
            
            print(f"✅ Multi-round EMA initialized:")
            print(f"   Decay: {ema_cfg.decay}")
            print(f"   Warmup updates: {ema_cfg.warmup_updates}")
            print(f"   Update every: {self.update_every} steps")
            print(f"   Device: {ema_cfg.device or 'same as model'}")
        else:
            self.ema = None
            print("ℹ️ EMA disabled in configuration")
    
    def _check_ema_enabled(self) -> bool:
        """Check if EMA is enabled in configuration."""
        return (hasattr(self.cfg, 'ema') and 
                getattr(self.cfg.ema, 'enable', False))
    
    def update(self) -> bool:
        """
        Update EMA parameters. Call this after each optimizer step.
        
        Returns:
            bool: True if EMA was updated, False otherwise
        """
        if not self.enabled:
            return False
        
        self.step_count += 1
        
        # Check if we should update this step
        if self.step_count % self.update_every != 0:
            return False
        
        # Update EMA
        self.ema.update(self.model)
        return True
    
    def apply_ema_to_model(self):
        """Apply EMA parameters to the model in-place."""
        if self.enabled:
            self.ema.copy_to(self.model)
    
    def get_ema_model_copy(self) -> Optional[nn.Module]:
        """
        Get a copy of the model with EMA parameters applied.
        This creates a new model instance.
        
        Returns:
            Model with EMA parameters, or None if EMA is disabled
        """
        if not self.enabled:
            return None
        
        # Create a deep copy
        import copy
        ema_model = copy.deepcopy(self.model)
        
        # Apply EMA parameters to the copy
        self.ema.copy_to(ema_model)
        return ema_model
    
    def create_ema_reference_model(self) -> Optional[nn.Module]:
        """
        Create a reference model using EMA parameters.
        This is useful for the reference model update strategy.
        
        Returns:
            Reference model with EMA parameters, or None if disabled
        """
        if not self.enabled:
            return None
        
        print("🔄 Creating EMA-based reference model...")
        ema_reference = self.get_ema_model_copy()
        
        if ema_reference is not None:
            print(f"✅ EMA reference model created with {self.ema.num_updates} updates")
            return ema_reference
        
        return None
    
    @contextmanager
    def ema_eval_context(self):
        """
        Context manager for evaluation with EMA parameters.
        
        Usage:
            with ema_manager.ema_eval_context():
                results = evaluate_model(model)
        """
        if self.enabled:
            with self.ema.apply_to(self.model):
                yield
        else:
            yield
    
    def save_ema_checkpoint(self, checkpoint_path: str, additional_data: Optional[Dict] = None):
        """
        Save checkpoint with EMA state.
        
        Args:
            checkpoint_path: Path to save checkpoint
            additional_data: Additional data to include in checkpoint
        """
        if not self.enabled:
            # Save regular checkpoint without EMA
            checkpoint_data = {
                'model_state_dict': self.model.state_dict(),
                'ema_enabled': False
            }
        else:
            # Save checkpoint with EMA state
            checkpoint_data = {
                'model_state_dict': self.model.state_dict(),
                'ema_state_dict': self.ema.state_dict(),
                'ema_enabled': True,
                'ema_num_updates': self.ema.num_updates
            }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        torch.save(checkpoint_data, checkpoint_path)
        
        if self.enabled:
            print(f"💾 EMA checkpoint saved: {checkpoint_path} (updates: {self.ema.num_updates})")
        else:
            print(f"💾 Regular checkpoint saved: {checkpoint_path}")
    
    def load_ema_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint with potential EMA state.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load EMA state if available and enabled
        if self.enabled and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            print(f"✅ EMA state loaded from checkpoint (updates: {self.ema.num_updates})")
        elif self.enabled:
            print("⚠️ EMA enabled but no EMA state in checkpoint, reinitializing EMA")
            # Reinitialize EMA with current model parameters
            self.ema = ModelEMA(self.model, self.ema.cfg)
        
        return checkpoint
    
    def get_ema_metrics_for_logging(self) -> Dict[str, float]:
        """
        Get EMA-related metrics for logging.
        
        Returns:
            Dictionary of EMA metrics
        """
        if not self.enabled:
            return {}
        
        return {
            'ema_num_updates': float(self.ema.num_updates),
            'ema_decay': self.ema.cfg.decay,
            'ema_step_count': float(self.step_count),
            'ema_effective_decay': self.ema._decay()
        }
    
    def reset_ema(self):
        """Reset EMA state and reinitialize with current model parameters."""
        if self.enabled:
            print("🔄 Resetting EMA state...")
            self.ema = ModelEMA(self.model, self.ema.cfg)
            self.step_count = 0
            print("✅ EMA reset completed")


def create_multi_round_ema(model: nn.Module, cfg) -> MultiRoundEMA:
    """
    Factory function to create MultiRoundEMA instance.
    
    Args:
        model: Model to apply EMA to
        cfg: Configuration object
        
    Returns:
        MultiRoundEMA instance
    """
    return MultiRoundEMA(model, cfg)


def integrate_ema_with_trainer(trainer, cfg) -> Optional[MultiRoundEMA]:
    """
    Integrate EMA with an existing trainer.
    
    Args:
        trainer: DPO trainer instance
        cfg: Configuration object
        
    Returns:
        MultiRoundEMA instance if enabled, None otherwise
    """
    if not hasattr(cfg, 'ema') or not getattr(cfg.ema, 'enable', False):
        return None
    
    # Create EMA for the policy model
    ema_manager = MultiRoundEMA(trainer.policy, cfg)
    
    # Optionally create EMA for reference model if specified
    use_ema_for_reference = getattr(cfg.ema, 'use_ema_for_reference', False)
    if use_ema_for_reference:
        print("🔄 Configuring EMA for reference model updates...")
        # This will be used in the reference model update strategy
    
    return ema_manager


# Utility functions for EMA-based reference model updates
def update_reference_with_ema(trainer, ema_manager: MultiRoundEMA):
    """
    Update the reference model using EMA parameters.
    
    Args:
        trainer: DPO trainer instance
        ema_manager: MultiRoundEMA manager
    """
    if not ema_manager.enabled:
        print("⚠️ EMA not enabled, cannot update reference with EMA")
        return False
    
    print("🎯 Updating reference model with EMA parameters...")
    
    try:
        # Apply EMA parameters to reference model
        ema_manager.ema.copy_to(trainer.reference)
        
        print(f"✅ Reference model updated with EMA (updates: {ema_manager.ema.num_updates})")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update reference with EMA: {e}")
        return False


# Example integration with training loop
class EMATrainingCallback:
    """
    Callback for integrating EMA updates into training loops.
    """
    
    def __init__(self, ema_manager: MultiRoundEMA):
        self.ema_manager = ema_manager
    
    def on_optimizer_step(self, step: int) -> Dict[str, float]:
        """Call after each optimizer step."""
        metrics = {}
        
        if self.ema_manager.enabled:
            updated = self.ema_manager.update()
            metrics.update(self.ema_manager.get_ema_metrics_for_logging())
            
            if updated:
                metrics['ema_updated'] = 1.0
            else:
                metrics['ema_updated'] = 0.0
        
        return metrics
    
    def on_evaluation(self, model, eval_fn, *args, **kwargs):
        """Evaluate with EMA parameters if enabled."""
        if self.ema_manager.enabled:
            print("📊 Evaluating with EMA parameters...")
            with self.ema_manager.ema_eval_context():
                return eval_fn(model, *args, **kwargs)
        else:
            return eval_fn(model, *args, **kwargs)
    
    def on_checkpoint_save(self, checkpoint_path: str, additional_data: Optional[Dict] = None):
        """Save checkpoint with EMA state."""
        self.ema_manager.save_ema_checkpoint(checkpoint_path, additional_data)
    
    def on_checkpoint_load(self, checkpoint_path: str):
        """Load checkpoint with EMA state."""
        return self.ema_manager.load_ema_checkpoint(checkpoint_path)
