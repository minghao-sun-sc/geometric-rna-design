"""
Improved Reference Manager with Robust CUDA Error Handling
Fixes CUDA assertion errors in model loading and GPU transfer operations.
"""

import os, copy, torch
import logging
import traceback
from src.models import AutoregressiveMultiGNNv1

logger = logging.getLogger(__name__)

def validate_model_parameters(model, name="model"):
    """
    Validate model parameters for common issues that cause CUDA errors.
    
    Args:
        model: PyTorch model to validate
        name: Name for logging purposes
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        logger.info(f"Validating {name} parameters...")
        
        invalid_params = []
        
        for param_name, param in model.named_parameters():
            # Check for NaN values
            if torch.isnan(param).any():
                invalid_params.append(f"{param_name}: contains NaN")
                continue
                
            # Check for Inf values  
            if torch.isinf(param).any():
                invalid_params.append(f"{param_name}: contains Inf")
                continue
                
            # Check for empty tensors (allow dummy_param which are legitimate in GVP architecture)
            if param.numel() == 0:
                if "dummy_param" not in param_name.lower():
                    invalid_params.append(f"{param_name}: empty tensor")
                    continue
                else:
                    # dummy_param tensors are expected to be empty in GVP architecture
                    logger.debug(f"Allowing empty dummy_param tensor: {param_name}")
                    continue
                
            # Check for unreasonable shapes
            if any(dim <= 0 for dim in param.shape):
                invalid_params.append(f"{param_name}: invalid shape {param.shape}")
                continue
                
            # Check for extreme values that might cause overflow
            if param.abs().max() > 1e6:
                logger.warning(f"{param_name}: has very large values (max: {param.abs().max():.2e})")
                
        if invalid_params:
            logger.error(f"Model validation failed for {name}:")
            for issue in invalid_params[:10]:  # Show first 10 issues
                logger.error(f"  - {issue}")
            if len(invalid_params) > 10:
                logger.error(f"  ... and {len(invalid_params) - 10} more issues")
            return False
            
        logger.info(f"✓ {name} parameters validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Parameter validation failed for {name}: {e}")
        return False

def safe_gpu_transfer(model, device, name="model"):
    """
    Safely transfer model to GPU with validation and error handling.
    
    Args:
        model: PyTorch model to transfer
        device: Target device (usually CUDA)
        name: Name for logging purposes
        
    Returns:
        model: Model on target device, or None if transfer failed
    """
    try:
        logger.info(f"Transferring {name} to {device}...")
        
        # Validate parameters before transfer
        if not validate_model_parameters(model, name):
            logger.error(f"Parameter validation failed for {name}, aborting GPU transfer")
            return None
            
        # Clear CUDA cache before transfer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
            # Check available memory
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_available = memory_total - memory_allocated
            
            logger.info(f"GPU memory: {memory_allocated:.1f}GB used, {memory_available:.1f}GB available")
            
            if memory_available < 1.0:  # Less than 1GB available
                logger.warning("Low GPU memory available, clearing cache...")
                torch.cuda.empty_cache()
        
        # Transfer model with error handling
        try:
            model_gpu = model.to(device)
            logger.info(f"✓ {name} successfully transferred to {device}")
            return model_gpu
            
        except RuntimeError as e:
            if "CUDA error" in str(e):
                logger.error(f"CUDA error during {name} transfer: {e}")
                logger.info("Attempting recovery...")
                
                # Try to recover
                torch.cuda.empty_cache()
                
                # Try again with explicit error checking
                try:
                    model_gpu = model.to(device)
                    logger.info(f"✓ {name} transfer succeeded after recovery")
                    return model_gpu
                except Exception as e2:
                    logger.error(f"Recovery failed for {name}: {e2}")
                    return None
            else:
                logger.error(f"Non-CUDA runtime error during {name} transfer: {e}")
                return None
                
    except Exception as e:
        logger.error(f"Unexpected error during {name} GPU transfer: {e}")
        traceback.print_exc()
        return None

def build_model_from_cfg(model_cfg):
    """Build model from configuration with validation."""
    try:
        logger.info("Building model from configuration...")
        
        model = AutoregressiveMultiGNNv1(
            node_in_dim=tuple(model_cfg.node_in_dim),
            node_h_dim=tuple(model_cfg.node_h_dim),
            edge_in_dim=tuple(model_cfg.edge_in_dim),
            edge_h_dim=tuple(model_cfg.edge_h_dim),
            num_layers=model_cfg.num_layers,
            drop_rate=model_cfg.drop_rate,
            out_dim=model_cfg.out_dim,
        )
        
        # Validate the built model
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model built successfully with {param_count:,} parameters")
        
        return model
        
    except Exception as e:
        logger.error(f"Model building failed: {e}")
        raise

def smart_load(model, ckpt_path, map_location="cpu"):
    """Load checkpoint with validation and error handling."""
    try:
        logger.info(f"Loading checkpoint: {ckpt_path}")
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        # Load checkpoint
        sd = torch.load(ckpt_path, map_location=map_location)
        
        # Handle different checkpoint formats
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
            
        # Validate checkpoint
        if not isinstance(sd, dict):
            raise ValueError(f"Invalid checkpoint format: expected dict, got {type(sd)}")
            
        # Load state dict with error handling
        missing, unexpected = model.load_state_dict(sd, strict=True)
        
        # Log any issues
        if missing:
            logger.warning(f"Missing keys in checkpoint: {len(missing)}")
            for key in missing[:5]:  # Show first 5
                logger.warning(f"  Missing: {key}")
                
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected)}")
            for key in unexpected[:5]:  # Show first 5
                logger.warning(f"  Unexpected: {key}")
                
        logger.info("✓ Checkpoint loaded successfully")
        return missing, unexpected
        
    except Exception as e:
        logger.error(f"Checkpoint loading failed: {e}")
        raise

def build_policy_and_reference(cfg, device):
    """
    Build policy and reference models with robust error handling.
    
    This is the main function that was causing CUDA assertion errors.
    """
    try:
        logger.info("Building policy and reference models...")
        
        # Build models on CPU first
        logger.info("Building policy model...")
        policy = build_model_from_cfg(cfg.model)
        
        logger.info("Building reference model...")
        reference = build_model_from_cfg(cfg.model)
        
        # Load checkpoint on CPU
        logger.info("Loading base checkpoint...")
        ckpt = cfg.paths.base_checkpoint
        
        missing1, unexpected1 = smart_load(policy, ckpt, map_location="cpu")
        missing2, unexpected2 = smart_load(reference, ckpt, map_location="cpu")
        
        # Validate models before GPU transfer
        if not validate_model_parameters(policy, "policy"):
            raise RuntimeError("Policy model validation failed")
            
        if not validate_model_parameters(reference, "reference"):
            raise RuntimeError("Reference model validation failed")
        
        # Transfer to GPU safely
        logger.info("Transferring models to GPU...")
        
        policy_gpu = safe_gpu_transfer(policy, device, "policy")
        if policy_gpu is None:
            raise RuntimeError("Failed to transfer policy model to GPU")
            
        reference_gpu = safe_gpu_transfer(reference, device, "reference")
        if reference_gpu is None:
            raise RuntimeError("Failed to transfer reference model to GPU")
        
        # Configure reference model
        reference_gpu.eval()
        for p in reference_gpu.parameters():
            p.requires_grad_(False)
            
        logger.info("✓ Policy and reference models built successfully")
        return policy_gpu, reference_gpu
        
    except Exception as e:
        logger.error(f"Failed to build policy and reference models: {e}")
        
        # Clean up on failure
        torch.cuda.empty_cache()
        
        # Re-raise with more context
        raise RuntimeError(f"Model building failed: {e}") from e

def build_policy_only(cfg, device):
    """Build only policy model (for SimPO training) with robust error handling."""
    try:
        logger.info("Building policy model...")
        
        # Build on CPU first
        policy = build_model_from_cfg(cfg.model)
        
        # Load checkpoint
        ckpt = cfg.paths.base_checkpoint
        smart_load(policy, ckpt, map_location="cpu")
        
        # Validate and transfer to GPU
        if not validate_model_parameters(policy, "policy"):
            raise RuntimeError("Policy model validation failed")
            
        policy_gpu = safe_gpu_transfer(policy, device, "policy")
        if policy_gpu is None:
            raise RuntimeError("Failed to transfer policy model to GPU")
            
        logger.info("✓ Policy model built successfully")
        return policy_gpu
        
    except Exception as e:
        logger.error(f"Failed to build policy model: {e}")
        torch.cuda.empty_cache()
        raise RuntimeError(f"Policy model building failed: {e}") from e

def save_checkpoint(root, name, model, optimizer, scheduler, step, best_metric, cfg):
    """Save checkpoint with error handling."""
    try:
        path = os.path.join(root, f"{name}.pt")
        
        # Handle custom schedulers that don't have state_dict
        scheduler_state = None
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            scheduler_state = scheduler.state_dict()
        
        obj = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler_state,
            "step": step,
            "best_metric": best_metric,
            "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else None,
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(obj, path)
        logger.info(f"✅ Saved checkpoint: {path}")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise

def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    """Load checkpoint with error handling."""
    try:
        logger.info(f"Loading checkpoint: {path}")
        
        obj = torch.load(path, map_location=device)
        
        # Load model state
        missing, unexpected = model.load_state_dict(obj["model"], strict=True)
        
        # Load optimizer state
        if optimizer is not None and obj.get("optimizer") is not None:
            optimizer.load_state_dict(obj["optimizer"])
            
        # Load scheduler state
        if scheduler is not None and obj.get("scheduler") is not None:
            scheduler.load_state_dict(obj["scheduler"])
            
        step = obj.get("step", 0)
        best_metric = obj.get("best_metric", None)
        
        logger.info(f"✓ Checkpoint loaded successfully (step: {step})")
        return step, best_metric
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise