# multiround/wandb_manager.py
"""
Enhanced WandB organization and management for multiround training.
Provides auto-generated run names, hierarchical tagging, and better workspace organization.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib

class MultiRoundWandBManager:
    """
    Enhanced WandB management for multiround experiments.
    
    Features:
    - Auto-generated unique run names with meaningful information
    - Hierarchical tagging (experiment_type, strategy, margin_info, etc.)
    - Run grouping and summary dashboards
    - Config fingerprinting for reproducibility
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        # Handle missing experiment config gracefully
        experiment_config = getattr(cfg, 'experiment', None)
        if experiment_config:
            self.experiment_name = getattr(experiment_config, 'name', 'unknown_experiment')
            self.experiment_strategy = getattr(experiment_config, 'strategy', 'plan_a')
        else:
            # Extract from wandb run_name if no experiment config
            run_name = getattr(cfg.wandb, 'run_name', 'unknown_experiment')
            self.experiment_name = run_name
            self.experiment_strategy = 'plan_a'
        
        self.dynamic_pairs = getattr(cfg.multiround, 'dynamic_pairs', False)
        
        # Generate enhanced run configuration
        self.run_config = self._generate_run_config()
        
    def _generate_run_config(self) -> Dict[str, Any]:
        """Generate enhanced run configuration with auto-naming and tagging."""
        
        # Generate unique, informative run name
        run_name = self._generate_run_name()
        
        # Generate hierarchical tags
        tags = self._generate_tags()
        
        # Generate run group for related experiments
        group = self._generate_group()
        
        # Generate run notes
        notes = self._generate_notes()
        
        # Create config fingerprint for reproducibility
        config_fingerprint = self._generate_config_fingerprint()
        
        return {
            'run_name': run_name,
            'tags': tags,
            'group': group,
            'notes': notes,
            'config_fingerprint': config_fingerprint,
            'enhanced_metadata': self._generate_enhanced_metadata()
        }
    
    def _generate_run_name(self) -> str:
        """Generate informative, unique run name."""
        timestamp = datetime.now().strftime("%m%d_%H%M")
        
        # Base components
        components = [
            self.experiment_strategy,  # plan_a, plan_b, etc.
            self.experiment_name       # dynamic_margins, etc.
        ]
        
        # Add margin information
        if self.dynamic_pairs:
            components.append("dynamic")
        else:
            margin = getattr(self.cfg.paths, 'pair_margin', 'unknown')
            components.append(f"m{margin}")
        
        # Add key hyperparameters
        beta = getattr(self.cfg.dpo, 'beta', 0.12)
        sft_lambda = getattr(self.cfg.dpo, 'sft_lambda', 0.1)
        components.extend([f"b{beta}", f"l{sft_lambda}"])
        
        # Add batch info
        batch_size = getattr(self.cfg.training, 'batch_size', 16)
        grad_accum = getattr(self.cfg.training, 'grad_accum_steps', 4)
        effective_batch = batch_size * grad_accum
        components.append(f"bs{effective_batch}")
        
        # Combine with timestamp
        run_name = "_".join(components) + f"_{timestamp}"
        
        # Ensure run name is not too long (wandb has limits)
        if len(run_name) > 100:
            # Use shorter version with hash
            short_name = f"{self.experiment_strategy}_{self.experiment_name}_{timestamp}"
            config_hash = self._generate_config_fingerprint()[:8]
            run_name = f"{short_name}_{config_hash}"
        
        return run_name
    
    def _generate_tags(self) -> List[str]:
        """Generate hierarchical tags for better organization."""
        tags = []
        
        # Experiment hierarchy
        tags.extend([
            "multiround",                    # Top level: training type
            self.experiment_strategy,        # Strategy level: plan_a, plan_b, etc.
            self.experiment_name            # Experiment level: dynamic_margins, etc.
        ])
        
        # Training configuration tags
        loss_type = getattr(self.cfg, 'loss_type', 'dpo')
        tags.append(loss_type)
        
        # Pair configuration tags
        if self.dynamic_pairs:
            tags.extend(["dynamic_pairs", "margin_switching"])
        else:
            margin = getattr(self.cfg.paths, 'pair_margin', 'unknown')
            tags.append(f"margin_{margin}")
        
        # Hyperparameter ranges (for easy filtering)
        beta = getattr(self.cfg.dpo, 'beta', 0.12)
        if beta <= 0.1:
            tags.append("low_beta")
        elif beta >= 0.2:
            tags.append("high_beta")
        else:
            tags.append("med_beta")
        
        # Training scale tags
        num_rounds = getattr(self.cfg.multiround, 'num_rounds', 5)
        epochs_per_round = getattr(self.cfg.multiround, 'epochs_per_round', 20)
        total_epochs = num_rounds * epochs_per_round
        
        if total_epochs <= 50:
            tags.append("short_training")
        elif total_epochs >= 150:
            tags.append("long_training")
        else:
            tags.append("standard_training")
        
        # Add environment tags
        tags.append("ribopo")
        
        # Add date tag for temporal organization
        date_tag = datetime.now().strftime("date_%Y%m")
        tags.append(date_tag)
        
        return tags
    
    def _generate_group(self) -> str:
        """Generate group name for related experiments."""
        # Group by experiment type and key parameters
        group_components = [
            self.experiment_strategy,
            self.experiment_name,
            f"r{getattr(self.cfg.multiround, 'num_rounds', 5)}"
        ]
        
        # Add date for temporal grouping (weekly groups)
        date_group = datetime.now().strftime("%Y_w%U")  # Year_weekNN
        group_components.append(date_group)
        
        return "_".join(group_components)
    
    def _generate_notes(self) -> str:
        """Generate informative notes for the run."""
        notes_parts = []
        
        # Add experiment description
        experiment_config = getattr(self.cfg, 'experiment', None)
        description = getattr(experiment_config, 'description', '') if experiment_config else ''
        if description:
            notes_parts.append(description)
        
        # Add key configuration details
        config_details = []
        
        if self.dynamic_pairs:
            config_details.append("Dynamic preference pairs")
            if hasattr(self.cfg.multiround, 'pair_configs'):
                config_details.append("0.25*std→0.125*std margin strategy")
        
        num_rounds = getattr(self.cfg.multiround, 'num_rounds', 5)
        epochs_per_round = getattr(self.cfg.multiround, 'epochs_per_round', 20)
        config_details.append(f"{num_rounds} rounds × {epochs_per_round} epochs")
        
        beta = getattr(self.cfg.dpo, 'beta', 0.12)
        sft_lambda = getattr(self.cfg.dpo, 'sft_lambda', 0.1)
        config_details.append(f"DPO β={beta}, SFT λ={sft_lambda}")
        
        if config_details:
            notes_parts.append(" | ".join(config_details))
        
        # Add generation timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        notes_parts.append(f"Generated: {timestamp}")
        
        return " • ".join(notes_parts)
    
    def _generate_config_fingerprint(self) -> str:
        """Generate fingerprint of key configuration for reproducibility."""
        # Select key config values that affect training
        key_config = {
            'experiment_name': self.experiment_name,
            'experiment_strategy': self.experiment_strategy,
            'dynamic_pairs': self.dynamic_pairs,
            'num_rounds': getattr(self.cfg.multiround, 'num_rounds', 5),
            'epochs_per_round': getattr(self.cfg.multiround, 'epochs_per_round', 20),
            'dpo_beta': getattr(self.cfg.dpo, 'beta', 0.12),
            'sft_lambda': getattr(self.cfg.dpo, 'sft_lambda', 0.1),
            'batch_size': getattr(self.cfg.training, 'batch_size', 16),
            'grad_accum_steps': getattr(self.cfg.training, 'grad_accum_steps', 4),
            'learning_rate': getattr(self.cfg.optimizer, 'lr', 1e-4),
            'loss_type': getattr(self.cfg, 'loss_type', 'dpo')
        }
        
        # Create hash of key config
        config_str = json.dumps(key_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _generate_enhanced_metadata(self) -> Dict[str, Any]:
        """Generate enhanced metadata for the run."""
        metadata = {
            'generation_time': datetime.now().isoformat(),
            'config_source': 'multiround_dynamic_config',
            'ribopo_version': '1.0',  # Could be read from version file
            'experiment_family': f"{self.experiment_strategy}_{self.experiment_name}",
            'training_characteristics': {
                'total_epochs': getattr(self.cfg.multiround, 'num_rounds', 5) * getattr(self.cfg.multiround, 'epochs_per_round', 20),
                'effective_batch_size': getattr(self.cfg.training, 'batch_size', 16) * getattr(self.cfg.training, 'grad_accum_steps', 4),
                'dynamic_pairs': self.dynamic_pairs,
                'reference_updates': getattr(self.cfg.multiround, 'update_reference', False)
            }
        }
        
        return metadata
    
    def get_wandb_config(self) -> Dict[str, Any]:
        """Get complete wandb configuration."""
        base_config = {
            'project': getattr(self.cfg.wandb, 'project', 'RiboPO-Multiround'),
            'entity': getattr(self.cfg.wandb, 'entity', None),
            'name': self.run_config['run_name'],
            'tags': self.run_config['tags'],
            'group': self.run_config['group'],
            'notes': self.run_config['notes'],
            'mode': getattr(self.cfg.wandb, 'mode', 'online')
        }
        
        # Add config directory if specified
        wandb_dir = getattr(self.cfg.wandb, 'dir', None)
        if wandb_dir:
            base_config['dir'] = wandb_dir
        
        return base_config
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters for logging."""
        hyperparams = {
            # Core training parameters
            'total_rounds': getattr(self.cfg.multiround, 'num_rounds', 5),
            'epochs_per_round': getattr(self.cfg.multiround, 'epochs_per_round', 20),
            'dpo_beta': getattr(self.cfg.dpo, 'beta', 0.12),
            'sft_lambda': getattr(self.cfg.dpo, 'sft_lambda', 0.1),
            'learning_rate': getattr(self.cfg.optimizer, 'lr', 1e-4),
            'batch_size': getattr(self.cfg.training, 'batch_size', 16),
            'grad_accum_steps': getattr(self.cfg.training, 'grad_accum_steps', 4),
            'loss_type': getattr(self.cfg, 'loss_type', 'dpo'),
            
            # Multiround-specific parameters
            'dynamic_pairs': self.dynamic_pairs,
            'update_reference': getattr(self.cfg.multiround, 'update_reference', False),
            # Handle both naming schemes for eval samples
            'eval_samples': (getattr(self.cfg.multiround, 'eval_samples', None) or 
                           getattr(self.cfg.multiround, 'n_samples_eval', 8)),
            'final_eval_samples': (getattr(self.cfg.multiround, 'final_eval_samples', None) or 
                                 getattr(self.cfg.multiround, 'n_samples_final_eval', 64)),
            
            # Experiment metadata
            'experiment_name': self.experiment_name,
            'experiment_strategy': self.experiment_strategy,
            'config_fingerprint': self.run_config['config_fingerprint'],
            
            # Enhanced metadata
            **self.run_config['enhanced_metadata']
        }
        
        return hyperparams
    
    def log_experiment_start(self, wandb_instance):
        """Log experiment start with enhanced metadata."""
        if wandb_instance:
            wandb_instance.log({
                "experiment/config_fingerprint": self.run_config['config_fingerprint'],
                "experiment/start_time": datetime.now().isoformat(),
                "experiment/enhanced_metadata": self.run_config['enhanced_metadata']
            })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of WandB manager configuration."""
        return {
            'run_name': self.run_config['run_name'],
            'tags_count': len(self.run_config['tags']),
            'group': self.run_config['group'],
            'config_fingerprint': self.run_config['config_fingerprint'],
            'dynamic_pairs': self.dynamic_pairs,
            'experiment_family': f"{self.experiment_strategy}_{self.experiment_name}"
        }