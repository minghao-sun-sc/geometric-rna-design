# multiround/pair_provider.py
"""
Dynamic preference pair provider for multiround training.
Supports switching between different preference pair datasets based on training rounds.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import json

# Import data validator (conditional to avoid circular imports)
try:
    from multiround.data_validator import MultiRoundDataValidator
    VALIDATOR_AVAILABLE = True
except ImportError:
    VALIDATOR_AVAILABLE = False

@dataclass
class PairConfig:
    """Configuration for a set of preference pairs."""
    margin_type: str
    description: str
    train_path: str
    val_path: str
    test_path: str
    
    def validate_paths(self) -> bool:
        """Check if all pair files exist."""
        paths = [self.train_path, self.val_path, self.test_path]
        return all(os.path.exists(path) for path in paths)
    
    def get_pair_counts(self) -> Dict[str, int]:
        """Count pairs in each split."""
        counts = {}
        for split, path in [("train", self.train_path), ("val", self.val_path), ("test", self.test_path)]:
            try:
                with open(path, 'r') as f:
                    counts[split] = sum(1 for _ in f)
            except Exception as e:
                print(f"Warning: Could not count pairs in {path}: {e}")
                counts[split] = 0
        return counts


class MultiRoundPairProvider:
    """
    Manages preference pairs for multiround training with support for:
    - Static pairs (same throughout training)
    - Dynamic pairs (different pairs per round or round group)
    - Validation and fallback mechanisms
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.dynamic_mode = getattr(cfg.multiround, 'dynamic_pairs', False)
        self.current_round = 1
        self.current_config = None
        
        # Initialize validation settings first (needed by setup methods)
        validation_cfg = getattr(cfg, 'validation', None)
        self.check_files = getattr(validation_cfg, 'check_pair_files', True) if validation_cfg else True
        self.fallback_enabled = getattr(validation_cfg, 'fallback_to_static', True) if validation_cfg else True
        self.fallback_margin = getattr(validation_cfg, 'static_fallback_margin', '125') if validation_cfg else '125'
        self.enable_data_validation = getattr(validation_cfg, 'enable_data_validation', True) if validation_cfg else True
        
        # Setup configurations after validation settings are ready
        if self.dynamic_mode:
            self._setup_dynamic_configs()
        else:
            self._setup_static_config()
        
        # Initialize data validator if available and enabled
        self.data_validator = None
        if VALIDATOR_AVAILABLE and self.enable_data_validation:
            self.data_validator = MultiRoundDataValidator(cfg)
            print("🔍 Data validator initialized")
    
    def _setup_dynamic_configs(self):
        """Setup configurations for dynamic pair switching."""
        self.pair_configs = {}
        
        # Parse round-specific configurations
        if hasattr(self.cfg.multiround, 'pair_configs'):
            for config_name, config_data in self.cfg.multiround.pair_configs.__dict__.items():
                if hasattr(config_data, 'pairs'):
                    pair_config = PairConfig(
                        margin_type=config_data.margin_type,
                        description=config_data.description,
                        train_path=config_data.pairs.train,
                        val_path=config_data.pairs.val,
                        test_path=config_data.pairs.test
                    )
                    self.pair_configs[config_name] = pair_config
        
        print(f"📊 Dynamic pair configs loaded: {list(self.pair_configs.keys())}")
        
        # Validate all configurations
        if self.check_files:
            self._validate_all_configs()
    
    def _setup_static_config(self):
        """Setup configuration for static pairs (original behavior)."""
        self.static_config = PairConfig(
            margin_type=getattr(self.cfg.paths, 'pair_margin', 'unknown'),
            description="Static preference pairs",
            train_path=self.cfg.paths.pairs.train,
            val_path=self.cfg.paths.pairs.val,
            test_path=self.cfg.paths.pairs.test
        )
        
        if self.check_files and not self.static_config.validate_paths():
            raise FileNotFoundError(f"Static pair files not found. Check paths in config.")
    
    def _validate_all_configs(self):
        """Validate all dynamic configurations."""
        for config_name, config in self.pair_configs.items():
            if not config.validate_paths():
                msg = f"Pair files missing for {config_name} (margin: {config.margin_type})"
                if self.fallback_enabled:
                    print(f"⚠️ Warning: {msg}. Will use fallback if needed.")
                else:
                    raise FileNotFoundError(f"{msg}. Fallback disabled.")
    
    def get_config_for_round(self, round_num: int) -> PairConfig:
        """Get the appropriate pair configuration for a given round."""
        if not self.dynamic_mode:
            return self.static_config
        
        # Dynamic mode: determine which config to use based on round
        if round_num <= 2:
            config_key = "rounds_1_2"
        elif round_num <= 5:
            config_key = "rounds_3_5"
        else:
            # For rounds beyond 5, use the last configuration
            config_key = "rounds_3_5"
        
        if config_key in self.pair_configs:
            config = self.pair_configs[config_key]
            if self.check_files and not config.validate_paths():
                return self._get_fallback_config()
            return config
        else:
            print(f"⚠️ No config found for {config_key}, using fallback")
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> PairConfig:
        """Get fallback configuration when dynamic configs fail."""
        if not self.fallback_enabled:
            raise RuntimeError("Dynamic pair loading failed and fallback is disabled")
        
        # Use margin125 as fallback (most reliable)
        fallback_config = PairConfig(
            margin_type=self.fallback_margin,
            description=f"Fallback to margin {self.fallback_margin}",
            train_path=f"data/pairs_margin{self.fallback_margin}/by_das/clean/train.clean.jsonl",
            val_path=f"data/pairs_margin{self.fallback_margin}/by_das/clean/val.clean.jsonl", 
            test_path=f"data/pairs_margin{self.fallback_margin}/by_das/clean/test.clean.jsonl"
        )
        
        if not fallback_config.validate_paths():
            raise FileNotFoundError(f"Fallback pair files not found for margin {self.fallback_margin}")
        
        print(f"🔄 Using fallback configuration: margin {self.fallback_margin}")
        return fallback_config
    
    def update_config_for_round(self, round_num: int) -> Dict[str, str]:
        """
        Update configuration for a specific round and return paths.
        
        Returns:
            dict: Paths for train/val/test pairs for this round
        """
        self.current_round = round_num
        self.current_config = self.get_config_for_round(round_num)
        
        # Log the configuration change
        print(f"📊 Round {round_num} pair configuration:")
        print(f"   Margin type: {self.current_config.margin_type}")
        print(f"   Description: {self.current_config.description}")
        
        # Count pairs for logging
        counts = self.current_config.get_pair_counts()
        print(f"   Pair counts: train={counts['train']}, val={counts['val']}, test={counts['test']}")
        
        # Validate data quality if validator is available
        if self.data_validator:
            print(f"🔍 Validating data quality for round {round_num}...")
            validation_result = self.data_validator.validate_pair_config(self.current_config)
            
            if not validation_result['overall_valid']:
                print(f"⚠️ Data quality issues detected:")
                for rec in validation_result['recommendations']:
                    print(f"   - {rec}")
                
                # Create filter script for problematic IDs
                if validation_result['problematic_ids']:
                    filter_script_path = os.path.join(
                        self.data_validator.validation_dir,
                        f"round_{round_num}_filter_script.py"
                    )
                    self.data_validator.create_length_mismatch_filter_script(filter_script_path)
                    print(f"📝 Filter script created: {filter_script_path}")
            else:
                print(f"✅ Data quality validation passed")
        
        return {
            'train': self.current_config.train_path,
            'val': self.current_config.val_path,
            'test': self.current_config.test_path
        }
    
    def get_current_margin_type(self) -> str:
        """Get the margin type for the current configuration."""
        if self.current_config:
            return self.current_config.margin_type
        return "unknown"
    
    def get_summary(self) -> Dict:
        """Get summary of pair provider configuration."""
        summary = {
            'dynamic_mode': self.dynamic_mode,
            'current_round': self.current_round,
            'current_margin_type': self.get_current_margin_type(),
            'fallback_enabled': self.fallback_enabled
        }
        
        if self.dynamic_mode:
            summary['available_configs'] = list(self.pair_configs.keys())
            summary['config_details'] = {
                name: {
                    'margin_type': config.margin_type,
                    'description': config.description,
                    'valid_paths': config.validate_paths()
                }
                for name, config in self.pair_configs.items()
            }
        
        return summary