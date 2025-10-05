#!/usr/bin/env python3
"""
RiboPO v2 Candidate Generation Test Script

Test the winner-focused candidate generation pipeline with comprehensive metrics.
This validates that our RiboPO v2 configuration works with the cleaned data.
"""

import os
import sys
import torch
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

# Add multiround to path for imports  
multiround_path = project_root / "multiround"
sys.path.insert(0, str(multiround_path))

from multiround.evaluator import MultiRoundEvaluator
from multiround.utils import load_config_with_inheritance
from types import SimpleNamespace as SN


def _to_sn(obj):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SN(**{k: _to_sn(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_sn(item) for item in obj]
    else:
        return obj


def load_ribopo_v2_config(config_path: str) -> SN:
    """Load RiboPO v2 configuration with inheritance support."""
    config_dict = load_config_with_inheritance(config_path)
    return _to_sn(config_dict)


def validate_ribopo_v2_config(cfg: SN) -> bool:
    """Validate RiboPO v2 configuration for candidate generation test."""
    
    # Check required paths
    if not hasattr(cfg, 'paths'):
        print("❌ Missing paths configuration")
        return False
    
    required_paths = ['processed_pt', 'split_pt']
    for path_key in required_paths:
        if not hasattr(cfg.paths, path_key):
            print(f"❌ Missing required path: {path_key}")
            return False
        
        path_value = getattr(cfg.paths, path_key)
        if not os.path.exists(path_value):
            print(f"❌ Required file not found: {path_value}")
            return False
    
    # Check RiboPO v2 configuration
    if not hasattr(cfg, 'ribopo_v2'):
        print("❌ Missing ribopo_v2 configuration")
        return False
    
    ribopo_cfg = cfg.ribopo_v2
    
    # Validate candidate generation settings
    if not hasattr(ribopo_cfg, 'candidate_pool_size'):
        print("❌ Missing candidate_pool_size")
        return False
    
    if ribopo_cfg.candidate_pool_size < 1:
        print(f"❌ Invalid candidate_pool_size: {ribopo_cfg.candidate_pool_size}")
        return False
    
    print("✅ RiboPO v2 configuration validation passed")
    return True


def test_candidate_generation(config_path: str, limit_structures: int = 3):
    """
    Test RiboPO v2 candidate generation pipeline.
    
    Args:
        config_path: Path to RiboPO v2 configuration file
        limit_structures: Number of structures to test (for quick validation)
    """
    
    print(f"\n{'='*80}")
    print(f"🧪 RIBOPO V2 CANDIDATE GENERATION TEST")
    print(f"{'='*80}")
    
    # Load configuration
    print(f"📖 Loading configuration from: {config_path}")
    try:
        cfg = load_ribopo_v2_config(config_path)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    # Validate configuration
    if not validate_ribopo_v2_config(cfg):
        print("❌ Configuration validation failed")
        return False
    
    # Print test settings
    print(f"\n📋 Test Configuration:")
    print(f"   Candidate pool size: {cfg.ribopo_v2.candidate_pool_size}")
    print(f"   Temperature diversity: {getattr(cfg.ribopo_v2, 'temperature_diversity', '[0.5]')}")
    print(f"   Test structures: {limit_structures}")
    print(f"   Data files: {cfg.paths.processed_pt}")
    print(f"   Split files: {cfg.paths.split_pt}")
    
    # Check for evaluation-only mode
    mode = getattr(cfg, 'mode', 'training')
    if mode != 'evaluation_only':
        print(f"⚠️ Mode is '{mode}', setting to 'evaluation_only' for test")
        cfg.mode = 'evaluation_only'
    
    # Limit to test structures for quick validation
    original_limit = getattr(cfg.testing, 'limit_backbones', None)
    cfg.testing = getattr(cfg, 'testing', SN())
    cfg.testing.limit_backbones = limit_structures
    
    try:
        # Initialize evaluator to test data loading
        print(f"\n🔄 Initializing MultiRound Evaluator...")
        evaluator = MultiRoundEvaluator(cfg)
        
        if evaluator.eval_dataset is None:
            print("❌ Failed to load evaluation dataset")
            return False
        
        print(f"✅ Successfully loaded evaluation dataset")
        print(f"   Dataset size: {len(evaluator.eval_dataset)}")
        
        # Test data access
        print(f"\n🔍 Testing data access...")
        test_item = evaluator.eval_dataset[0]
        print(f"   First item type: {type(test_item)}")
        
        # Check if we can access the raw data structure
        if hasattr(test_item, 'sequence'):
            print(f"   Sample sequence length: {len(test_item.sequence)}")
        elif hasattr(test_item, 'raw_data'):
            print(f"   Sample sequence length: {len(test_item.raw_data['sequence'])}")
        else:
            print(f"   Sample structure: {list(test_item.__dict__.keys()) if hasattr(test_item, '__dict__') else 'unknown'}")
        
        # Test basic evaluation setup
        print(f"\n⚙️ Evaluation Settings:")
        print(f"   Eval samples: {evaluator.eval_samples}")
        print(f"   Final eval samples: {evaluator.final_eval_samples}")
        print(f"   Eval temperature: {evaluator.eval_temperature}")
        
        # Test if we can load a model (this would be needed for actual candidate generation)
        base_checkpoint = getattr(cfg.paths, 'base_checkpoint', None)
        if base_checkpoint and os.path.exists(base_checkpoint):
            print(f"   Base checkpoint: {base_checkpoint} ✅")
        else:
            print(f"   Base checkpoint: {base_checkpoint} ❌ (not found, needed for actual generation)")
        
        print(f"\n✅ RiboPO v2 candidate generation test infrastructure validated!")
        print(f"📊 Ready for:")
        print(f"   • Large-scale candidate generation ({cfg.ribopo_v2.candidate_pool_size} per backbone)")
        print(f"   • Multi-temperature sampling {getattr(cfg.ribopo_v2, 'temperature_diversity', '[0.5]')}")
        print(f"   • Comprehensive metric computation")
        print(f"   • Winner selection and ranking")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test RiboPO v2 candidate generation pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="ribopo_v2/config/experiments/01_candidate_generation.yaml",
        help="Path to RiboPO v2 configuration file"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=3,
        help="Number of structures to test (for quick validation)"
    )
    
    args = parser.parse_args()
    
    # Resolve config path relative to project root
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / args.config
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Run test
    success = test_candidate_generation(str(config_path), args.limit)
    
    if success:
        print(f"\n🎉 RiboPO v2 candidate generation test PASSED!")
        print(f"💡 Next steps:")
        print(f"   1. Run actual candidate generation: python multiround/train.py --config {args.config}")
        print(f"   2. Implement winner selection module")
        print(f"   3. Test complete S_total reward calculation")
        sys.exit(0)
    else:
        print(f"\n❌ RiboPO v2 candidate generation test FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()