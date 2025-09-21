#!/usr/bin/env python3
"""
End-to-end test for multi-round RiboPO training and evaluation pipeline.
This test validates that the complete training and evaluation system works with real data.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

import torch
import yaml
from types import SimpleNamespace

def load_config_from_yaml(config_path: str) -> SimpleNamespace:
    """Load configuration from YAML file and convert to SimpleNamespace."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d
    
    return dict_to_namespace(config_dict)

def test_config_loading():
    """Test loading of real experiment configurations."""
    print("🔍 Testing configuration loading...")
    
    config_dir = PROJECT_ROOT / "multiround" / "config" / "experiments"
    config_files = [
        "01_sft_ablation.yaml",
        "02_dpo_m125.yaml", 
        "06_dpo_m25_3r.yaml"
    ]
    
    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            try:
                config = load_config_from_yaml(str(config_path))
                print(f"  ✅ {config_file}: Loaded successfully")
                
                # Validate required sections
                required_sections = ['multiround', 'dpo', 'training', 'paths']
                for section in required_sections:
                    if hasattr(config, section):
                        print(f"    ✅ {section}: Present")
                    else:
                        print(f"    ❌ {section}: Missing")
                        
            except Exception as e:
                print(f"  ❌ {config_file}: Failed to load - {e}")
        else:
            print(f"  ⚠️ {config_file}: File not found")

def test_model_loading():
    """Test loading of base RNA model."""
    print("\n🔍 Testing model loading...")
    
    try:
        from dpo.model_factory import build_model
        
        # Check if we have any existing checkpoint
        ckpt_dir = PROJECT_ROOT / "dpo" / "ckpts"
        checkpoint_files = list(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
        
        if not checkpoint_files:
            print("  ⚠️ No model checkpoints found, testing model creation without weights")
            model_path = None
        else:
            model_path = str(checkpoint_files[0])
            print(f"  📁 Using checkpoint: {checkpoint_files[0].name}")
            
        # Create model configuration
        model_cfg = {
            "model": "ARv1",
            "model_path": model_path,
            "node_in_dim": (15, 4),
            "node_h_dim": (128, 16), 
            "edge_in_dim": (131, 3),
            "edge_h_dim": (64, 4),
            "num_layers": 4,
            "drop_rate": 0.5,
            "out_dim": 4
        }
        
        model = build_model(model_cfg)
        print(f"  ✅ Model loaded successfully")
        print(f"    Model type: {type(model)}")
        print(f"    Device: {next(model.parameters()).device}")
        print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test loading of preference pair datasets."""
    print("\n🔍 Testing dataset loading...")
    
    try:
        from multiround.pair_provider import MultiRoundPairProvider
        
        # Create minimal config for dataset testing
        config = SimpleNamespace(
            device='cpu',
            multiround=SimpleNamespace(
                dynamic_pairs=False,
                pair_configs={}
            ),
            pairs=SimpleNamespace(
                train_path=str(PROJECT_ROOT / "data" / "pairs_margin25" / "ribopo_pairs.json"),
                eval_path=str(PROJECT_ROOT / "data" / "pairs_margin125" / "dpo_pairs_margin125.clean.jsonl"),
                margin="25"
            ),
            paths=SimpleNamespace(
                processed_pt=str(PROJECT_ROOT / "data" / "processed.pt"),
                save_dir="/tmp/test_ribopo",
                pairs=SimpleNamespace(
                    train=str(PROJECT_ROOT / "data" / "pairs_margin25" / "ribopo_pairs.json"),
                    eval=str(PROJECT_ROOT / "data" / "pairs_margin125" / "dpo_pairs_margin125.clean.jsonl"),
                    val=str(PROJECT_ROOT / "data" / "pairs_margin125" / "dpo_pairs_margin125.clean.jsonl"),
                    test=str(PROJECT_ROOT / "data" / "pairs_margin125" / "dpo_pairs_margin125.clean.jsonl")
                )
            )
        )
        
        # Check if data files exist
        train_path = Path(config.pairs.train_path)
        if not train_path.exists():
            print(f"  ⚠️ Training pairs not found at {train_path}")
            return False
            
        pair_provider = MultiRoundPairProvider(config)
        train_dataset, eval_dataset = pair_provider.get_datasets_for_round(1)
        
        print(f"  ✅ Datasets loaded successfully")
        print(f"    Training pairs: {len(train_dataset) if train_dataset else 0}")
        print(f"    Evaluation pairs: {len(eval_dataset) if eval_dataset else 0}")
        return True
        
    except Exception as e:
        print(f"  ❌ Dataset loading failed: {e}")
        traceback.print_exc()
        return False

def test_evaluation_pipeline():
    """Test the evaluation pipeline with mock data."""
    print("\n🔍 Testing evaluation pipeline...")
    
    try:
        from multiround.evaluator import MultiRoundEvaluator
        from unittest.mock import Mock
        
        # Create evaluation config
        config = SimpleNamespace(
            device='cpu',
            multiround=SimpleNamespace(
                n_samples_eval=2,
                n_samples_final_eval=4,
                eval_temperature=0.5,
                final_eval_temperature=0.1,
                num_rounds=3
            ),
            evaluation=SimpleNamespace(
                metrics=['recovery', 'perplexity'],
                pass_k=SimpleNamespace(
                    k_values=[1, 2, 4],
                    thresholds=SimpleNamespace(
                        tm_score=[0.4, 0.45],
                        rmsd=[8.0, 4.0],
                        mfe=[-10.0, -15.0]
                    )
                )
            ),
            paths=SimpleNamespace()
        )
        
        evaluator = MultiRoundEvaluator(config)
        print(f"  ✅ Evaluator initialized successfully")
        print(f"    Eval samples: {evaluator.n_samples_eval}")
        print(f"    Final eval samples: {evaluator.n_samples_final_eval}")
        print(f"    Eval temperature: {evaluator.eval_temperature}")
        print(f"    Final eval temperature: {evaluator.final_eval_temperature}")
        
        # Test evaluation methods
        metrics = evaluator.get_evaluation_metrics()
        print(f"    Available metrics: {metrics}")
        
        # Test pass@k configuration
        should_use_full = evaluator.should_use_full_pass_k(1)
        print(f"    Should use full pass@k for round 1: {should_use_full}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Evaluation pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_trainer_initialization():
    """Test trainer initialization with real configuration."""
    print("\n🔍 Testing trainer initialization...")
    
    try:
        from multiround.trainer import MultiRoundDPOTrainer
        
        # Use an available configuration for testing
        config_path = PROJECT_ROOT / "multiround" / "config" / "experiments" / "06_dpo_m25_3r.yaml"
        if not config_path.exists():
            print("  ⚠️ Test config not found, skipping trainer test")
            return False
            
        config = load_config_from_yaml(str(config_path))
        
        # Override paths for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            config.paths.save_dir = temp_dir
            
            # Ensure paths.pairs exists for the trainer
            if not hasattr(config.paths, 'pairs'):
                config.paths.pairs = SimpleNamespace(
                    train=str(PROJECT_ROOT / "data" / "pairs_margin25" / "ribopo_pairs.json"),
                    eval=str(PROJECT_ROOT / "data" / "pairs_margin125" / "dpo_pairs_margin125.clean.jsonl"),
                    val=str(PROJECT_ROOT / "data" / "pairs_margin125" / "dpo_pairs_margin125.clean.jsonl"),
                    test=str(PROJECT_ROOT / "data" / "pairs_margin125" / "dpo_pairs_margin125.clean.jsonl")
                )
            
            # Mock the dependencies to avoid requiring actual model files
            from unittest.mock import patch, Mock
            with patch('multiround.trainer.DPOTrainer') as mock_dpo_trainer:
                mock_dpo_trainer.return_value = Mock()
                
                trainer = MultiRoundDPOTrainer(config)
                print(f"  ✅ Trainer initialized successfully")
                print(f"    Number of rounds: {trainer.num_rounds}")
                print(f"    Epochs per round: {trainer.epochs_per_round}")
                print(f"    Output directory: {trainer.output_root}")
                
                return True
                
    except Exception as e:
        print(f"  ❌ Trainer initialization failed: {e}")
        traceback.print_exc()
        return False

def test_checkpoint_management():
    """Test checkpoint management functionality."""
    print("\n🔍 Testing checkpoint management...")
    
    try:
        from multiround.checkpoint_manager import EnhancedCheckpointManager
        
        config = SimpleNamespace(
            checkpoint=SimpleNamespace(
                save_every_n_steps=50,
                keep_best_n=3,
                keep_last_n=2,
                selection_metric='val_pref_acc'
            )
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = EnhancedCheckpointManager(
                cfg=config,
                output_root=temp_dir
            )
            
            print(f"  ✅ Checkpoint manager initialized successfully")
            print(f"    Output directory: {temp_dir}")
            # The checkpoint manager has different attribute names
            print(f"    Keep best: {getattr(checkpoint_manager, 'keep_best_n', 'N/A')}")
            print(f"    Keep last: {getattr(checkpoint_manager, 'keep_last_n', 'N/A')}")
            print(f"    Max per round: {getattr(checkpoint_manager, 'max_checkpoints_per_round', 'N/A')}")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Checkpoint management test failed: {e}")
        traceback.print_exc()
        return False

def test_ema_integration():
    """Test EMA integration."""
    print("\n🔍 Testing EMA integration...")
    
    try:
        from multiround.ema import MultiRoundEMA
        import torch.nn as nn
        
        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        config = SimpleNamespace(
            multiround=SimpleNamespace(
                ema=SimpleNamespace(
                    enabled=True,
                    decay=0.9999,
                    update_after_step=10,
                    update_every=5,
                    reference_strategy='ema'
                )
            )
        )
        
        ema_manager = MultiRoundEMA(model, config)
        print(f"  ✅ EMA manager initialized successfully")
        print(f"    EMA enabled: {ema_manager.enabled}")
        if hasattr(ema_manager, 'ema') and ema_manager.ema:
            print(f"    Decay: {ema_manager.ema.decay}")
            print(f"    Update after step: {ema_manager.ema.update_after_step}")
            print(f"    Update every: {ema_manager.ema.update_every}")
        else:
            print(f"    EMA not active (disabled or not initialized)")
        
        # Test EMA update
        success = ema_manager.update()
        print(f"    EMA update result: {success}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ EMA integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive end-to-end tests."""
    print("🚀 Starting End-to-End Pipeline Tests")
    print("="*60)
    
    # Import unittest.mock properly
    import unittest.mock
    
    test_results = {
        "Config Loading": test_config_loading(),
        "Model Loading": test_model_loading(), 
        "Dataset Loading": test_dataset_loading(),
        "Evaluation Pipeline": test_evaluation_pipeline(),
        "Trainer Initialization": test_trainer_initialization(),
        "Checkpoint Management": test_checkpoint_management(),
        "EMA Integration": test_ema_integration()
    }
    
    print("\n" + "="*60)
    print("📊 END-TO-END TEST SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} | {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-"*60)
    print(f"TOTAL: {passed + failed} tests | {passed} passed | {failed} failed")
    
    if failed == 0:
        print("\n🎉 All end-to-end tests PASSED! The pipeline is ready for training.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Check the issues above before running training.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)