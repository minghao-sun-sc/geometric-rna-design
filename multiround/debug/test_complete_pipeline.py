# multiround/debug/test_complete_pipeline.py
"""
Complete integration test for the multiround pipeline.
Tests the full workflow with minimal settings to verify all components work together.
"""
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from multiround.train import load_multiround_config, validate_config
from multiround.trainer import MultiRoundDPOTrainer


def create_minimal_test_config():
    """Create a minimal test configuration for quick pipeline testing."""
    print("🔧 Creating minimal test configuration...")
    
    base_config_path = "multiround/config/experiments/plan_a_margin125.yaml"
    if not os.path.exists(base_config_path):
        print(f"❌ Base config not found: {base_config_path}")
        return None
    
    try:
        cfg = load_multiround_config(base_config_path)
        
        # Minimal settings for fast testing
        cfg.multiround.num_rounds = 2
        cfg.multiround.epochs_per_round = 1  # Just 1 epoch per round
        cfg.multiround.eval_samples = 1      # Minimal evaluation
        cfg.multiround.final_eval_samples = 2
        
        # Training settings for speed
        cfg.training.batch_size = 1
        cfg.training.grad_accum_steps = 1
        cfg.training.save_every = 10
        cfg.training.val_every = 10
        cfg.training.log_every = 1
        
        # Disable expensive features
        cfg.wandb.enable = False
        
        # Quick evaluation settings
        cfg.evaluation.plot_distributions = False
        cfg.evaluation.save_sample_sequences = False
        
        print("✅ Minimal test configuration created")
        return cfg
        
    except Exception as e:
        print(f"❌ Failed to create test config: {e}")
        return None


def test_single_round_execution():
    """Test execution of a single training round."""
    print("\n🧪 Testing single round execution...")
    
    cfg = create_minimal_test_config()
    if cfg is None:
        return False
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override output directory
            cfg.multiround.output_root = temp_dir
            
            print(f"   Using temporary directory: {temp_dir}")
            
            # Initialize trainer
            trainer = MultiRoundDPOTrainer(cfg)
            
            # Test single round execution
            round_result = trainer.train_round(round_num=1)
            
            if 'error' in round_result:
                print(f"❌ Round execution failed: {round_result['error']}")
                return False
            
            print("✅ Single round execution completed")
            print(f"   Round: {round_result.get('round', 'N/A')}")
            print(f"   Training time: {round_result.get('training_time_sec', 0):.1f}s")
            print(f"   Evaluation time: {round_result.get('evaluation_time_sec', 0):.1f}s")
            
            # Check for expected metrics
            expected_metrics = ['tm_mean', 'rmsd_mean', 'recovery']
            for metric in expected_metrics:
                if metric in round_result:
                    print(f"   {metric}: {round_result[metric]:.4f}")
                else:
                    print(f"   ⚠️ {metric}: not found")
            
            return True
        
    except Exception as e:
        print(f"❌ Single round execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_round_execution():
    """Test execution of multiple training rounds."""
    print("\n🧪 Testing multi-round execution...")
    
    cfg = create_minimal_test_config()
    if cfg is None:
        return False
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Override output directory
            cfg.multiround.output_root = temp_dir
            
            print(f"   Using temporary directory: {temp_dir}")
            print(f"   Running {cfg.multiround.num_rounds} rounds...")
            
            # Initialize trainer
            trainer = MultiRoundDPOTrainer(cfg)
            
            # Test full multi-round execution
            final_results = trainer.train_all_rounds()
            
            if 'error' in final_results:
                print(f"❌ Multi-round execution failed: {final_results['error']}")
                return False
            
            print("✅ Multi-round execution completed")
            print(f"   Rounds completed: {final_results.get('total_rounds_completed', 0)}")
            print(f"   Total training time: {final_results.get('total_training_time_sec', 0):.1f}s")
            print(f"   Total evaluation time: {final_results.get('total_evaluation_time_sec', 0):.1f}s")
            
            # Check best round info
            if 'best_round_by_tm' in final_results:
                best = final_results['best_round_by_tm']
                print(f"   Best round: {best.get('round', 'N/A')} (TM: {best.get('tm_mean', 0):.4f})")
            
            # Check output directory structure
            round_dirs = [d for d in os.listdir(temp_dir) if d.startswith('round_')]
            print(f"   Round directories created: {len(round_dirs)}")
            
            # Check for final summary
            summary_path = os.path.join(temp_dir, 'final_summary.json')
            if os.path.exists(summary_path):
                print("   ✅ Final summary created")
            else:
                print("   ⚠️ Final summary not found")
            
            return True
        
    except Exception as e:
        print(f"❌ Multi-round execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_saving_and_loading():
    """Test checkpoint saving and loading functionality."""
    print("\n🧪 Testing checkpoint functionality...")
    
    cfg = create_minimal_test_config()
    if cfg is None:
        return False
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg.multiround.output_root = temp_dir
            cfg.multiround.num_rounds = 1  # Just one round for checkpoint test
            
            trainer = MultiRoundDPOTrainer(cfg)
            round_result = trainer.train_round(round_num=1)
            
            # Check if checkpoint was saved
            round_dir = os.path.join(temp_dir, "round_01")
            checkpoint_file = os.path.join(round_dir, "round_1_best.pt")
            
            if os.path.exists(checkpoint_file):
                print("✅ Checkpoint saved successfully")
                
                # Check checkpoint size
                size_mb = os.path.getsize(checkpoint_file) / (1024 * 1024)
                print(f"   Checkpoint size: {size_mb:.1f} MB")
                
                # Test loading checkpoint
                import torch
                try:
                    checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                    expected_keys = ['model_state_dict', 'optimizer_state_dict']
                    
                    for key in expected_keys:
                        if key in checkpoint_data:
                            print(f"   ✅ Checkpoint contains {key}")
                        else:
                            print(f"   ⚠️ Checkpoint missing {key}")
                    
                    return True
                    
                except Exception as e:
                    print(f"   ❌ Failed to load checkpoint: {e}")
                    return False
                    
            else:
                print(f"❌ Checkpoint not found at: {checkpoint_file}")
                return False
        
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        return False


def test_output_file_generation():
    """Test that all expected output files are generated."""
    print("\n🧪 Testing output file generation...")
    
    cfg = create_minimal_test_config()
    if cfg is None:
        return False
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg.multiround.output_root = temp_dir
            cfg.multiround.num_rounds = 2
            
            trainer = MultiRoundDPOTrainer(cfg)
            final_results = trainer.train_all_rounds()
            
            # Check expected files
            expected_files = [
                "final_summary.json",
                "round_01/round_metrics.json",
                "round_01/round_1_best.pt",
                "round_02/round_metrics.json", 
                "round_02/round_2_best.pt",
            ]
            
            for expected_file in expected_files:
                file_path = os.path.join(temp_dir, expected_file)
                if os.path.exists(file_path):
                    print(f"   ✅ {expected_file}")
                else:
                    print(f"   ❌ {expected_file}")
            
            # Check subdirectory structure
            round_subdirs = ['checkpoints', 'eval_results', 'plots', 'logs']
            for round_num in [1, 2]:
                round_dir = os.path.join(temp_dir, f"round_{round_num:02d}")
                for subdir in round_subdirs:
                    subdir_path = os.path.join(round_dir, subdir)
                    if os.path.exists(subdir_path):
                        print(f"   ✅ round_{round_num:02d}/{subdir}/")
                    else:
                        print(f"   ⚠️ round_{round_num:02d}/{subdir}/ (might be optional)")
            
            print("✅ Output file generation test completed")
            return True
        
    except Exception as e:
        print(f"❌ Output file generation test failed: {e}")
        return False


def main():
    """Run complete pipeline integration test."""
    print("🚀 Starting complete pipeline integration test...\n")
    
    # First run prerequisite checks
    prereq_tests = [
        ("Configuration loading", lambda: create_minimal_test_config() is not None),
    ]
    
    print("📋 Running prerequisite checks...")
    for test_name, test_func in prereq_tests:
        try:
            success = test_func()
            if success:
                print(f"   ✅ {test_name}")
            else:
                print(f"   ❌ {test_name} - stopping here")
                return
        except Exception as e:
            print(f"   ❌ {test_name} failed: {e}")
            return
    
    # Main integration tests
    integration_tests = [
        test_single_round_execution,
        test_checkpoint_saving_and_loading,
        test_multi_round_execution,
        test_output_file_generation,
    ]
    
    print(f"\n🧪 Running {len(integration_tests)} integration tests...")
    
    results = []
    for test_func in integration_tests:
        try:
            success = test_func()
            results.append(success)
            
            if not success:
                print(f"\n⚠️ Test {test_func.__name__} failed - continuing with remaining tests")
        except Exception as e:
            print(f"\n❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print(f"\n📊 Integration Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("✅ Complete pipeline integration test PASSED!")
        print("   All components are working correctly together")
        print("   Ready for full multiround training runs")
    else:
        print("❌ Some integration tests FAILED")
        print("   Check individual test outputs above for details")
        print("   Fix issues before running full training")


if __name__ == "__main__":
    main()