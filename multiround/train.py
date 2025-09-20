# multiround/train.py
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import argparse
import os
import sys
import yaml
import torch
from types import SimpleNamespace as SN
from datetime import datetime

from multiround.trainer import MultiRoundDPOTrainer
from multiround.utils import load_config_with_inheritance, deep_merge_dicts


def _to_sn(obj):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SN(**{k: _to_sn(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_sn(item) for item in obj]
    else:
        return obj


def load_multiround_config(config_path: str) -> SN:
    """
    Load multiround configuration with inheritance support.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        SimpleNamespace: Configuration object
    """
    # Load config with inheritance
    config_dict = load_config_with_inheritance(config_path)
    
    # Handle path resolution for pair selection
    if 'paths' in config_dict and 'pair_margin' in config_dict['paths']:
        margin = config_dict['paths']['pair_margin']
        margin_key = f"margin{margin}"
        
        if 'pairs' in config_dict['paths'] and margin_key in config_dict['paths']['pairs']:
            # Update active pairs to use the selected margin
            margin_pairs = config_dict['paths']['pairs'][margin_key]
            config_dict['paths']['pairs'].update(margin_pairs)
            print(f"🔧 Using preference pairs with {margin}*std margin")
    
    # Convert to SimpleNamespace for dot-access
    return _to_sn(config_dict)


def validate_config(cfg: SN) -> bool:
    """
    Validate multiround configuration.
    
    Args:
        cfg: Configuration object
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_sections = ['paths', 'multiround', 'dpo', 'training']
    for section in required_sections:
        if not hasattr(cfg, section):
            print(f"❌ Missing required config section: {section}")
            return False
    
    # Check required paths
    required_paths = ['processed_pt', 'split_pt', 'base_checkpoint']
    for path_key in required_paths:
        if not hasattr(cfg.paths, path_key):
            print(f"❌ Missing required path: {path_key}")
            return False
        
        path_value = getattr(cfg.paths, path_key)
        if not os.path.exists(path_value):
            print(f"❌ Required file not found: {path_value}")
            return False
    
    # Check pair files (conditional validation for dynamic vs static pairs)
    dynamic_pairs = getattr(cfg.multiround, 'dynamic_pairs', False)
    
    if dynamic_pairs:
        print("ℹ️ Dynamic pairs mode: skipping static pair file validation")
        # For dynamic pairs, we only need to validate that the base data files exist
        # The pair files will be generated dynamically during training
    else:
        # Static pairs mode: validate that all pair files exist
        pair_files = ['train', 'val', 'test']
        for split in pair_files:
            if hasattr(cfg.paths.pairs, split):
                pair_path = getattr(cfg.paths.pairs, split)
                if not os.path.exists(pair_path):
                    print(f"❌ Static pair file not found: {pair_path}")
                    print(f"💡 Hint: For dynamic pairs, set 'multiround.dynamic_pairs: true' in config")
                    return False
        print("✅ Static pair files validated")
    
    # Validate multiround settings
    if not hasattr(cfg.multiround, 'num_rounds') or cfg.multiround.num_rounds < 1:
        print(f"❌ Invalid num_rounds: {getattr(cfg.multiround, 'num_rounds', 'missing')}")
        return False
    
    if not hasattr(cfg.multiround, 'epochs_per_round') or cfg.multiround.epochs_per_round < 1:
        print(f"❌ Invalid epochs_per_round: {getattr(cfg.multiround, 'epochs_per_round', 'missing')}")
        return False
    
    # Validate pass@k configuration if present
    if hasattr(cfg, 'evaluation') and hasattr(cfg.evaluation, 'pass_k'):
        passk_cfg = cfg.evaluation.pass_k
        
        # Validate k_values
        if hasattr(passk_cfg, 'k_values'):
            k_values = passk_cfg.k_values
            if not isinstance(k_values, list) or not k_values:
                print(f"❌ Invalid pass@k k_values: must be non-empty list")
                return False
            
            # Check k_values are positive integers
            for k in k_values:
                if not isinstance(k, int) or k < 1:
                    print(f"❌ Invalid k value: {k} (must be positive integer)")
                    return False
            
            # Check k_values are reasonable (≤ final_eval_samples)
            max_k = max(k_values)
            final_samples = (getattr(cfg.multiround, 'final_eval_samples', None) or 
                           getattr(cfg.multiround, 'n_samples_final_eval', 64))
            if max_k > final_samples:
                print(f"⚠️ Warning: max k_value {max_k} > final_eval_samples {final_samples}")
        
        # Validate thresholds
        if hasattr(passk_cfg, 'thresholds'):
            thresholds = passk_cfg.thresholds
            
            # Check TM score thresholds
            if hasattr(thresholds, 'tm_score'):
                tm_thresholds = thresholds.tm_score
                if isinstance(tm_thresholds, list):
                    for tm_thr in tm_thresholds:
                        if not (0.0 <= tm_thr <= 1.0):
                            print(f"❌ Invalid TM threshold: {tm_thr} (must be 0.0-1.0)")
                            return False
            
            # Check RMSD thresholds
            if hasattr(thresholds, 'rmsd'):
                rmsd_thresholds = thresholds.rmsd
                if isinstance(rmsd_thresholds, list):
                    for rmsd_thr in rmsd_thresholds:
                        if rmsd_thr <= 0:
                            print(f"❌ Invalid RMSD threshold: {rmsd_thr} (must be positive)")
                            return False
            
            # Check MFE thresholds (more negative is better, so they should be negative)
            if hasattr(thresholds, 'mfe'):
                mfe_thresholds = thresholds.mfe
                if isinstance(mfe_thresholds, list):
                    for mfe_thr in mfe_thresholds:
                        if mfe_thr > 0:
                            print(f"⚠️ Warning: MFE threshold {mfe_thr} is positive (more negative usually indicates better stability)")
        
        print("✅ Pass@k configuration validated")
    
    print("✅ Configuration validation passed")
    return True


def print_config_summary(cfg: SN):
    """Print a summary of the configuration."""
    print(f"\n{'='*60}")
    print(f"📋 MULTIROUND DPO CONFIGURATION SUMMARY")
    print(f"{'='*60}")
    
    # Experiment info
    experiment_name = getattr(cfg, 'experiment', {})
    if hasattr(experiment_name, 'name'):
        print(f"Experiment: {experiment_name.name}")
        print(f"Description: {getattr(experiment_name, 'description', 'N/A')}")
    
    # Training setup
    print(f"Rounds: {cfg.multiround.num_rounds}")
    print(f"Epochs per round: {cfg.multiround.epochs_per_round}")
    print(f"Pair margin: {getattr(cfg.paths, 'pair_margin', 'unknown')}*std")
    print(f"DPO beta: {cfg.dpo.beta}")
    print(f"SFT lambda: {cfg.dpo.sft_lambda}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Learning rate: {cfg.optimizer.lr}")
    
    # Evaluation setup (handle both naming schemes)
    eval_samples = (getattr(cfg.multiround, 'eval_samples', None) or 
                   getattr(cfg.multiround, 'n_samples_eval', 8))
    final_eval_samples = (getattr(cfg.multiround, 'final_eval_samples', None) or 
                         getattr(cfg.multiround, 'n_samples_final_eval', 64))
    print(f"Eval samples (per round): {eval_samples}")
    print(f"Final eval samples: {final_eval_samples}")
    
    # Output (show actual enhanced run name that will be used)
    output_root = getattr(cfg.multiround, 'output_root', 'runs/multiround')
    # Import here to get the actual enhanced run name
    from multiround.wandb_manager import MultiRoundWandBManager
    wandb_manager = MultiRoundWandBManager(cfg)
    enhanced_run_name = wandb_manager.get_wandb_config()['name']
    print(f"Output: {output_root}/{enhanced_run_name}")
    print(f"   Enhanced run name: {enhanced_run_name}")
    print(f"   Base run name: {cfg.wandb.run_name or 'multiround_run'}")
    
    # WandB (reuse manager instance)
    print(f"WandB: {'enabled' if cfg.wandb.enable else 'disabled'}")
    if cfg.wandb.enable:
        print(f"  Project: {cfg.wandb.project}")
        print(f"  Run name: {cfg.wandb.run_name}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-round DPO training for RNA inverse folding")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to multiround configuration file"
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=None,
        help="Override WandB run name from config"
    )
    parser.add_argument(
        "--num_rounds", 
        type=int, 
        default=None,
        help="Override number of rounds from config"
    )
    parser.add_argument(
        "--epochs_per_round", 
        type=int, 
        default=None,
        help="Override epochs per round from config"
    )
    parser.add_argument(
        "--start_from_round", 
        type=int, 
        default=1,
        help="Start from specific round (for resuming)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Override device (cuda/cpu)"
    )
    parser.add_argument(
        "--wandb_mode", 
        type=str, 
        choices=["online", "offline", "disabled"],
        default=None,
        help="Override WandB mode"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Override output directory"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"📖 Loading configuration from: {args.config}")
    
    try:
        cfg = load_multiround_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)
    
    # Apply CLI overrides
    if args.run_name:
        cfg.wandb.run_name = args.run_name
    
    if args.num_rounds:
        cfg.multiround.num_rounds = args.num_rounds
    
    if args.epochs_per_round:
        cfg.multiround.epochs_per_round = args.epochs_per_round
    
    if args.start_from_round > 1:
        cfg.multiround.current_round = args.start_from_round
    
    if args.device:
        cfg.device = args.device
    
    if args.wandb_mode:
        if args.wandb_mode == "disabled":
            cfg.wandb.enable = False
        else:
            cfg.wandb.enable = True
            cfg.wandb.mode = args.wandb_mode
    
    if args.output_dir:
        cfg.multiround.output_root = args.output_dir
    
    # Validate configuration
    if not validate_config(cfg):
        print("❌ Configuration validation failed")
        sys.exit(1)
    
    # Print configuration summary
    print_config_summary(cfg)
    
    # Check CUDA availability
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available, falling back to CPU")
        cfg.device = "cpu"
    
    print(f"🖥️ Using device: {cfg.device}")
    
    # Create trainer and run
    try:
        print(f"🚀 Initializing MultiRound DPO Trainer...")
        trainer = MultiRoundDPOTrainer(cfg)
        
        print(f"🏃 Starting training...")
        start_time = datetime.now()
        
        # Run all rounds
        final_results = trainer.train_all_rounds()
        
        end_time = datetime.now()
        total_time = end_time - start_time
        
        print(f"\n🎉 Training completed successfully!")
        print(f"⏱️ Total time: {total_time}")
        
        # Print final summary
        if 'best_round_by_tm' in final_results:
            best = final_results['best_round_by_tm']
            print(f"🏆 Best round: {best['round']} (TM: {best['tm_mean']:.4f})")
        
        print(f"💾 Results saved to: {trainer.output_root}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()