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
    
    # Check pair files
    pair_files = ['train', 'val', 'test']
    for split in pair_files:
        if hasattr(cfg.paths.pairs, split):
            pair_path = getattr(cfg.paths.pairs, split)
            if not os.path.exists(pair_path):
                print(f"❌ Pair file not found: {pair_path}")
                return False
    
    # Validate multiround settings
    if not hasattr(cfg.multiround, 'num_rounds') or cfg.multiround.num_rounds < 1:
        print(f"❌ Invalid num_rounds: {getattr(cfg.multiround, 'num_rounds', 'missing')}")
        return False
    
    if not hasattr(cfg.multiround, 'epochs_per_round') or cfg.multiround.epochs_per_round < 1:
        print(f"❌ Invalid epochs_per_round: {getattr(cfg.multiround, 'epochs_per_round', 'missing')}")
        return False
    
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
    
    # Evaluation setup
    print(f"Eval samples (per round): {getattr(cfg.multiround, 'eval_samples', 8)}")
    print(f"Final eval samples: {getattr(cfg.multiround, 'final_eval_samples', 64)}")
    
    # Output
    output_root = getattr(cfg.multiround, 'output_root', 'runs/multiround')
    run_name = cfg.wandb.run_name or 'multiround_run'
    print(f"Output: {output_root}/{run_name}")
    
    # WandB
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