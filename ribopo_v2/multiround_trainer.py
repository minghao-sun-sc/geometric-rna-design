#!/usr/bin/env python3
"""
RiboPO v2 Multi-Round Training Script

Integrates winner-focused DPO training with multi-round reference model updates
and dynamic preference pair switching.

Key features:
- Multi-round training with reference model updates
- Dynamic preference pair switching (rounds 1-2: 0.25*std, rounds 3-5: 0.125*std)
- Winner-focused candidate generation and selection
- Pass@K evaluation on rounds 1, 3, 5
- Comprehensive model selection using pass@8 with TM >= 0.45
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import logging

import torch
import wandb

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

from multiround.utils import load_config_with_inheritance
from ribopo_v2.candidate_evaluation import RiboPOv2CandidateEvaluator
from ribopo_v2.winner_selection import WinnerSelector, WinnerConfig
from multiround.trainer import MultiRoundDPOTrainer  # Existing multiround infrastructure
from multiround.evaluator import MultiRoundEvaluator
from dpo.bench.eval_full import eval_full_metrics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce verbosity for specific modules to focus on errors
logging.getLogger('ribopo_v2.candidate_evaluation').setLevel(logging.WARNING)
logging.getLogger('WinnerSelector').setLevel(logging.WARNING)


class RiboPOv2MultiRoundTrainer:
    """
    RiboPO v2 Multi-Round Training Coordinator
    
    Manages the complete multi-round training pipeline including:
    - Winner selection and preference pair generation
    - Reference model updates
    - Dynamic pair switching
    - Model selection and evaluation
    """
    
    def __init__(self, config_path: str):
        """Initialize the multi-round trainer."""
        config_dict = load_config_with_inheritance(config_path)
        
        # Convert dict to object with dot notation access recursively
        from types import SimpleNamespace
        
        def dict_to_namespace(d):
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [dict_to_namespace(i) for i in d]
            else:
                return d
        
        self.config = dict_to_namespace(config_dict)
        self.setup_paths()
        self.setup_logging()
        
        # Initialize components
        self.candidate_evaluator = None
        self.winner_selector = None
        self.multiround_trainer = None
        self.evaluator = None
        
        # Training state
        self.current_round = 1
        self.best_models = {}  # Track best model per round
        self.training_history = {}  # Track training metrics
        
        logger.info(f"Initialized RiboPO v2 Multi-Round Trainer")
        logger.info(f"Configuration: {self.config.experiment.name}")
        logger.info(f"Total rounds: {self.config.multiround.num_rounds}")
        
    def setup_paths(self):
        """Set up all required paths and directories."""
        self.base_dir = Path(self.config.paths.project_root)
        self.output_dir = self.base_dir / self.config.multiround.output_root
        self.pairs_dir = self.base_dir / self.config.paths.pairs_dir
        self.winners_dir = self.base_dir / self.config.paths.winners_dir
        
        # Create directories
        for dir_path in [self.output_dir, self.pairs_dir, self.winners_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Round-specific directories
        self.round_dirs = {}
        for round_num in range(1, self.config.multiround.num_rounds + 1):
            round_dir = self.output_dir / f"round_{round_num}"
            round_dir.mkdir(parents=True, exist_ok=True)
            self.round_dirs[round_num] = round_dir
            
    def setup_logging(self):
        """Set up WandB logging."""
        if self.config.wandb.enable:
            wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=self.config.wandb.run_name,
                tags=self.config.wandb.tags,
                notes=self.config.wandb.notes,
                group=self.config.wandb.group,
                config=self._namespace_to_dict(self.config)
            )
    
    def _namespace_to_dict(self, obj):
        """Convert SimpleNamespace objects to dicts recursively."""
        from types import SimpleNamespace
        
        if isinstance(obj, SimpleNamespace):
            return {k: self._namespace_to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, list):
            return [self._namespace_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._namespace_to_dict(v) for k, v in obj.items()}
        else:
            return obj
    
    def _copy_namespace(self, obj):
        """Create a deep copy of SimpleNamespace objects."""
        import copy
        from types import SimpleNamespace
        
        if isinstance(obj, SimpleNamespace):
            return SimpleNamespace(**{k: self._copy_namespace(v) for k, v in vars(obj).items()})
        elif isinstance(obj, list):
            return [self._copy_namespace(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._copy_namespace(v) for k, v in obj.items()}
        else:
            return copy.deepcopy(obj)
            
    def initialize_components(self):
        """Initialize all training components."""
        logger.info("Initializing training components...")
        
        # Initialize candidate evaluator
        self.candidate_evaluator = RiboPOv2CandidateEvaluator(self.config)
        
        # Initialize winner selector with dynamic configuration
        winner_config = WinnerConfig(
            min_tm_threshold=self.config.ribopo_v2.winner_gates.tm_min,
            max_rmsd_threshold=self.config.ribopo_v2.winner_gates.rmsd_max,
            min_plddt_threshold=getattr(self.config.ribopo_v2.winner_gates, 'min_plddt', 0.6),
            tm_weight=self.config.ribopo_v2.reward_formula.tm_weight,
            inf_weight=self.config.ribopo_v2.reward_formula.inf_weight,
            ed_weight=self.config.ribopo_v2.reward_formula.ed_weight
        )
        self.winner_selector = WinnerSelector(winner_config)
        
        # Initialize multiround trainer
        self.multiround_trainer = MultiRoundDPOTrainer(self.config)
        
        # Initialize evaluator
        self.evaluator = MultiRoundEvaluator(self.config)
        
        logger.info("All components initialized successfully")
        
    def get_pair_margin_for_round(self, round_num: int) -> str:
        """Get the appropriate pair margin setting for the current round."""
        if hasattr(self.config.multiround, 'dynamic_pairs') and self.config.multiround.dynamic_pairs:
            if round_num <= 2:
                return "0.25std"  # Use 0.25*std pairs for rounds 1-2
            else:
                return "0.125std"  # Use 0.125*std pairs for rounds 3-5
        else:
            return "0.125std"  # Default to more strict pairs
            
    def generate_winners_and_pairs(self, round_num: int, model_path: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate winners and preference pairs for the current round.
        
        Args:
            round_num: Current training round
            model_path: Path to model checkpoint (None for base model)
            
        Returns:
            Tuple of (winners_file, pairs_file) paths
        """
        logger.info(f"Generating winners and pairs for round {round_num}")
        
        # Set up round-specific configuration
        round_config = self._copy_namespace(self.config)
        margin_setting = self.get_pair_margin_for_round(round_num)
        
        # Configure candidate generation for this round
        if model_path is not None:
            round_config.paths.model_checkpoint = model_path
            logger.info(f"Using model checkpoint: {model_path}")
        else:
            logger.info("Using base gRNAde checkpoint")
            
        # Generate candidates and evaluate
        logger.info(f"Generating {self.config.ribopo_v2.candidate_pool_size} candidates per backbone")
        
        output_dir = self.round_dirs[round_num] / "candidate_generation"
        output_dir.mkdir(exist_ok=True)
        
        # Run candidate evaluation
        results = self.candidate_evaluator.run_candidate_evaluation(
            limit_backbones=None,  # Evaluate all backbones
            output_dir=str(output_dir)
        )
        
        # Load already-generated winners and pairs from candidate evaluation
        logger.info(f"Loading winners and pairs generated by candidate evaluation")
        
        # Load winners from the saved file
        winners_source = output_dir / "winners.json"
        winners_file = self.winners_dir / f"winners_round_{round_num}.json"
        
        # Copy winners file to expected location
        import shutil
        shutil.copy2(str(winners_source), str(winners_file))
        
        # Load pairs from the saved file  
        pairs_source = output_dir / "preference_pairs.jsonl"
        pairs_file = self.pairs_dir / f"pairs_round_{round_num}.jsonl"
        
        # Copy pairs file to expected location
        shutil.copy2(str(pairs_source), str(pairs_file))
        
        # Log statistics
        if self.config.wandb.enable:
            wandb.log({
                f"round_{round_num}/total_candidates": results.get('total_candidates', 0),
                f"round_{round_num}/total_winners": results.get('total_winners', 0),
                f"round_{round_num}/total_pairs": results.get('total_pairs', 0),
                f"round_{round_num}/margin_setting": margin_setting,
            })
            
        logger.info(f"Loaded {results.get('total_winners', 0)} winners and {results.get('total_pairs', 0)} pairs for round {round_num}")
        
        return str(winners_file), str(pairs_file)
        
    def update_reference_model(self, round_num: int, best_model_path: str):
        """Update the reference model for the next round."""
        if not self.config.multiround.update_reference:
            logger.info("Reference model update disabled")
            return
            
        logger.info(f"Updating reference model for round {round_num + 1}")
        
        # Copy the best model from current round to be the reference for next round
        reference_dir = self.round_dirs[round_num + 1] if round_num + 1 <= self.config.multiround.num_rounds else None
        
        if reference_dir:
            reference_path = reference_dir / "reference_model.pt"
            shutil.copy2(best_model_path, reference_path)
            logger.info(f"Reference model updated: {reference_path}")
            
            # Update configuration for next round
            self.config.paths.reference_model = str(reference_path)
            
            if self.config.wandb.enable:
                wandb.log({
                    f"round_{round_num}/reference_updated": True,
                    f"round_{round_num}/reference_source": best_model_path
                })
                
    def select_best_model(self, round_num: int, candidate_models: List[str]) -> str:
        """
        Select the best model from candidates using pass@8 with TM >= 0.45.
        
        Args:
            round_num: Current round number
            candidate_models: List of candidate model checkpoint paths
            
        Returns:
            Path to the selected best model
        """
        logger.info(f"Selecting best model for round {round_num} from {len(candidate_models)} candidates")
        
        best_model = None
        best_score = -1
        best_metrics = None
        
        # Evaluate each candidate model
        for model_path in candidate_models:
            logger.info(f"Evaluating candidate model: {model_path}")
            
            try:
                # Run evaluation with pass@8
                eval_results = self.evaluator.evaluate_checkpoint(
                    model_path,
                    n_samples=8,
                    temperature=0.5,
                    save_designs=False
                )
                
                # Calculate pass@8 with TM >= 0.45
                tm_scores = eval_results.get('sc_score_tm', [])
                if len(tm_scores) > 0:
                    pass_8_tm_045 = np.mean(np.array(tm_scores) >= 0.45)
                    
                    # Tie-breaker: lower (better) MFE
                    mfe_scores = eval_results.get('vienna_mfe', [])
                    tie_breaker = -np.mean(mfe_scores) if len(mfe_scores) > 0 else 0
                    
                    # Combined score for selection
                    combined_score = pass_8_tm_045 + 0.001 * tie_breaker  # Small weight for tie-breaker
                    
                    logger.info(f"Model {Path(model_path).name}: pass@8_TM≥0.45={pass_8_tm_045:.3f}, MFE_mean={np.mean(mfe_scores):.2f}")
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_model = model_path
                        best_metrics = {
                            'pass_8_tm_045': pass_8_tm_045,
                            'mfe_mean': np.mean(mfe_scores) if len(mfe_scores) > 0 else 0,
                            'tm_mean': np.mean(tm_scores),
                            'combined_score': combined_score
                        }
                        
            except Exception as e:
                logger.warning(f"Failed to evaluate model {model_path}: {e}")
                continue
                
        if best_model is None:
            logger.warning("No valid model found, using first candidate")
            best_model = candidate_models[0]
            best_metrics = {'pass_8_tm_045': 0, 'mfe_mean': 0, 'tm_mean': 0, 'combined_score': 0}
        
        logger.info(f"Selected best model: {Path(best_model).name}")
        logger.info(f"Best model metrics: {best_metrics}")
        
        # Log to wandb
        if self.config.wandb.enable:
            wandb.log({
                f"round_{round_num}/best_model_pass_8_tm_045": best_metrics['pass_8_tm_045'],
                f"round_{round_num}/best_model_mfe_mean": best_metrics['mfe_mean'],
                f"round_{round_num}/best_model_tm_mean": best_metrics['tm_mean'],
                f"round_{round_num}/best_model_combined_score": best_metrics['combined_score']
            })
            
        # Store best model info
        self.best_models[round_num] = {
            'model_path': best_model,
            'metrics': best_metrics
        }
        
        return best_model
        
    def train_round(self, round_num: int) -> str:
        """
        Train a single round and return the path to the best model.
        
        Args:
            round_num: Round number (1-based)
            
        Returns:
            Path to the best model checkpoint from this round
        """
        logger.info(f"Starting training for round {round_num}/{self.config.multiround.num_rounds}")
        
        # Generate winners and pairs for this round
        if round_num == 1:
            # First round uses base model
            winners_file, pairs_file = self.generate_winners_and_pairs(round_num)
        else:
            # Subsequent rounds use best model from previous round
            prev_best_model = self.best_models[round_num - 1]['model_path']
            winners_file, pairs_file = self.generate_winners_and_pairs(round_num, prev_best_model)
            
        # Update configuration for this round
        round_config = self._copy_namespace(self.config)
        round_config.multiround.current_round = round_num
        round_config.paths.pairs.train = pairs_file
        round_config.paths.pairs.val = pairs_file  # Use same pairs for validation
        
        # Set round-specific learning rate
        if hasattr(self.config.training, 'lr_per_round'):
            lr_key = f"round_{round_num}"
            if hasattr(self.config.training.lr_per_round, lr_key):
                round_config.optimizer.lr = getattr(self.config.training.lr_per_round, lr_key)
                logger.info(f"Using learning rate: {round_config.optimizer.lr}")
                
        # Set output directory for this round
        round_config.paths.save_dir = str(self.round_dirs[round_num])
        
        # Train this round
        logger.info(f"Training round {round_num} for {self.config.training.epochs} epochs")
        
        # Use multiround trainer for actual training
        trainer_output = self.multiround_trainer.train_round(round_num)
        
        # Get candidate models for selection
        round_dir = Path(self.round_dirs[round_num])
        checkpoint_pattern = "checkpoint_*.pt"
        candidate_models = list(round_dir.glob(checkpoint_pattern))
        
        if not candidate_models:
            raise RuntimeError(f"No checkpoints found in {round_dir}")
            
        # Select best model using our criteria
        best_model_path = self.select_best_model(round_num, [str(p) for p in candidate_models])
        
        # Copy best model to standard location
        best_model_final = self.round_dirs[round_num] / "best_model.pt"
        shutil.copy2(best_model_path, best_model_final)
        
        logger.info(f"Round {round_num} completed. Best model: {best_model_final}")
        
        return str(best_model_final)
        
    def run_final_evaluation(self, final_model_path: str):
        """Run comprehensive final evaluation with pass@K analysis."""
        logger.info("Running final evaluation with comprehensive metrics")
        
        # Run evaluation with multiple sample sizes
        eval_configs = [
            {'n_samples': 200, 'temperature': 0.5, 'name': 'final_temp05'},
            {'n_samples': 200, 'temperature': 0.1, 'name': 'final_temp01'},
        ]
        
        final_results = {}
        
        for eval_config in eval_configs:
            logger.info(f"Running {eval_config['name']} evaluation")
            
            results = self.evaluator.evaluate_checkpoint(
                final_model_path,
                n_samples=eval_config['n_samples'],
                temperature=eval_config['temperature'],
                save_designs=True
            )
            
            # Calculate pass@K metrics for all K values
            pass_k_results = self.evaluator.calculate_pass_k_metrics(
                results,
                k_values=self.config.eval.pass_k.k_values,
                thresholds=self.config.eval.pass_k.full_thresholds
            )
            
            final_results[eval_config['name']] = {
                'basic_metrics': results,
                'pass_k_metrics': pass_k_results
            }
            
            # Log to wandb
            if self.config.wandb.enable:
                for metric_name, value in results.items():
                    if isinstance(value, (int, float)):
                        wandb.log({f"final/{eval_config['name']}/{metric_name}": value})
                        
                for k in self.config.eval.pass_k.k_values:
                    for threshold_type, threshold_values in self.config.eval.pass_k.full_thresholds.items():
                        for threshold in threshold_values:
                            metric_key = f"pass_{k}_{threshold_type}_{threshold}"
                            if metric_key in pass_k_results:
                                wandb.log({f"final/{eval_config['name']}/{metric_key}": pass_k_results[metric_key]})
                                
        # Save final results
        final_results_file = self.output_dir / "final_evaluation_results.json"
        with open(final_results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x))
            
        logger.info(f"Final evaluation completed. Results saved to: {final_results_file}")
        
        return final_results
        
    def run_training(self):
        """Run the complete multi-round training pipeline."""
        try:
            logger.info("Starting RiboPO v2 multi-round training")
            
            # Initialize all components
            self.initialize_components()
            
            # Train each round
            for round_num in range(1, self.config.multiround.num_rounds + 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"ROUND {round_num}/{self.config.multiround.num_rounds}")
                logger.info(f"{'='*60}")
                
                # Train this round
                best_model_path = self.train_round(round_num)
                
                # Update reference model for next round
                if round_num < self.config.multiround.num_rounds:
                    self.update_reference_model(round_num, best_model_path)
                    
                # Run intermediate evaluation if this is a full eval round
                if round_num in self.config.multiround.eval_rounds_full:
                    logger.info(f"Running full evaluation for round {round_num}")
                    eval_results = self.evaluator.evaluate_checkpoint(
                        best_model_path,
                        n_samples=64,
                        temperature=0.5,
                        save_designs=True
                    )
                    
                    # Log round results
                    if self.config.wandb.enable:
                        for metric_name, value in eval_results.items():
                            if isinstance(value, (int, float)):
                                wandb.log({f"round_{round_num}/eval/{metric_name}": value})
                                
                logger.info(f"Round {round_num} completed successfully")
                
            # Run final comprehensive evaluation
            final_model = self.best_models[self.config.multiround.num_rounds]['model_path']
            final_results = self.run_final_evaluation(final_model)
            
            # Save training summary
            self.save_training_summary(final_results)
            
            logger.info("Multi-round training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            if self.config.wandb.enable:
                wandb.finish()
                
    def save_training_summary(self, final_results: Dict):
        """Save a comprehensive training summary."""
        summary = {
            'experiment': {
                'name': self.config.experiment.name,
                'description': self.config.experiment.description,
                'version': self.config.experiment.version,
                'completion_time': datetime.now().isoformat()
            },
            'configuration': {
                'num_rounds': self.config.multiround.num_rounds,
                'epochs_per_round': self.config.training.epochs,
                'candidate_pool_size': self.config.ribopo_v2.candidate_pool_size,
                'winner_pool_size': self.config.ribopo_v2.winner_pool_size,
                'reward_formula': dict(self.config.ribopo_v2.reward_formula),
                'dynamic_pairs': getattr(self.config.multiround, 'dynamic_pairs', False)
            },
            'best_models_by_round': self.best_models,
            'training_history': self.training_history,
            'final_evaluation': final_results
        }
        
        summary_file = self.output_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else str(x))
            
        logger.info(f"Training summary saved to: {summary_file}")


def main():
    """Main entry point for RiboPO v2 multi-round training."""
    parser = argparse.ArgumentParser(description="RiboPO v2 Multi-Round Training")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from specific round checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate configuration without training")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RiboPOv2MultiRoundTrainer(args.config)
    
    if args.dry_run:
        logger.info("Dry run mode: Configuration validated successfully")
        return
        
    # Run training
    trainer.run_training()


if __name__ == "__main__":
    main()