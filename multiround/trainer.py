# multiround/trainer.py
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import json
import copy
import torch
import wandb
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from dpo.trainer import DPOTrainer
from dpo.ref_manager import save_checkpoint, load_checkpoint
from dpo.utils import AverageMeter, now_str
from dpo.losses import dpo_step_losses, ce_on_sequence
from torch.cuda.amp import autocast
from multiround.evaluator import MultiRoundEvaluator
from multiround.pair_provider import MultiRoundPairProvider
from multiround.wandb_manager import MultiRoundWandBManager
from multiround.utils import (
    plot_round_progression, save_round_metrics, 
    create_round_output_dir, update_reference_model
)


class MultiRoundDPOTrainer:
    """
    Multi-round DPO trainer implementing Plan A:
    - 5 rounds of training (configurable)
    - 20 epochs per round
    - Reference model updated each round by cloning policy
    - Comprehensive evaluation after each round
    - Rich logging and visualization
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        
        # Multiround settings
        self.num_rounds = getattr(cfg.multiround, 'num_rounds', 5)
        self.epochs_per_round = getattr(cfg.multiround, 'epochs_per_round', 20)
        self.current_round = getattr(cfg.multiround, 'current_round', 1)
        
        # Initialize dynamic pair provider
        self.pair_provider = MultiRoundPairProvider(cfg)
        
        # Initialize single-round trainer for each round
        self.trainer = None
        self.evaluator = MultiRoundEvaluator(cfg)
        
        # Tracking
        self.round_metrics = []  # List of metrics per round
        self.best_checkpoints = []  # Track best checkpoint per round
        self.round_start_time = None
        
        # Initialize enhanced wandb management first (for run name generation)
        self.wandb_manager = MultiRoundWandBManager(cfg)
        
        # Output organization using enhanced run name
        enhanced_run_name = self.wandb_manager.get_wandb_config()['name']
        self.output_root = os.path.join(
            getattr(cfg.multiround, 'output_root', 'runs/multiround'),
            enhanced_run_name
        )
        os.makedirs(self.output_root, exist_ok=True)
        if cfg.wandb.enable:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize wandb with enhanced organization and auto-generated configuration."""
        # Get enhanced wandb configuration from manager
        wandb_config = self.wandb_manager.get_wandb_config()
        hyperparams = self.wandb_manager.get_hyperparameters()
        
        # Add pair provider summary to hyperparams
        pair_summary = self.pair_provider.get_summary()
        hyperparams.update({
            'pair_provider_mode': 'dynamic' if pair_summary['dynamic_mode'] else 'static',
            'fallback_enabled': pair_summary['fallback_enabled']
        })
        
        # Set hyperparameters in wandb config
        wandb_config['config'] = hyperparams
        
        # Initialize wandb run with enhanced configuration
        wandb.init(**wandb_config)
        
        # Log experiment start and enhanced metadata
        self.wandb_manager.log_experiment_start(wandb)
        
        # Enhanced logging output
        print(f"🚀 Enhanced wandb run: {wandb.run.name} ({wandb.run.id})")
        print(f"📊 View at: {wandb.run.url}")
        print(f"🏷️ Tags: {', '.join(wandb_config['tags'])}")
        print(f"👥 Group: {wandb_config['group']}")
        
        # Log WandB manager summary
        manager_summary = self.wandb_manager.get_summary()
        print(f"🔧 Config fingerprint: {manager_summary['config_fingerprint']}")
        print(f"📁 Experiment family: {manager_summary['experiment_family']}")
    
    def train_all_rounds(self) -> Dict:
        """
        Execute all rounds of training.
        
        Returns:
            dict: Final summary of all rounds
        """
        print(f"\n🚀 Starting Multi-Round DPO Training")
        print(f"   Rounds: {self.num_rounds}")
        print(f"   Epochs per round: {self.epochs_per_round}")
        print(f"   Output: {self.output_root}")
        print(f"   Pair margin: {getattr(self.cfg.paths, 'pair_margin', 'unknown')}\n")
        
        try:
            for round_num in range(1, self.num_rounds + 1):
                print(f"\n{'='*60}")
                print(f"🔄 ROUND {round_num}/{self.num_rounds}")
                print(f"{'='*60}")
                
                self.current_round = round_num
                round_result = self.train_round(round_num)
                self.round_metrics.append(round_result)
                
                # Log round completion
                if self.cfg.wandb.enable:
                    wandb.log({
                        "round/completed": round_num,
                        "round/total_rounds": self.num_rounds,
                        **{f"round_{round_num}/{k}": v for k, v in round_result.items() 
                           if isinstance(v, (int, float))}
                    })
                
                # Early stopping check (optional)
                if self._should_stop_early(round_result):
                    print(f"🛑 Early stopping triggered after round {round_num}")
                    break
            
            # Final evaluation and summary
            final_summary = self._generate_final_summary()
            self._save_multiround_results(final_summary)
            
            return final_summary
            
        except KeyboardInterrupt:
            print(f"\n⚠️ Training interrupted. Saving current progress...")
            return self._generate_final_summary()
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            raise
        finally:
            if self.cfg.wandb.enable:
                wandb.finish()
    
    def train_round(self, round_num: int) -> Dict:
        """
        Execute a single round of training.
        
        Args:
            round_num: Current round number (1-indexed)
            
        Returns:
            dict: Round results and metrics
        """
        self.round_start_time = datetime.now()
        round_dir = create_round_output_dir(self.output_root, round_num)
        
        print(f"📁 Round {round_num} output: {round_dir}")
        
        # Step 1: Update reference model (clone policy → reference)
        if round_num > 1 and getattr(self.cfg.multiround, 'update_reference', True):
            print(f"🔄 Updating reference model from previous round's policy...")
            self._update_reference_model()
        
        # Step 2: Initialize/update trainer for this round
        self._setup_trainer_for_round(round_num, round_dir)
        
        # Log pair configuration to wandb
        if self.cfg.wandb.enable:
            current_margin = self.pair_provider.get_current_margin_type()
            pair_counts = self.pair_provider.current_config.get_pair_counts()
            wandb.log({
                f"round_{round_num}/margin_type": current_margin,
                f"round_{round_num}/train_pairs": pair_counts['train'],
                f"round_{round_num}/val_pairs": pair_counts['val'],
                f"round_{round_num}/test_pairs": pair_counts['test'],
                "round": round_num
            })
            print(f"📊 Logged margin type '{current_margin}' and pair counts to wandb")
        
        # Step 3: Train for specified epochs
        print(f"🏃 Training for {self.epochs_per_round} epochs...")
        train_start = datetime.now()
        
        # Execute training for this round
        # Instead of modifying config, we'll call a custom training loop
        training_result = self._train_for_round(self.epochs_per_round)
        
        train_time = datetime.now() - train_start
        print(f"✅ Training completed in {train_time}")
        
        # Step 4: Evaluate on test set
        print(f"📊 Evaluating round {round_num}...")
        eval_start = datetime.now()
        
        # Use appropriate sample count for evaluation
        is_final_round = (round_num == self.num_rounds)
        eval_result = self.evaluator.evaluate_round(
            model=self.trainer.policy,
            round_num=round_num,
            output_dir=round_dir,
            is_final_round=is_final_round
        )
        
        eval_time = datetime.now() - eval_start
        print(f"✅ Evaluation completed in {eval_time}")
        
        # Step 5: Save checkpoint
        checkpoint_path = self._save_round_checkpoint(round_num, round_dir, eval_result)
        
        # Step 6: Compile round results
        round_result = {
            'round': round_num,
            'timestamp': self.round_start_time.isoformat(),
            'training_time_sec': train_time.total_seconds(),
            'evaluation_time_sec': eval_time.total_seconds(),
            'checkpoint_path': checkpoint_path,
            **training_result,  # Include training metrics
            **eval_result,      # Include evaluation metrics
        }
        
        # Step 7: Save round results
        save_round_metrics(round_result, round_dir)
        
        # Step 8: Update progress visualization
        if round_num > 1:  # Need at least 2 rounds for progression plots
            plot_round_progression(self.round_metrics + [round_result], round_dir)
        
        print(f"✅ Round {round_num} completed successfully!")
        self._print_round_summary(round_result)
        
        return round_result
    
    def _setup_trainer_for_round(self, round_num: int, round_dir: str):
        """Setup trainer for the current round with dynamic pair loading."""
        # Update pair provider for this round
        new_pair_paths = self.pair_provider.update_config_for_round(round_num)
        
        # Safely update config paths for this round (temporary modification)
        original_paths = {
            'train': self.cfg.paths.pairs.train,
            'val': self.cfg.paths.pairs.val, 
            'test': self.cfg.paths.pairs.test
        }
        
        # Temporarily update the config with new pair paths
        self.cfg.paths.pairs.train = new_pair_paths['train']
        self.cfg.paths.pairs.val = new_pair_paths['val']
        self.cfg.paths.pairs.test = new_pair_paths['test']
        
        try:
            # Create new DPOTrainer instance with updated config
            self.trainer = DPOTrainer(self.cfg)
            
            # Log the pair switching
            current_margin = self.pair_provider.get_current_margin_type()
            print(f"✅ DPOTrainer initialized for round {round_num} with margin {current_margin} pairs")
            
        finally:
            # Always restore original paths to keep config clean
            self.cfg.paths.pairs.train = original_paths['train']
            self.cfg.paths.pairs.val = original_paths['val']
            self.cfg.paths.pairs.test = original_paths['test']
        
        # Override the save directory for this round (trainer doesn't modify config)
        self.trainer.save_root = round_dir
        os.makedirs(round_dir, exist_ok=True)
        os.makedirs(os.path.join(round_dir, "steps"), exist_ok=True)
        
        # Set the correct number of epochs for this round
        # Note: We'll handle this by controlling the training loop rather than modifying config
        
        # Load checkpoint from previous round if continuing
        if round_num > 1 and hasattr(self, 'best_checkpoints') and self.best_checkpoints:
            prev_checkpoint = self.best_checkpoints[-1]['path']
            if os.path.exists(prev_checkpoint):
                print(f"🔄 Loading checkpoint from round {round_num-1}: {prev_checkpoint}")
                load_checkpoint(prev_checkpoint, self.trainer.policy, self.trainer.optimizer)
                
        # Update reference model if needed (clone policy state to reference)
        if round_num > 1 and getattr(self.cfg.multiround, 'update_reference', False):
            print(f"🔄 Updating reference model from round {round_num-1} policy...")
            self.trainer.reference.load_state_dict(self.trainer.policy.state_dict())
            print("✅ Reference model updated")
    
    def _update_reference_model(self):
        """Update reference model by cloning current policy."""
        if self.trainer is not None:
            update_reference_model(self.trainer.policy, self.trainer.reference)
            print("✅ Reference model updated")
    
    def _train_for_round(self, num_epochs: int) -> Dict:
        """
        Train the policy model for a specific number of epochs for this round.
        
        Args:
            num_epochs: Number of epochs to train for this round
            
        Returns:
            dict: Training metrics from this round
        """
        cfg = self.cfg
        self.trainer.policy.train()
        
        # Initialize tracking for this round
        round_start_step = self.trainer.global_step
        round_train_losses = []
        round_val_losses = []
        final_train_loss = None
        final_val_loss = None
        
        for epoch in range(num_epochs):
            meters = defaultdict(AverageMeter)
            
            for batch in self.trainer.train_loader:
                self.trainer.global_step += 1
                
                # Move batch to device
                batch.graph = batch.graph.to(self.trainer.device)
                batch.winner_seq = batch.winner_seq.to(self.trainer.device)
                batch.loser_seq = batch.loser_seq.to(self.trainer.device)

                with autocast(enabled=cfg.training.precision in ["fp16", "bf16"], 
                             dtype=torch.bfloat16 if cfg.training.precision=="bf16" else torch.float16):
                    # DPO forward
                    out = dpo_step_losses(
                        model=self.trainer.policy,
                        ref_model=self.trainer.reference,
                        batch=batch,
                        beta=cfg.dpo.beta,
                        label_smoothing=cfg.dpo.label_smoothing,
                        max_len=cfg.dpo.max_len
                    )
                    loss = out["loss_dpo"]

                    # optional SFT on winners
                    if cfg.dpo.sft_lambda and cfg.dpo.sft_lambda > 0:
                        loss_sft = ce_on_sequence(self.trainer.policy, batch.graph, batch.winner_seq, max_len=cfg.dpo.max_len)
                        loss = loss + cfg.dpo.sft_lambda * loss_sft
                        out["loss_sft"] = loss_sft.detach()

                # backward (grad-accum)
                self.trainer.scaler.scale(loss / cfg.training.grad_accum_steps).backward()

                if self.trainer.global_step % cfg.training.grad_accum_steps == 0:
                    if cfg.optimizer.grad_clip_norm and cfg.optimizer.grad_clip_norm > 0:
                        self.trainer.scaler.unscale_(self.trainer.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.trainer.policy.parameters(), cfg.optimizer.grad_clip_norm)
                    self.trainer.scaler.step(self.trainer.optimizer)
                    self.trainer.scaler.update()
                    self.trainer.optimizer.zero_grad(set_to_none=True)
                    self.trainer.scheduler.step()

                # meters
                meters["train/loss"].update(loss.detach().item(), 1)
                for k in ["loss_dpo", "pref_acc", "margin"]:
                    meters[f"train/{k}"].update(out[k].item(), 1)
                meters["train/lr"].update(self.trainer.optimizer.param_groups[0]["lr"], 1)

                # log every step
                if (self.trainer.global_step % cfg.training.log_every) == 0 and wandb.run is not None:
                    log = {k: v.avg for k, v in meters.items()}
                    log["epoch"] = epoch
                    log["step"] = self.trainer.global_step
                    log["round"] = self.current_round
                    wandb.log(log, step=self.trainer.global_step)

                # periodic eval (but use round-specific frequency)
                if (self.trainer.global_step % cfg.training.val_every) == 0:
                    val_stats = self.trainer.evaluate(self.trainer.val_loader, split="val")
                    if wandb.run is not None:
                        wandb.log({f"val/{k}": v for k, v in val_stats.items()}, step=self.trainer.global_step)
                    
                    round_val_losses.append(val_stats.get("loss_dpo", 0.0))
                    final_val_loss = val_stats.get("loss_dpo", 0.0)

            # End of epoch - capture training loss
            round_train_losses.append(meters["train/loss"].avg)
            final_train_loss = meters["train/loss"].avg
        
        # Compile round training results
        total_steps_this_round = self.trainer.global_step - round_start_step
        training_result = {
            'total_steps': total_steps_this_round,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'mean_train_loss': np.mean(round_train_losses) if round_train_losses else 0.0,
            'train_loss_history': round_train_losses,
            'val_loss_history': round_val_losses,
        }
        
        print(f"📊 Round {self.current_round} training completed:")
        print(f"   Steps: {total_steps_this_round}")
        print(f"   Final train loss: {final_train_loss:.4f}")
        if final_val_loss is not None:
            print(f"   Final val loss: {final_val_loss:.4f}")
        
        return training_result
    
    def _save_round_checkpoint(self, round_num: int, round_dir: str, eval_result: Dict) -> str:
        """Save checkpoint for the current round."""
        checkpoint_name = f"round_{round_num}_best.pt"
        checkpoint_path = os.path.join(round_dir, checkpoint_name)
        
        # Determine if this is the best checkpoint
        metric_name = getattr(self.cfg.checkpoints, 'metric_for_best', 'tm_mean')
        current_metric = eval_result.get(metric_name, 0.0)
        
        # Save checkpoint using the correct signature
        save_checkpoint(
            round_dir,  # root directory
            f"round_{round_num}_best",  # name (without .pt extension)
            self.trainer.policy,
            self.trainer.optimizer,
            self.trainer.scheduler,
            self.trainer.global_step,
            current_metric,
            self.cfg
        )
        
        # Track as best checkpoint for this round
        checkpoint_info = {
            'round': round_num,
            'path': checkpoint_path,
            'metric_value': current_metric,
            'eval_metrics': eval_result
        }
        self.best_checkpoints.append(checkpoint_info)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def _should_stop_early(self, round_result: Dict) -> bool:
        """Check if training should stop early (placeholder for future implementation)."""
        # Could implement early stopping based on metric convergence
        return False
    
    def _print_round_summary(self, round_result: Dict):
        """Print a summary of the round results."""
        print(f"\n📈 Round {round_result['round']} Summary:")
        
        # Training metrics
        if 'final_train_loss' in round_result:
            print(f"   Training Loss: {round_result['final_train_loss']:.4f}")
        if 'final_val_loss' in round_result:
            print(f"   Validation Loss: {round_result['final_val_loss']:.4f}")
        
        # Primary evaluation metrics
        primary_metrics = ['tm_mean', 'rmsd_mean', 'mfe_mean']
        for metric in primary_metrics:
            if metric in round_result:
                print(f"   {metric.upper()}: {round_result[metric]:.4f}")
        
        # Secondary metrics
        secondary_metrics = ['plddt_mean', 'gdt_mean', 'diversity_3mer_mean']
        for metric in secondary_metrics:
            if metric in round_result:
                print(f"   {metric.upper()}: {round_result[metric]:.4f}")
        
        print()
    
    def _generate_final_summary(self) -> Dict:
        """Generate final summary of all rounds."""
        if not self.round_metrics:
            return {'error': 'No rounds completed'}
        
        summary = {
            'total_rounds_completed': len(self.round_metrics),
            'total_training_time_sec': sum(r.get('training_time_sec', 0) for r in self.round_metrics),
            'total_evaluation_time_sec': sum(r.get('evaluation_time_sec', 0) for r in self.round_metrics),
            'best_round_by_tm': None,
            'final_round_metrics': self.round_metrics[-1] if self.round_metrics else {},
            'round_progression': self.round_metrics
        }
        
        # Find best round by TM score
        best_tm = -1
        best_round = None
        for r in self.round_metrics:
            tm_score = r.get('tm_mean', 0)
            if tm_score > best_tm:
                best_tm = tm_score
                best_round = r
        
        if best_round:
            summary['best_round_by_tm'] = {
                'round': best_round['round'],
                'tm_mean': best_round.get('tm_mean', 0),
                'rmsd_mean': best_round.get('rmsd_mean', 0),
                'mfe_mean': best_round.get('mfe_mean', 0)
            }
        
        return summary
    
    def _save_multiround_results(self, summary: Dict):
        """Save final multiround results."""
        results_path = os.path.join(self.output_root, 'final_summary.json')
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Final results saved: {results_path}")
        
        # Log to wandb
        if self.cfg.wandb.enable:
            wandb.log({
                "final/total_rounds": summary['total_rounds_completed'],
                "final/best_tm_score": summary.get('best_round_by_tm', {}).get('tm_mean', 0),
                "final/total_training_hours": summary['total_training_time_sec'] / 3600,
            })