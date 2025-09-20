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
            # Support resuming from a specific round
            start_round = self.current_round
            if start_round > 1:
                print(f"🔄 Resuming training from round {start_round}")
                # Load the best checkpoint from the previous round
                self._load_checkpoint_for_resume(start_round)
            
            for round_num in range(start_round, self.num_rounds + 1):
                print(f"\n{'='*60}")
                print(f"🔄 ROUND {round_num}/{self.num_rounds}")
                print(f"{'='*60}")
                
                self.current_round = round_num
                round_result = self.train_round(round_num)
                self.round_metrics.append(round_result)
                
                # Log round completion with monotonic step tracking
                if self.cfg.wandb.enable:
                    # Use current step offset to maintain monotonicity, ensure minimum step 1
                    current_step = max(1, getattr(self, 'wandb_step_offset', 0) + getattr(self.trainer, 'global_step', 0))
                    wandb.log({
                        "round/completed": round_num,
                        "round/total_rounds": self.num_rounds,
                        **{f"round_{round_num}/{k}": v for k, v in round_result.items() 
                           if isinstance(v, (int, float))}
                    }, step=current_step)
                
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
        
        # Step 1: Initialize/update trainer for this round
        self._setup_trainer_for_round(round_num, round_dir)
        
        # Step 2: Update reference model (clone policy → reference) AFTER trainer setup
        if round_num > 1 and getattr(self.cfg.multiround, 'update_reference', False):
            print(f"🔄 Updating reference model from previous round's policy...")
            self._update_reference_model()
        
        # Log pair configuration to wandb with monotonic step tracking
        if self.cfg.wandb.enable:
            current_margin = self.pair_provider.get_current_margin_type()
            pair_counts = self.pair_provider.current_config.get_pair_counts()
            # Use current step offset to maintain monotonicity, ensure minimum step 1
            current_step = max(1, getattr(self, 'wandb_step_offset', 0) + getattr(self.trainer, 'global_step', 0))
            wandb.log({
                f"round_{round_num}/margin_type": current_margin,
                f"round_{round_num}/train_pairs": pair_counts['train'],
                f"round_{round_num}/val_pairs": pair_counts['val'],
                f"round_{round_num}/test_pairs": pair_counts['test'],
                "progress/round": round_num,
                "round": round_num  # Keep for backward compatibility
            }, step=current_step)
            
            # Update run summary for dashboard visibility
            if wandb.run:
                wandb.run.summary.update({
                    "current_round": round_num,
                    "round_start_time": datetime.now().isoformat()
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
        
        # Step 5: Save checkpoint and select best
        checkpoint_path = self._save_round_checkpoint(round_num, round_dir, eval_result)
        
        # Step 5b: Select best checkpoint for reference model update
        best_checkpoint_info = self.select_best_checkpoint_from_round(round_num, round_dir)
        if best_checkpoint_info and best_checkpoint_info.get('is_best', False):
            # Update the tracking with the new best checkpoint
            if self.best_checkpoints and self.best_checkpoints[-1]['round'] != round_num:
                self.best_checkpoints[-1] = best_checkpoint_info
            
            # Save selection rationale
            selection_path = os.path.join(round_dir, "evaluation", f"checkpoint_selection_round_{round_num}.json")
            with open(selection_path, 'w') as f:
                json.dump(best_checkpoint_info, f, indent=2)
            print(f"💾 Checkpoint selection rationale saved: {selection_path}")
        
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
            
            # Initialize step offset for proper W&B step tracking across rounds
            if not hasattr(self, 'wandb_step_offset'):
                self.wandb_step_offset = 0
            if round_num > 1:
                # Calculate offset based on previous rounds to maintain monotonic stepping
                self.wandb_step_offset = getattr(self, 'total_steps_completed', 0)
                print(f"🔄 W&B step offset for round {round_num}: {self.wandb_step_offset}")
            
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
        
        # Load resume checkpoint if this is a resume scenario
        self._load_resume_checkpoint_if_needed()
        
        # Load checkpoint from previous round if continuing (non-resume scenario)
        if (round_num > 1 and hasattr(self, 'best_checkpoints') and self.best_checkpoints 
            and not hasattr(self, '_resume_from_round')):
            prev_checkpoint = self.best_checkpoints[-1]['path']
            if os.path.exists(prev_checkpoint):
                print(f"🔄 Loading checkpoint from round {round_num-1}: {prev_checkpoint}")
                load_checkpoint(prev_checkpoint, self.trainer.policy, self.trainer.optimizer)
                
        # Reference model update is now handled in train_round() after trainer setup
        # This ensures consistent timing and logic
    
    def _update_reference_model(self):
        """Update reference model by cloning current policy."""
        if self.trainer is None:
            raise RuntimeError("Cannot update reference model: trainer not initialized")
        
        # Use consistent state_dict copying method
        self.trainer.reference.load_state_dict(self.trainer.policy.state_dict())
        print("✅ Reference model updated from current policy")
    
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
        
        # Calculate steps per epoch based on dataset size and gradient accumulation
        train_dataset_size = len(self.trainer.train_loader.dataset)
        batches_per_epoch = (train_dataset_size + cfg.training.batch_size - 1) // cfg.training.batch_size
        # With gradient accumulation, we process grad_accum_steps batches per optimizer step
        steps_per_epoch = batches_per_epoch  # Each batch is one forward pass (one step)
        print(f"📊 Training: {train_dataset_size} samples, {batches_per_epoch} batches/steps per epoch (grad_accum every {cfg.training.grad_accum_steps} steps)")
        
        # Create iterator for the dataloader
        train_iter = iter(self.trainer.train_loader)
        
        for epoch in range(num_epochs):
            meters = defaultdict(AverageMeter)
            print(f"📖 Starting epoch {epoch}/{num_epochs-1} (step {self.trainer.global_step})")
            
            # Process exactly steps_per_epoch batches for this epoch
            for step_in_epoch in range(steps_per_epoch):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    # Reset iterator if we've exhausted the dataloader
                    train_iter = iter(self.trainer.train_loader)
                    batch = next(train_iter)
                self.trainer.global_step += 1
                
                # Move batch to device
                batch.graph = batch.graph.to(self.trainer.device)
                batch.winner_seq = batch.winner_seq.to(self.trainer.device)
                batch.loser_seq = batch.loser_seq.to(self.trainer.device)

                # Safe precision access with bf16 default, fallback to fp32 if unsupported
                precision = getattr(cfg.training, 'precision', 'bf16')
                # Check if bf16 is supported by the current GPU
                if precision == 'bf16' and not torch.cuda.is_bf16_supported():
                    print("⚠️ BF16 not supported on this GPU, falling back to FP32")
                    precision = 'fp32'
                with autocast(enabled=precision in ["fp16", "bf16"], 
                             dtype=torch.bfloat16 if precision=="bf16" else torch.float16):
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
                    # Add clear progress tracking (0-based epoch numbering)
                    log["progress/epoch"] = epoch
                    log["progress/round"] = self.current_round
                    log["progress/step"] = self.trainer.global_step
                    log["progress/epoch_in_round"] = epoch  # Clear indicator of epoch within current round
                    log["progress/total_epochs_completed"] = (self.current_round - 1) * self.epochs_per_round + epoch
                    log["progress/step_in_epoch"] = step_in_epoch
                    log["progress/steps_per_epoch"] = steps_per_epoch
                    wandb_step = max(1, self.wandb_step_offset + self.trainer.global_step)
                    wandb.log(log, step=wandb_step)
                    
                    # Update run summary with current progress (visible in dashboard)
                    if wandb.run:
                        wandb.run.summary.update({
                            "current_round": self.current_round,
                            "current_epoch": epoch,
                            "current_step": self.trainer.global_step,
                            "total_epochs_completed": (self.current_round - 1) * self.epochs_per_round + epoch,
                            "steps_per_epoch": steps_per_epoch
                        })

                # periodic eval (but use round-specific frequency)
                if (self.trainer.global_step % cfg.training.val_every) == 0:
                    val_stats = self.trainer.evaluate(self.trainer.val_loader, split="val")
                    if wandb.run is not None:
                        wandb_step = max(1, self.wandb_step_offset + self.trainer.global_step)
                        wandb.log({f"val/{k}": v for k, v in val_stats.items()}, step=wandb_step)
                    
                    round_val_losses.append(val_stats.get("loss_dpo", 0.0))
                    final_val_loss = val_stats.get("loss_dpo", 0.0)
                
                # Periodic checkpoint saving every save_every steps
                if (self.trainer.global_step % cfg.training.save_every) == 0:
                    self._save_step_checkpoint(epoch, self.trainer.global_step, self.current_round)

            # End of epoch - flush any remaining accumulated gradients
            if self.trainer.global_step % cfg.training.grad_accum_steps != 0:
                print(f"🔄 Flushing accumulated gradients at end of epoch {epoch}")
                if cfg.optimizer.grad_clip_norm and cfg.optimizer.grad_clip_norm > 0:
                    self.trainer.scaler.unscale_(self.trainer.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainer.policy.parameters(), cfg.optimizer.grad_clip_norm)
                self.trainer.scaler.step(self.trainer.optimizer)
                self.trainer.scaler.update()
                self.trainer.optimizer.zero_grad(set_to_none=True)
                self.trainer.scheduler.step()
            
            # End of epoch - capture training loss
            round_train_losses.append(meters["train/loss"].avg)
            final_train_loss = meters["train/loss"].avg
            print(f"✅ Completed epoch {epoch}/{num_epochs-1} (step {self.trainer.global_step}, loss: {final_train_loss:.4f})")
            
            # Log epoch completion to WandB
            if self.cfg.wandb.enable and wandb.run is not None:
                wandb_step = max(1, self.wandb_step_offset + self.trainer.global_step)
                wandb.log({
                    "epoch_completion/epoch": epoch,
                    "epoch_completion/round": self.current_round,
                    "epoch_completion/avg_loss": final_train_loss,
                    "epoch_completion/steps_completed": self.trainer.global_step
                }, step=wandb_step)
        
        # Compile round training results
        total_steps_this_round = self.trainer.global_step - round_start_step
        
        # Update total completed steps for W&B step offset tracking
        if not hasattr(self, 'total_steps_completed'):
            self.total_steps_completed = 0
        self.total_steps_completed += total_steps_this_round
        
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
    
    def _save_step_checkpoint(self, epoch: int, step: int, round_num: int):
        """Save periodic checkpoint during training."""
        round_dir = os.path.join(self.output_root, f"round_{round_num:02d}")
        checkpoints_dir = os.path.join(round_dir, "checkpoints")
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        checkpoint_name = f"step_{step}_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
        
        # Save step checkpoint
        save_checkpoint(
            checkpoints_dir,  # checkpoints subdirectory
            f"step_{step}_epoch_{epoch}",  # name (without .pt extension)
            self.trainer.policy,
            self.trainer.optimizer,
            self.trainer.scheduler,
            self.trainer.global_step,
            0.0,  # no eval metric for step checkpoints
            self.cfg
        )
        
        print(f"💾 Step checkpoint saved: {checkpoint_path}")
    
    def _save_round_checkpoint(self, round_num: int, round_dir: str, eval_result: Dict) -> str:
        """Save checkpoint for the current round."""
        checkpoint_name = f"round_{round_num}_best.pt"
        # Use the checkpoints subdirectory created by create_round_output_dir()
        checkpoint_path = os.path.join(round_dir, "checkpoints", checkpoint_name)
        
        # Determine if this is the best checkpoint
        # Handle config structure safely - checkpoints may be a SimpleNamespace or dict
        checkpoints_cfg = getattr(self.cfg, 'checkpoints', None)
        if checkpoints_cfg is not None:
            metric_name = getattr(checkpoints_cfg, 'metric_for_best', 'tm_mean')
        else:
            metric_name = 'tm_mean'  # fallback default
        current_metric = eval_result.get(metric_name, 0.0)
        
        # Save checkpoint using the correct signature and checkpoints subdirectory
        save_checkpoint(
            os.path.join(round_dir, "checkpoints"),  # checkpoints subdirectory
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
    
    def select_best_checkpoint_from_round(self, round_num: int, round_dir: str) -> Dict:
        """
        Select the best checkpoint from the current round using pass@k metrics.
        
        Args:
            round_num: Current round number
            round_dir: Directory containing round data
            
        Returns:
            dict: Information about the selected best checkpoint
        """
        # Load evaluation results for candidate selection
        eval_results_path = os.path.join(round_dir, "evaluation", f"eval_results_round_{round_num}.json")
        if not os.path.exists(eval_results_path):
            print(f"⚠️ No evaluation results found at {eval_results_path}")
            if self.best_checkpoints:
                return self.best_checkpoints[-1]
            return None
        
        try:
            with open(eval_results_path, 'r') as f:
                eval_data = json.load(f)
            
            # Primary metric: pass@8 with TM-score >= 0.45
            primary_metric_key = "passk_tm_0.45_k8"
            primary_score = eval_data.get(primary_metric_key, 0.0)
            
            # Tie-breaker: MFE (lower is better)
            mfe_score = eval_data.get('mfe_mean', 0.0)
            
            # Create candidate info
            candidate_info = {
                'round': round_num,
                'path': os.path.join(round_dir, "checkpoints", f"round_{round_num}_best.pt"),
                'primary_metric': primary_score,
                'tiebreaker_metric': mfe_score,
                'selection_criteria': {
                    'primary': f"{primary_metric_key} = {primary_score:.4f}",
                    'tiebreaker': f"mfe_mean = {mfe_score:.4f}"
                },
                'eval_metrics': eval_data
            }
            
            # Log selection rationale
            print(f"🎯 Checkpoint selection for round {round_num}:")
            print(f"   Primary metric ({primary_metric_key}): {primary_score:.4f}")
            print(f"   Tie-breaker (MFE): {mfe_score:.4f}")
            
            # Compare with previous best if available
            if self.best_checkpoints and round_num > 1:
                prev_best = self.best_checkpoints[-1]
                prev_primary = prev_best.get('primary_metric', 0.0)
                prev_mfe = prev_best.get('tiebreaker_metric', 0.0)
                
                # Selection logic: higher pass@k is better, lower MFE is better (for ties)
                is_better = False
                if primary_score > prev_primary:
                    is_better = True
                    reason = f"Higher pass@k ({primary_score:.4f} > {prev_primary:.4f})"
                elif abs(primary_score - prev_primary) < 1e-6:  # Essentially equal
                    if mfe_score < prev_mfe:  # Lower MFE is better
                        is_better = True
                        reason = f"Equal pass@k, better MFE ({mfe_score:.4f} < {prev_mfe:.4f})"
                    else:
                        reason = f"Equal pass@k, worse MFE ({mfe_score:.4f} >= {prev_mfe:.4f})"
                else:
                    reason = f"Lower pass@k ({primary_score:.4f} < {prev_primary:.4f})"
                
                print(f"   Comparison with round {prev_best['round']}: {reason}")
                
                if is_better:
                    print(f"✅ Round {round_num} checkpoint selected as new best")
                    candidate_info['selection_reason'] = reason
                    candidate_info['is_best'] = True
                else:
                    print(f"🔄 Keeping round {prev_best['round']} checkpoint as best")
                    candidate_info['selection_reason'] = reason
                    candidate_info['is_best'] = False
                    return prev_best
            else:
                candidate_info['is_best'] = True
                candidate_info['selection_reason'] = "First round or no previous checkpoint"
            
            return candidate_info
            
        except Exception as e:
            print(f"❌ Error during checkpoint selection: {e}")
            import traceback
            traceback.print_exc()
            if self.best_checkpoints:
                return self.best_checkpoints[-1]
            return None
    
    def _load_checkpoint_for_resume(self, start_round: int):
        """Load checkpoint from previous round when resuming training."""
        try:
            # Find checkpoint from the previous round (in checkpoints subdirectory)
            prev_round = start_round - 1
            round_dir = os.path.join(self.output_root, f"round_{prev_round:02d}")
            checkpoint_path = os.path.join(round_dir, "checkpoints", f"round_{prev_round}_best.pt")
            
            if os.path.exists(checkpoint_path):
                print(f"📂 Found checkpoint from round {prev_round}: {checkpoint_path}")
                
                # Store checkpoint info for later loading (avoid double trainer creation)
                self._resume_checkpoint_path = checkpoint_path
                self._resume_from_round = prev_round
                
                # Load checkpoint metadata to get step count for W&B offset
                try:
                    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                    if 'step' in checkpoint_data:
                        # Seed W&B step offset from loaded checkpoint
                        self.wandb_step_offset = checkpoint_data['step']
                        print(f"🔄 Resume: W&B step offset set to {self.wandb_step_offset}")
                except Exception as e:
                    print(f"⚠️ Could not load checkpoint metadata for step offset: {e}")
                    self.wandb_step_offset = 0
                
                print(f"✅ Resume setup complete - will load checkpoint after trainer initialization")
                return
            else:
                print(f"❌ No checkpoint found for resume: {checkpoint_path}")
                
        except Exception as e:
            print(f"❌ Resume setup failed: {e}")
            
        # If we get here, resume failed - continue with fresh training
        self._resume_checkpoint_path = None
        self._resume_from_round = None
    
    def _load_resume_checkpoint_if_needed(self):
        """Load resume checkpoint after trainer is properly initialized."""
        if not hasattr(self, '_resume_checkpoint_path') or self._resume_checkpoint_path is None:
            return
            
        try:
            print(f"📂 Loading resume checkpoint: {self._resume_checkpoint_path}")
            step, best_metric = load_checkpoint(
                self._resume_checkpoint_path,
                self.trainer.policy,
                self.trainer.optimizer
            )
            
            print(f"✅ Resume checkpoint loaded: step {step}, metric: {best_metric}")
            
            # Clear resume state
            self._resume_checkpoint_path = None
            self._resume_from_round = None
            
        except Exception as e:
            print(f"❌ Resume checkpoint loading failed: {e}")
            # Clear resume state on failure
            self._resume_checkpoint_path = None
            self._resume_from_round = None
    
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
        
        # Log to wandb with monotonic step tracking
        if self.cfg.wandb.enable:
            # Use final step for summary metrics, ensure minimum step 1
            final_step = max(1, getattr(self, 'wandb_step_offset', 0) + getattr(self.trainer, 'global_step', 0))
            wandb.log({
                "final/total_rounds": summary['total_rounds_completed'],
                "final/best_tm_score": summary.get('best_round_by_tm', {}).get('tm_mean', 0),
                "final/total_training_hours": summary['total_training_time_sec'] / 3600,
            }, step=final_step)