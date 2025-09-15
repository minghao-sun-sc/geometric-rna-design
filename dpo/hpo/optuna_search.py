from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import copy
import yaml
import json
import torch
import wandb
import optuna
import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as SN
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Dict, Any, Optional

from dpo.trainer import DPOTrainer, SimPOTrainer


def _to_sn(o):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(o, dict):
        return SN(**{k: _to_sn(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_to_sn(v) for v in o]
    return o


def load_cfg(path: str) -> SN:
    """Load YAML config and convert to SimpleNamespace."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_sn(raw)


class HPOObjective:
    """Objective function for hyperparameter optimization."""
    
    def __init__(self, config_path: str, output_dir: str, algo: str = "dpo"):
        self.base_cfg = load_cfg(config_path)
        self.algo = algo
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track metrics across trials for normalization
        self.metric_history = {
            "val/pref_acc": [],
            "val/margin": [],
            "val/perplexity": []
        }
    
    def compute_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute HPO score from validation metrics.
        
        For DPO:
        score = pref_acc + α * clip(margin, 0, m_cap) - γ * normalized_ppl
        
        For SimPO:
        score = reward_acc + α * clip(z_margin, 0, m_cap) - γ * normalized_ppl
        """
        if self.algo == "simpo":
            acc = float(metrics.get("val/simpo/reward_acc", 0.0))
            margin = float(metrics.get("val/simpo/z_margin", 0.0))
        else:  # DPO
            acc = float(metrics.get("val/pref_acc", 0.0))
            margin = float(metrics.get("val/margin", 0.0))
        
        # Perplexity (if available from SFT component)
        ppl = float(metrics.get("val/perplexity", 1.5))
        
        # Dynamic normalization based on observed values
        if len(self.metric_history["val/perplexity"]) > 2:
            ppl_mean = sum(self.metric_history["val/perplexity"]) / len(self.metric_history["val/perplexity"])
            ppl_std = max(0.1, torch.std(torch.tensor(self.metric_history["val/perplexity"])).item())
            z_ppl = (ppl - ppl_mean) / ppl_std
        else:
            # Initial estimate
            z_ppl = (ppl - 1.5) / 0.2
        
        # Compute score with clipped margin
        alpha = 0.05
        m_cap = 2.0
        gamma = 0.1
        
        score = acc + alpha * min(max(margin, 0.0), m_cap) - gamma * z_ppl
        
        return score
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Execute one HPO trial."""
        
        # Deep copy base config
        cfg = copy.deepcopy(self.base_cfg)
        
        # Get search space from config if available
        search_space = getattr(cfg.hpo, 'search_space', None) if hasattr(cfg, 'hpo') else None
        
        # Suggest hyperparameters based on algorithm
        if self.algo == "dpo":
            # DPO hyperparameters
            if search_space:
                beta = trial.suggest_float("beta", 
                                          getattr(search_space, 'beta_min', 0.05),
                                          getattr(search_space, 'beta_max', 0.5), 
                                          log=True)
                sft_lambda = trial.suggest_float("sft_lambda", 
                                                getattr(search_space, 'sft_lambda_min', 0.0),
                                                getattr(search_space, 'sft_lambda_max', 0.25))
                lr = trial.suggest_float("lr", 
                                        getattr(search_space, 'lr_min', 5e-6),
                                        getattr(search_space, 'lr_max', 3e-4), 
                                        log=True)
            else:
                # Default ranges
                beta = trial.suggest_float("beta", 0.05, 0.5, log=True)
                sft_lambda = trial.suggest_float("sft_lambda", 0.0, 0.25)
                lr = trial.suggest_float("lr", 5e-6, 3e-4, log=True)
            
            # Set in config
            cfg.dpo.beta = beta
            cfg.dpo.sft_lambda = sft_lambda
            cfg.loss_type = "dpo"
            
        elif self.algo == "simpo":
            # SimPO hyperparameters
            if search_space:
                beta = trial.suggest_float("beta", 
                                          getattr(search_space, 'beta_min', 0.5),
                                          getattr(search_space, 'beta_max', 3.0), 
                                          log=True)
                gamma = trial.suggest_float("gamma", 
                                           getattr(search_space, 'gamma_min', 0.0),
                                           getattr(search_space, 'gamma_max', 1.5))
                sft_lambda = trial.suggest_float("sft_lambda", 
                                                getattr(search_space, 'sft_lambda_min', 0.0),
                                                getattr(search_space, 'sft_lambda_max', 0.20))
                lr = trial.suggest_float("lr", 
                                        getattr(search_space, 'lr_min', 1e-6),
                                        getattr(search_space, 'lr_max', 2e-4), 
                                        log=True)
            else:
                # Default ranges
                beta = trial.suggest_float("beta", 0.5, 3.0, log=True)
                gamma = trial.suggest_float("gamma", 0.0, 1.5)  # margin parameter
                sft_lambda = trial.suggest_float("sft_lambda", 0.0, 0.20)
                lr = trial.suggest_float("lr", 1e-6, 2e-4, log=True)
            
            # Set in config
            if not hasattr(cfg, "simpo"):
                cfg.simpo = SN()
            cfg.simpo.beta = beta
            cfg.simpo.gamma = gamma
            cfg.simpo.sft_lambda = sft_lambda
            cfg.loss_type = "simpo"
            
        else:
            raise ValueError(f"Unknown algorithm: {self.algo}")
        
        # Common hyperparameters
        cfg.optimizer.lr = lr
        
        # Optional: suggest other hyperparameters
        if trial.suggest_categorical("warmup", [True, False]):
            cfg.scheduler.warmup_steps = trial.suggest_int("warmup_steps", 100, 1000)
        else:
            cfg.scheduler.warmup_steps = 0
        
        # HPO-specific settings
        cfg.seed = cfg.seed if hasattr(cfg, "seed") else 42
        cfg.wandb.enable = True  # Keep wandb for detailed tracking
        cfg.wandb.project = "ribopo_hpo"
        cfg.wandb.run_name = f"{self.algo}_trial_{trial.number:03d}"
        cfg.wandb.tags = [self.algo, "hpo", f"trial_{trial.number}"]
        
        # Quick training for HPO
        cfg.training.epochs = 2  # Short epochs for quick eval
        cfg.training.val_every = 50  # Frequent validation
        cfg.training.log_every = 10  # Moderate logging frequency
        cfg.training.save_every = 10000  # Don't save checkpoints frequently
        
        # Add max_steps if supported (we'll add this to trainers)
        cfg.training.max_steps = cfg.hpo.max_steps if hasattr(cfg, "hpo") else 4000
        
        # Create trainer
        if self.algo == "simpo":
            trainer = SimPOTrainer(cfg)
        else:
            trainer = DPOTrainer(cfg)
        
        # Train and get metrics
        trainer.train()
        
        # Get best validation metrics
        best_metrics = getattr(trainer, "best_metrics", {})
        
        # If no best_metrics, use latest validation
        if not best_metrics:
            best_metrics = trainer.evaluate(trainer.val_loader, split="val")
        
        # Update metric history
        for key in ["val/pref_acc", "val/margin", "val/perplexity"]:
            if key in best_metrics:
                self.metric_history[key].append(best_metrics[key])
        
        # Compute score
        score = self.compute_score(best_metrics)
        
        # Save trial results
        trial_result = {
            "trial_number": trial.number,
            "algorithm": self.algo,
            "hyperparameters": trial.params,
            "score": score,
            "metrics": best_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to JSON
        result_file = self.output_dir / f"trial_{trial.number:03d}.json"
        with open(result_file, "w") as f:
            json.dump(trial_result, f, indent=2)
        
        # Report to optuna for pruning (though we're not using pruning)
        trial.report(score, step=cfg.training.max_steps)
        
        # Clean up GPU memory
        del trainer
        torch.cuda.empty_cache()
        
        return score


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for DPO/SimPO")
    parser.add_argument("--config", type=str, required=True, help="Path to base config YAML")
    parser.add_argument("--algo", type=str, default="dpo", choices=["dpo", "simpo"],
                       help="Algorithm to optimize")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, 
                       help="Optuna storage URL (e.g., sqlite:///study.db)")
    parser.add_argument("--output_dir", type=str, default="dpo/hpo/optuna_results",
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_jobs", type=int, default=1, 
                       help="Number of parallel jobs (use 1 for GPU)")
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.algo}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up study name
    if args.study_name is None:
        args.study_name = f"{args.algo}_hpo_{timestamp}"
    
    # Create sampler and pruner
    sampler = TPESampler(
        seed=args.seed,
        multivariate=True,
        group=True,
        n_startup_trials=5  # Initial random trials
    )
    
    # No pruning as requested
    pruner = None
    
    # Storage setup
    if args.storage is None:
        args.storage = f"sqlite:///{output_dir}/study.db"
    
    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    # Create objective
    objective = HPOObjective(
        config_path=args.config,
        output_dir=output_dir,
        algo=args.algo
    )
    
    # Run optimization
    print(f"🚀 Starting HPO for {args.algo.upper()}")
    print(f"📊 Study: {args.study_name}")
    print(f"💾 Results: {output_dir}")
    print(f"🎯 Trials: {args.trials}")
    
    study.optimize(
        objective,
        n_trials=args.trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
    # Print top trials
    print("\n" + "="*60)
    print("🏆 Top 5 Trials")
    print("="*60)
    
    top_trials = sorted(
        study.trials,
        key=lambda x: x.value if x.value is not None else -float('inf'),
        reverse=True
    )[:5]
    
    for i, trial in enumerate(top_trials, 1):
        print(f"\n#{i} Trial {trial.number} - Score: {trial.value:.4f}")
        print(f"   Parameters:")
        for param, value in trial.params.items():
            if isinstance(value, float):
                print(f"     {param}: {value:.6f}")
            else:
                print(f"     {param}: {value}")
    
    # Save final summary
    summary = {
        "study_name": args.study_name,
        "algorithm": args.algo,
        "total_trials": len(study.trials),
        "best_trial": {
            "number": study.best_trial.number,
            "score": study.best_value,
            "params": study.best_params
        },
        "top_5_trials": [
            {
                "number": t.number,
                "score": t.value,
                "params": t.params
            }
            for t in top_trials
        ]
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ HPO Complete! Summary saved to {summary_file}")
    
    # Generate config for best trial
    best_config = copy.deepcopy(load_cfg(args.config))
    best_params = study.best_params
    
    if args.algo == "dpo":
        best_config.dpo.beta = best_params["beta"]
        best_config.dpo.sft_lambda = best_params["sft_lambda"]
    else:  # simpo
        if not hasattr(best_config, "simpo"):
            best_config.simpo = {}
        best_config.simpo["beta"] = best_params["beta"]
        best_config.simpo["gamma"] = best_params["gamma"]
        best_config.simpo["sft_lambda"] = best_params["sft_lambda"]
    
    best_config.optimizer["lr"] = best_params["lr"]
    
    # Save best config
    best_config_file = output_dir / "best_config.yaml"
    with open(best_config_file, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"📝 Best config saved to {best_config_file}")


if __name__ == "__main__":
    main()