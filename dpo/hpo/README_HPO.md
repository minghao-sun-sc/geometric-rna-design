# RNA Preference Optimization - Hyperparameter Search

## Overview
This HPO system uses Optuna to find optimal hyperparameters for DPO and SimPO training on RNA structure prediction tasks.

## Key Features
- **Parallel Search**: 6 different search regions (3 SimPO, 3 DPO)
- **Smart Scoring**: Combines preference accuracy, margin, and perplexity
- **Quick Trials**: 5k steps per trial for rapid exploration
- **Full Logging**: WandB integration + local JSON results

## Search Spaces

### SimPO (Configs 1-3, A100 GPUs)
| Region | Beta Range | Gamma Range | SFT λ Range | LR Range |
|--------|------------|-------------|-------------|----------|
| 1 (Conservative) | 0.5-1.5 | 0.0-0.8 | 0.0-0.15 | 1e-6 to 5e-5 |
| 2 (Balanced) | 1.0-2.5 | 0.3-1.0 | 0.02-0.20 | 2e-6 to 1e-4 |
| 3 (Aggressive) | 2.0-4.0 | 0.5-1.5 | 0.0-0.12 | 5e-6 to 1.5e-4 |

### DPO (Configs 4-6, A40 GPUs)
| Region | Beta Range | SFT λ Range | LR Range |
|--------|------------|-------------|----------|
| 4 (Conservative) | 0.05-0.15 | 0.0-0.15 | 5e-6 to 1e-4 |
| 5 (Balanced) | 0.1-0.35 | 0.05-0.25 | 1e-5 to 2e-4 |
| 6 (Aggressive) | 0.25-0.6 | 0.0-0.20 | 2e-5 to 3e-4 |

## Usage

### Quick Test (Single Config)
```bash
# Test SimPO config 1 with 3 trials
bash dpo/scripts/run_hpo_single.sh 1

# Test DPO config 4 with 3 trials  
bash dpo/scripts/run_hpo_single.sh 4
```

### Full HPO (All Configs via SLURM)
```bash
# Submit all 6 parallel jobs (20 trials each)
bash dpo/scripts/submit_all_hpo.sh

# Monitor jobs
squeue -u $USER

# Check logs
tail -f logs/hpo_*.out
```

### Direct Python Execution
```bash
# Run SimPO HPO
python -m dpo.hpo.optuna_search \
    --config dpo/hpo/optuna_1.yaml \
    --algo simpo \
    --trials 20 \
    --study_name simpo_custom \
    --output_dir dpo/hpo/optuna_results

# Run DPO HPO
python -m dpo.hpo.optuna_search \
    --config dpo/hpo/optuna_4.yaml \
    --algo dpo \
    --trials 20 \
    --study_name dpo_custom \
    --output_dir dpo/hpo/optuna_results
```

## Output Structure
```
dpo/hpo/optuna_results/
├── simpo_20250114_123456/
│   ├── study.db              # Optuna database
│   ├── summary.json           # Top trials summary
│   ├── best_config.yaml       # Best hyperparameters
│   └── trial_*.json           # Individual trial results
└── dpo_20250114_234567/
    └── ...
```

## Scoring Function
The HPO objective combines multiple metrics:
```
score = accuracy + 0.05 * clip(margin, 0, 2.0) - 0.1 * z_score(perplexity)
```

- **Accuracy**: Primary metric (pref_acc for DPO, reward_acc for SimPO)
- **Margin**: Reward difference stability (capped at 2.0)
- **Perplexity**: Regularization to prevent degenerate solutions

## GPU Configuration Adjustments

### For A40 GPUs (48GB memory)
Configs 4-6 are pre-configured for A40s with:
- batch_size: 28
- precision: fp16
- Higher memory headroom

### For A100 GPUs (40GB memory)  
Configs 1-3 are pre-configured for A100s with:
- batch_size: 24
- precision: bf16
- Optimized for tensor cores

### Adjusting for Other GPUs
Modify batch_size in the config files:
- V100 (32GB): batch_size: 16-20
- T4 (16GB): batch_size: 8-12
- RTX 3090 (24GB): batch_size: 12-16

## Next Steps After HPO

1. **Select Top Candidates**: Review `summary.json` for best trials
2. **Full Evaluation**: Run top 3-5 configs with full training:
   ```bash
   python -m dpo.train --config dpo/hpo/optuna_results/*/best_config.yaml
   ```
3. **Benchmark**: Evaluate on full test set with 98×8 benchmark
4. **Multi-Seed Validation**: Test stability across 2-3 seeds

## Troubleshooting

### CUDA Out of Memory
- Reduce batch_size in config
- Increase grad_accum_steps to maintain effective batch size
- Switch to fp16 from bf16 if needed

### Slow Training
- Reduce max_steps for quicker trials
- Decrease val_every frequency
- Use fewer num_workers if I/O bound

### Poor HPO Results
- Expand search ranges if all trials hit boundaries
- Increase trials from 20 to 50
- Check if data quality issues are dominating

## Advanced Options

### Custom Scoring
Modify `compute_score()` in `optuna_search.py` to weight metrics differently.

### Pruning
Enable pruning for early stopping of bad trials:
```python
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1000)
```

### Distributed HPO
Use Ray Tune integration for multi-node optimization (requires additional setup).

## Contact
For issues or questions, please refer to the main project documentation.