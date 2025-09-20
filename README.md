# RiboPO

Preference Optimization (DPO/SimPO) for RNA inverse folding (temperorary name)


Multi-Round PO for the training. 

Structural and Thermostability Rewards & Feedback

Various Evaluation Metrics



## Project Overview

RiboPO implements preference optimization methods (DPO/SimPO) to fine-tune the gRNAde RNA inverse folding model. The goal is to improve RNA sequence design that correctly folds into target 3D structures, critical for RNA therapeutics and synthetic biology.

## Key Commands

### Training
```bash
# SimPO training (recommended - more memory efficient)
python -m dpo.train --config dpo/configs/defaults.yaml --loss_type simpo

# DPO training  
python -m dpo.train --config dpo/configs/defaults.yaml --loss_type dpo

# Override specific parameters
python -m dpo.train --config dpo/configs/defaults.yaml --run_name custom_name --wandb_mode offline

# SLURM submission (SoC cluster)
sbatch dpo/scripts/train.slurm dpo/configs/defaults.yaml
```

### Evaluation
```bash
# Full evaluation with sampling (comprehensive metrics)
python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml --n_samples 8 --temperature 0.5

# Basic teacher-forced evaluation (fast)
python -m dpo.bench.eval_benchmark --config dpo/configs/bench.yaml

# SLURM submission
sbatch dpo/scripts/eval.slurm dpo/configs/bench_full.yaml
sbatch dpo/scripts/eval_dna.slurm dpo/configs/bench_full.yaml  # DNA cluster
```

### Hyperparameter Optimization
```bash
# Run HPO with Optuna
sbatch dpo/scripts/hpo_optuna.slurm dpo/hpo/optuna_1.yaml
```

### Testing & Debugging
```bash
# Run debug tests from project root
python dpo/debug/test_vienna_examples_suite.py
python dpo/debug/test_evaluator_usalign_pair.py
```

## Architecture

### Core Components
- **src/**: Original gRNAde codebase (unmodified, preserve as-is)
  - `models.py`: AutoregressiveMultiGNNv1 model definition
  - `data/`: Dataset classes, RNA graph featurizers
  - `evaluator.py`: Evaluation metrics (recovery, perplexity, self-consistency)
  
- **dpo/**: Preference optimization implementation (all modifications here)
  - `train.py`: Main training entry point
  - `trainer.py`: DPO/SimPO trainer with graph batching
  - `losses.py`: DPO and SimPO loss implementations
  - `data.py`: Preference pair dataset with RBF error handling
  - `bench/`: Evaluation pipelines (eval_full.py, eval_benchmark.py)
  - `hpo/`: Hyperparameter optimization configs

- multiround ribopo

### Key Design Patterns
1. **Zero-modification approach**: All DPO logic isolated in `dpo/` folder
2. **Graph batching**: PyTorch Geometric's Batch for GPU efficiency
3. **Error resilience**: Graceful RBF featurization failure handling
4. **Device flexibility**: CPU featurization to avoid CUDA issues
5. **Multi-round training**: Reference model reset each round (DPO only)

## Data Flow

1. **Preference pairs** from `data/pairs_margin125/by_das/clean/`
   - Winners: RMSD < 8Å AND pLDDT > 0.7
   - Losers: Opposite criteria
   - Confidence margin: 0.125σ difference

2. **Featurization** converts PDB → PyG graphs
   - Node features: RNA backbone geometry
   - Edge features: Distance-based interactions
   - RBF expansion for continuous features

3. **Loss computation**
   - SimPO: `-log σ(β*(avg_logp_w - avg_logp_l) - γ)` with length norm
   - DPO: `-log σ(β*(log π/π_ref difference))` + SFT regularization

## Critical Configuration

### SimPO Parameters (Recommended)
```yaml
loss_type: simpo
simpo:
  beta: 2.0      # Reward scaling [1.5-2.5]
  gamma: 0.5     # Target margin [0.3-1.2]  
  sft_lambda: 0  # Optional regularization
```

### DPO Parameters
```yaml
loss_type: dpo
dpo:
  beta: 0.12     # Temperature [0.1-0.5]
  sft_lambda: 0.12  # Regularization [0.1-0.2]
```

### Training Settings
```yaml
training:
  batch_size: 4        # Adjust for GPU memory
  grad_accum_steps: 4  # Effective batch = 16
  num_workers: 8       # CPU parallelization
optimizer:
  lr: 1.8e-4          # [1e-4 to 5e-6]
```

## Environment Setup

### Required Paths (in configs)
- `PROJECT_PATH`: /mnt/rna01/smh/projects/ribopo
- `tools/EternaFold`: 2D structure evaluation
- `tools/x3dna-v2.4`: 3D structure tools
- `tools/USalign`: Structure alignment

### SLURM Configuration
- Update paths in `dpo/scripts/*.slurm`
- Conda activation: `source ~/miniconda3/bin/activate rna`
- GPU constraint: `--constraint="xgpg|xgph"` for A100s

## Common Issues & Solutions

1. **RBF expansion errors**: Automatically skipped, check logs for frequency
2. **GPU memory**: Reduce batch_size or increase grad_accum_steps
3. **Featurization failures**: Check PDB file validity, sequence-structure length match
4. **SLURM paths**: Ensure absolute paths in configs match cluster setup

## Monitoring

W&B metrics (wandb.ai):
- **SimPO**: `reward_acc` (key metric), `avg_logp_w/l`, `z_margin`
- **DPO**: `pref_acc`, `margin`, `loss_dpo`, `sft_loss`
- **Validation**: `val/recovery`, `val/perplexity`
- **Training**: `loss`, `lr`, GPU utilization

## Development Workflow

1. **Make changes only in `dpo/` directory** - never modify `src/`
2. **Test locally first**: Use debug scripts in `dpo/debug/`
3. **Check GPU utilization**: Aim for >70% with proper batching
4. **Monitor convergence**: SimPO typically converges faster than DPO
5. **Validate checkpoints**: Use eval_benchmark.py for quick validation




Important commands:

eval baselines

python evaluate_baselines.py --models rdesign --limit 15 --output_dir /tmp/test_baseline_eval_limited


