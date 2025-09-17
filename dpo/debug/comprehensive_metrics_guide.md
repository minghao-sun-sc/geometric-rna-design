# Comprehensive Metrics Guide for RiboPO Evaluation

## Overview

The RiboPO evaluation pipeline now supports comprehensive metrics including all the additional metrics you requested. Here's how to access them.

## Available Metrics

### 1. Basic Metrics (Fast)
- `recovery`: Sequence recovery rate
- `perplexity`: Model perplexity

### 2. 2D Structure Metrics
- `sc_eternafold`: Secondary structure self-consistency using EternaFold

### 3. 3D Structure Metrics (Extended RhoFold)
- `sc_rhofold`: Comprehensive 3D evaluation including:
  - **RMSD**: Root mean square deviation (Å)
  - **TM-score**: Template modeling score (0-1, higher is better)
  - **GDT**: Global distance test (0-1, higher is better)
  - **pLDDT**: Predicted local confidence score (0-1, higher is better)
  - **INF scores**: Interaction Network Fidelity
    - INF (all): Overall interaction fidelity
    - INF (WC): Watson-Crick base pair fidelity
    - INF (non-WC): Non-Watson-Crick interaction fidelity
    - INF (stack): Base stacking interaction fidelity
  - **Clash score**: MolProbity structural clash score (lower is better)

### 4. Thermodynamics Metrics (Vienna RNA)
- `sc_score_vienna`: Vienna RNA package metrics including:
  - **MFE**: Minimum free energy (kcal/mol, lower is better)
  - **Ensemble Defect/nt**: Normalized ensemble defect (lower is better)
  - **P(target)**: Probability of target structure (higher is better)
  - **Shannon Entropy**: Positional entropy (higher means more flexible)
  - **Tm**: Melting temperature (°C)

## How to Run Comprehensive Evaluation

### Option 1: Quick Test (Basic + 3D metrics only)
```bash
python -m dpo.bench.eval_full \
  --config dpo/configs/experiments/test_small_eval.yaml \
  --n_samples 8 \
  --temperature 0.1 \
  --metrics recovery perplexity sc_rhofold
```

### Option 2: All 2D + 3D metrics
```bash
python -m dpo.bench.eval_full \
  --config dpo/configs/experiments/test_small_eval.yaml \
  --n_samples 8 \
  --temperature 0.1 \
  --metrics recovery perplexity sc_eternafold sc_rhofold
```

### Option 3: Complete Comprehensive Evaluation (ALL metrics)
```bash
python -m dpo.bench.eval_full \
  --config dpo/configs/experiments/test_small_eval.yaml \
  --n_samples 8 \
  --temperature 0.1 \
  --metrics recovery perplexity sc_eternafold sc_rhofold sc_score_vienna
```

### Option 4: Full Benchmark (All metrics + pass@k analysis)
```bash
python -m dpo.bench.eval_benchmark \
  --config dpo/configs/experiments/test_small_eval.yaml \
  --n_samples 64 \
  --temperature 0.1
```

## Expected Runtime

- **Basic metrics only**: ~5-10 minutes for 98 structures
- **Basic + 3D extended**: ~15-20 minutes for 98 structures  
- **All metrics**: ~25-30 minutes for 98 structures (includes Vienna thermodynamics)
- **Full benchmark**: ~45-60 minutes for 98 structures (includes pass@k analysis)

## Example Output

When running with all metrics, you'll see output like:

```
[Evaluating] dpo_v10_step_5k <- runs/offline_dpo/dpo_v10_beta0.10_lambda0.10_1e-4/steps/step_0005000.pt
  Recovery: 0.5044
  Perplexity: 1.15
  2D Self-consistency (EternaFold): 0.4123
  3D Self-consistency:
    RMSD: 12.11 Å
    TM-score: 0.2411
    GDT: 0.2348
    pLDDT: 0.8234
    % RMSD ≤ 2Å: 36.73%
    % TM ≥ 0.45: 22.96%
    % GDT ≥ 0.50: 19.90%
    INF (all): 0.3456
    INF (WC): 0.4123
    INF (non-WC): 0.2789
    INF (stack): 0.5234
    Clash score: 12.34
  Thermodynamics (Vienna):
    MFE: -23.45 kcal/mol
    Ensemble Defect/nt: 0.1234
    P(target): 0.3456
    Shannon Entropy: 2.345
    Tm: 67.8 °C
```

## Output Files

Results are saved to:
- **JSON**: `dpo/eval_results/eval_{split}_{checkpoint_names}_{timestamp}.json`
- **CSV**: `dpo/eval_results/{split}/full_eval_{split}.csv`
- **Latest symlink**: `dpo/eval_results/eval_{split}_latest.json`

## Configuration Options

You can customize evaluation by:

1. **Adjusting sample count**: `--n_samples N` (more samples = better statistics, longer runtime)
2. **Changing temperature**: `--temperature T` (lower = more deterministic, higher = more diverse)
3. **Selecting specific metrics**: `--metrics metric1 metric2 ...`
4. **Saving designs**: `--save_designs` (saves generated sequences to FASTA files)

## Performance Tips

1. **Start with fewer samples** (--n_samples 2-4) for testing
2. **Use basic metrics first** to verify the pipeline works
3. **Run overnight for full evaluations** with high sample counts
4. **Check intermediate results** in the symlinked latest.json file

## Troubleshooting

If you encounter issues:

1. **Check external tools** are properly installed (Vienna RNA, Phenix, etc.)
2. **Verify checkpoint paths** in your config file
3. **Start with a smaller dataset** (like test_small_eval.yaml)
4. **Monitor GPU memory** usage during evaluation

The evaluation pipeline is now fully functional with all comprehensive metrics!