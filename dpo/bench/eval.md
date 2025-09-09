# DPO-RNA Evaluation and Benchmarking Guide

## Overview

This evaluation framework benchmarks RNA inverse folding models, comparing the base gRNAde model against DPO-finetuned checkpoints. It evaluates how well models can predict RNA sequences that fold into given 3D structures.

## Metrics Explained

### 1. **Token Accuracy** (`token_acc`)
- **What it measures**: Percentage of correctly predicted nucleotides across all sequences
- **Calculation**: `correct_tokens / total_tokens`
- **Interpretation**: Higher is better. Base gRNAde achieves ~68.3%

### 2. **Sequence Exact Match** (`seq_exact`)
- **What it measures**: Percentage of sequences predicted perfectly (100% match)
- **Calculation**: `sequences_with_all_tokens_correct / total_sequences`
- **Interpretation**: Higher is better. Very challenging metric (base model: 0%)

### 3. **Negative Log-Likelihood per Token** (`nll_per_token`)
- **What it measures**: Average cross-entropy loss per nucleotide
- **Calculation**: `sum(cross_entropy_loss) / total_tokens`
- **Interpretation**: Lower is better. Measures model confidence

### 4. **Perplexity** (`ppl`)
- **What it measures**: Exponential of NLL, interpretable as "average branching factor"
- **Calculation**: `exp(nll_per_token)`
- **Interpretation**: Lower is better. PPL=2.36 means model is choosing between ~2.36 equally likely options on average

### 5. **Graphs Count**
- **What it measures**: Number of test structures evaluated
- **Current test set**: 98 RNA structures from DAS split

## How Metrics Are Computed

The evaluation uses **teacher forcing**: given a 3D RNA structure and the ground-truth sequence, the model predicts each nucleotide conditioned on:
- The 3D backbone geometry (via GNN encoder)
- Previous ground-truth nucleotides (autoregressive)

```python
# Simplified evaluation loop
for structure, true_sequence in test_set:
    # Model predicts logits for each position
    logits = model(structure, true_sequence)  # [L, 4] 
    
    # Compute metrics
    predictions = logits.argmax(dim=-1)
    correct = (predictions == true_sequence).sum()
    nll = cross_entropy(logits, true_sequence)
```

## Running Evaluations

### 1. Evaluate Base Model Only

```bash
python -m dpo.bench.eval_benchmark --config dpo/configs/bench.yaml
```

### 2. Compare Multiple Checkpoints

Edit `dpo/configs/bench.yaml` to add your trained checkpoints:

```yaml
paths:
  checkpoints:
    - name: "BASE_gRNAde_1state_das"
      path: checkpoints/gRNAde_ARv1_1state_das.h5
    - name: "DPO_round1_best"
      path: runs/offline_dpo/dpo_rna_v2/best.pt
    - name: "DPO_round1_step1000"  
      path: runs/offline_dpo/dpo_rna_v2/steps/step_001000.pt
```

Then run the same command - it will evaluate all checkpoints and create a comparison table.

### 3. Change Test Split

To evaluate on different data splits, modify `split_name` in config:

```yaml
paths:
  split_name: test  # Options: train | val | test
```

## Output Files

### CSV Summary
- **Location**: `runs/benchmark/benchmark_summary.csv`
- **Contents**: One row per checkpoint with all metrics
- **Use case**: Easy comparison, plotting, analysis

### WandB Logging
- **Project**: DPO-RNA (configurable)
- **Metrics**: All metrics logged with checkpoint name prefix
- **Visualization**: Automatic comparison charts between models

## Baseline Results (gRNAde on DAS Test Set)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Token Accuracy | 68.33% | Predicts 2/3 of nucleotides correctly |
| Sequence Exact | 0% | No perfect sequences (very hard task) |
| Perplexity | 2.36 | Good confidence, low uncertainty |
| NLL/token | 0.857 | Moderate loss per position |

## Expected Improvements from DPO

Based on protein DPO results, we expect fine-tuning to potentially:
- **Increase token accuracy** by 3-8% (to ~71-76%)
- **Reduce perplexity** by 10-20% (to ~2.0-2.1)
- **Achieve some exact matches** (>0% sequence exact)

## Troubleshooting

### Device Errors
- Ensure `featurizer.device: cpu` in configs
- Set `num_workers: 0` to avoid CUDA multiprocessing issues

### Sequence Length Mismatches
- The evaluation handles length mismatches gracefully by using the featurized graph sequence
- Some structures may be skipped if severe mismatches occur

### Missing Checkpoints
- Verify checkpoint paths exist before running
- Use absolute paths or paths relative to project root

## Configuration Files

- **Basic benchmark config**: `dpo/configs/bench.yaml` - for teacher-forced sequence recovery only
- **Full evaluation config**: `dpo/configs/bench_full.yaml` - for all metrics including self-consistency
- **Single eval config**: `dpo/configs/eval.yaml` - for evaluating one DPO checkpoint

## Full Evaluation with Self-Consistency Metrics

### Running Complete gRNAde-style Evaluation

The full evaluation (`eval_full.py`) includes all metrics from the original gRNAde paper:

```bash
# Evaluate with all metrics (sampling + self-consistency)
python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml

# Customize sampling parameters
python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml \
    --n_samples 16 \
    --temperature 0.8 \
    --save_designs
```

### Additional Metrics Explained

#### 2D Self-Consistency (Secondary Structure)
- **What it measures**: Whether designed sequences fold into the target secondary structure
- **Tool**: EternaFold (already installed in `tools/EternaFold`)
- **Metric**: Matthews Correlation Coefficient (MCC) between predicted and target structures
- **Interpretation**: Higher is better (range: -1 to 1, good: >0.5)

#### 3D Self-Consistency (Tertiary Structure)  
- **What it measures**: Whether designed sequences fold into the target 3D structure
- **Tool**: RhoFold (already installed in `tools/rhofold`)
- **Metrics computed**:
  - **RMSD**: Root Mean Square Deviation in Ångstroms (lower is better, good: <2Å)
  - **TM-score**: Template Modeling score (higher is better, range: 0-1, good: >0.45)
  - **GDT**: Global Distance Test (higher is better, range: 0-1, good: >0.50)
  - **Threshold percentages**: % of samples meeting quality thresholds

### Expected Results

Based on gRNAde paper benchmarks on DAS test set:

| Metric | Base gRNAde | Expected with DPO |
|--------|-------------|-------------------|
| Recovery | 68.3% | 71-75% |
| Perplexity | 2.36 | 2.0-2.2 |
| 2D Self-consistency (MCC) | ~0.50 | 0.52-0.55 |
| 3D RMSD | ~3.5Å | 3.0-3.3Å |
| 3D TM-score | ~0.40 | 0.42-0.45 |
| % RMSD ≤ 2Å | ~15% | 20-25% |

### Sampling vs Teacher-Forcing

- **Teacher-forcing** (`eval_benchmark.py`): Uses ground-truth sequence as input, measures prediction accuracy
- **Sampling** (`eval_full.py`): Generates new sequences, evaluates if they fold correctly
- Sampling is more realistic but computationally expensive (requires folding predictions)

### Computational Requirements

- **Basic evaluation**: ~1 minute for 98 test structures
- **Full evaluation with RhoFold**: ~30-60 minutes depending on n_samples
- **GPU recommended** for RhoFold predictions
- **Disk space**: ~100MB for saved designs if `save_designs=true`