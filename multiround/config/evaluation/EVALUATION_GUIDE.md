# Comprehensive Evaluation Guide

## 🎯 Quick Start

Use `00_debug_comprehensive.yaml` as your evaluation template:

```bash
python -m dpo.bench.eval_full --config multiround/config/evaluation/eval_run/00_debug_comprehensive.yaml
```

## 📝 Configuration Guide

### Required Modifications for Each Evaluation

1. **Output Directory** (REQUIRED):
   ```yaml
   eval:
     out_dir: multiround/eval_multiround/{your_experiment_name}/
   ```

2. **Checkpoints** (REQUIRED):
   ```yaml
   paths:
     checkpoints:
       - name: "your_checkpoint_name"
         path: path/to/your/checkpoint.pt
   ```

### Optional Modifications

3. **Dataset Size**:
   ```yaml
   paths:
     small_dataset: false  # true=17 structures (quick), false=98 structures (full)
   ```

4. **Pass@k Analysis**:
   ```yaml
   eval:
     passk:
       enable: false  # true=pass@k analysis (slower), false=metrics only
   ```

## 📊 Evaluation Modes

### Mode 1: Quick Testing
```yaml
small_dataset: true      # 17 structures
passk.enable: false      # No pass@k
n_samples: 8            # 8 samples per structure
# Runtime: ~30-45 minutes
```

### Mode 2: Standard Evaluation
```yaml
small_dataset: false     # 98 structures  
passk.enable: false      # No pass@k
n_samples: 8            # 8 samples per structure
# Runtime: ~2-3 hours
```

### Mode 3: Comprehensive Analysis
```yaml
small_dataset: false     # 98 structures
passk.enable: true       # With pass@k
n_samples_passk: 64     # 64 samples for pass@k
# Runtime: ~8-12 hours
```

## 🔍 Metrics Included (29 Total)

### Core Metrics (Always Computed)
- **Sequence Recovery**: % correctly recovered nucleotides
- **Perplexity**: Model confidence (lower = better)
- **2D Self-Consistency**: EternaFold MCC score
- **3D Self-Consistency**: RhoFold metrics (RMSD, TM-score, GDT, pLDDT)
- **Thermodynamics**: Vienna RNA (MFE, Ensemble Defect)
- **Diversity**: 3-mer correlation with native

### Advanced Metrics (RhoFold Extended)
- **Interaction Network Fidelity**: INF_all, INF_WC, INF_non-WC, INF_stack
- **Clash Scores**: MolProbity all-atom clash detection
- **lDDT**: Local distance difference test
- **MCQ**: Mean circular quantities (torsional accuracy)

## 📈 Baseline Performance (gRNAde)

From evaluation on 17 test structures:

| Metric | Value | Interpretation |
|--------|--------|----------------|
| Sequence Recovery | 47.9% | Moderate baseline |
| TM-score | 0.126 | Low structural similarity |
| RMSD | 15.0 Å | High deviation |
| pLDDT | 0.565 | Moderate confidence |
| Vienna MFE | -22.2 kcal/mol | Good thermodynamics |
| INF_all | 0.429 | Moderate contact accuracy |

### Pass@k Baseline Results
- **TM-score ≥ 0.45**: 8.8% → 11.8% (k=1→8)
- **RMSD ≤ 4.0Å**: 8.8% → 11.8% (k=1→8)  
- **MFE ≤ -15.0**: 96.9% → 100% (k=1→8)

## 📁 Output Structure

```
multiround/eval_multiround/{experiment_name}/
├── full_eval_{name}.csv                    # Main results table
├── eval_{name}_{timestamp}.json            # Detailed JSON results
└── passk_analysis/                         # Pass@k analysis (if enabled)
    ├── {name}_metric_distributions.png     # Distribution plots
    ├── {name}_metric_statistics.json       # Statistical summaries
    └── {name}_passk_detailed.json          # Detailed pass@k results
```

## 🚀 Example Configurations

### Single Checkpoint Evaluation
```yaml
eval:
  out_dir: multiround/eval_multiround/exp21_dpo_final/
paths:
  checkpoints:
    - name: "exp21_round5_final"
      path: multiround/runs/21_no_ref_dpo_0.13_dynamic_10e/round_05/final.pt
```

### Multi-Checkpoint Comparison
```yaml
eval:
  out_dir: multiround/eval_multiround/comparison_dpo_rounds/
paths:
  checkpoints:
    - name: "baseline_gRNAde"
      path: checkpoints/gRNAde_ARv1_1state_das.h5
    - name: "exp21_round1"
      path: multiround/runs/21_no_ref_dpo_0.13_dynamic_10e/round_01/best.pt
    - name: "exp21_round3"
      path: multiround/runs/21_no_ref_dpo_0.13_dynamic_10e/round_03/best.pt
    - name: "exp21_final"
      path: multiround/runs/21_no_ref_dpo_0.13_dynamic_10e/round_05/final.pt
```

## ⚠️ Important Notes

1. **Vienna MFE Success Rate**: Expect ~23% success due to sequence length mismatches (normal)
2. **High Clash Scores**: Expected for unrefined structures (use `use_relax: true` for analysis)
3. **Memory Requirements**: RhoFold requires significant GPU memory (~8GB per structure)
4. **Runtime Estimates**: Include structure prediction overhead (~2-3 min per structure)

## 🔧 Troubleshooting

### Common Issues
- **CUDA OOM**: Reduce `batch_size` or `n_samples`
- **Slow evaluation**: Disable `use_lddt` or use `small_dataset: true`
- **Vienna failures**: Normal for mismatched sequences, filtered automatically
- **Missing checkpoints**: Verify file paths are correct

### Performance Optimization
- **Quick testing**: `small_dataset: true`, `passk.enable: false`
- **Production**: `small_dataset: false`, `passk.enable: true`  
- **GPU memory**: Use `batch_size: 16` for smaller GPUs
- **CPU workers**: Adjust `num_workers` based on available cores

This evaluation system provides comprehensive analysis for comparing DPO training results against the baseline gRNAde model.