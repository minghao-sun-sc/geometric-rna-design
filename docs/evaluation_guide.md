# RNA Inverse Folding Evaluation Guide

## Overview

The RiboPO evaluation system provides comprehensive assessment of RNA inverse folding models using 12 validated metrics across structural, thermodynamic, and sequence diversity aspects. This guide covers the complete working evaluation pipeline, metrics, and usage instructions.

## Quick Start

### **Current Working Pipeline (Recommended)**

```bash
# Multi-checkpoint comparison (RECOMMENDED)
python -m dpo.bench.eval_full --config multiround/config/evaluation/01_eval_base_dpo_t05.yaml

# Single checkpoint evaluation  
python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml

# Custom sampling for robust statistics
python -m dpo.bench.eval_full --config multiround/config/evaluation/01_eval_base_dpo_t05.yaml \
    --max_structures 98 --n_samples 8 --temperature 0.1

# Full evaluation with all optimizations
python -m dpo.bench.eval_full --config multiround/config/evaluation/01_eval_base_dpo_t05.yaml \
    --max_structures 98 --use_relax --save_designs
```

### **Key Files**
- **Main Pipeline**: `dpo/bench/eval_full.py` - Complete working evaluation
- **Core Functions**: `src/evaluator.py` - All 12 metrics (fully validated)  
- **Multi-checkpoint Config**: `multiround/config/evaluation/01_eval_base_dpo_t05.yaml`

## Configuration

The evaluation is controlled via YAML configuration files. Key parameters:

```yaml
eval:
  # Sampling parameters
  n_samples: 1                  # Number of sequences to sample per structure (1-64)
  temperature: 0.1              # Sampling temperature (0.01-1.0)
  
  # Structure optimization
  use_relax: false              # Enable Amber relaxation
                                # - true: Reports both pre/post relax clash scores
                                # - false: Reports only pre-relax clash score, post-relax = NaN
  use_lddt: true                # Enable lDDT calculation (isolated OpenStructure v2)
  
  # Data paths
  split_name: test              # Dataset split: train | val | test
  out_dir: runs/benchmark_full  # Output directory for results
  
  # Metrics to compute
  metrics:
    - recovery                  # Sequence recovery (teacher-forced)
    - perplexity               # Model confidence/uncertainty
    - sc_eternafold            # 2D structure self-consistency
    - sc_rhofold               # 3D structure self-consistency + all structural metrics
    - vienna_mfe               # Thermodynamic stability (MFE)
    - vienna_ED                # Thermodynamic designability (Ensemble Defect)
    - diversity_3mer           # Sequence diversity (3-mer correlation)
  
  # Save options
  save_designs: false          # Save all designed sequences as FASTA files
```

## Input Data

### Required Files
- **Processed data**: `data/processed.pt` - Featurized RNA structures 
- **Data split**: `data/das_split.pt` - Train/validation/test splits
- **Model checkpoint**: Path specified in config (`checkpoints/` or `runs/`)

### Input Structure Format
Each input structure contains:
- **Sequence**: Native RNA sequence (string)
- **Coordinates**: 3D atomic coordinates (tensor)
- **Secondary structure**: Dot-bracket notation (string)
- **Metadata**: PDB ID, RFAM family, structural annotations

## Evaluation Process

### 1. Sequence Sampling
For each input structure:
- Sample `n_samples` sequences from the model at given `temperature`
- Lower temperature (0.1) = more conservative, higher quality
- Higher temperature (1.0) = more diverse, potentially lower quality

### 2. Structure Prediction
Designed sequences are folded using:
- **RhoFold**: 3D structure prediction (coordinates + confidence scores)
- **EternaFold**: 2D structure prediction (secondary structure)
- **Vienna RNA**: Thermodynamic ensemble analysis

### 3. Metric Calculation
Multiple evaluation metrics are computed and aggregated.

## Evaluation Metrics

### Basic Metrics
- **Recovery**: Sequence identity with native sequence (0-1, higher is better)
- **Perplexity**: Model confidence in predictions (lower is better)

### 2D Structure Metrics
- **EternaFold Self-Consistency**: MCC between predicted and native 2D structure (0-1, higher is better)

### 3D Structure Metrics (RhoFold Self-Consistency)
- **RMSD**: C4' coordinate deviation in Ångstroms (lower is better)
- **TM-score**: Template modeling score (0-1, higher is better, >0.45 = similar fold)
- **GDT_TS**: Global distance test (0-1, higher is better, >0.5 = good topology)
- **pLDDT**: Predicted local distance difference test from RhoFold (0-100, >70 = high confidence)
- **lDDT**: Local distance difference test vs native (0-1, higher is better, >0.7 = high accuracy) ✅ **FIXED: Chain mapping for multi-chain structures**

### Advanced Structural Metrics
- **INF (Interaction Network Fidelity)**: Base pairing accuracy
  - `INF-ALL`: All interactions (0-1, higher is better)
  - `INF-WC`: Watson-Crick pairs (0-1, higher is better) 
  - `INF-NWC`: Non-Watson-Crick pairs (0-1, higher is better)
  - `INF-STACK`: Base stacking (0-1, higher is better)
- **Clashscore (Pre-relax)**: Atomic clash detection before relaxation (lower is better, <50 = good quality)
- **Clashscore (Post-relax)**: Atomic clash detection after relaxation (lower is better, <20 = excellent quality, NaN when `use_relax=false`)
- **MCQ**: Mean of circular quantities for torsion angles (lower is better)

### Thermodynamic Metrics (Vienna RNA)
- **MFE**: Minimum free energy in kcal/mol (more negative = more stable)
- **Ensemble Defect (ED)**: Expected number of incorrectly paired nucleotides (lower is better)
- **Shannon Entropy**: Structural ensemble diversity (higher = more flexible)
- **p(S0)**: Probability of target structure in ensemble (0-1, higher is better)
- **Melting Temperature (Tm)**: Temperature at 50% folding (°C, context-dependent)

### Sequence Diversity Metrics
- **3-mer Correlation**: Sequence similarity via trinucleotide usage (0-1, higher = more similar)
- **Trimer Profile Novelty (TPN)**: Novelty vs natural RNAs (0-1, higher = more novel)

## Output Format

### JSON Results
```json
{
  "model_name": "eval_dpo_v5_best_temp0.1_samp1",
  "n_samples": 1,
  "temperature": 0.1,
  "dataset_size": 98,
  
  "metrics": {
    "recovery": 0.847,
    "perplexity": 2.134,
    "sc_eternafold": 0.723,
    "sc_rmsd": 3.45,
    "sc_tm": 0.678,
    "sc_gdt": 0.589,
    "sc_plddt": 78.9,
    "lddt": 0.654,
    "lddt_success_rate": 0.83,
    "inf_all": 0.712,
    "inf_wc": 0.834,
    "inf_nwc": 0.567,
    "inf_stack": 0.723,
    "clashscore_pre_relax": 45.2,
    "clashscore_post_relax": 23.4,
    "vienna_mfe": -12.34,
    "vienna_ED": 2.45,
    "vienna_pS0": 0.67,
    "vienna_Tm": 45.6,
    "diversity_3mer": 0.712
  },
  
  "success_rates": {
    "rmsd_within_8A": 0.78,
    "rmsd_within_2A": 0.34,
    "tm_within_0.45": 0.67,
    "gdt_within_0.5": 0.59,
    "plddt_within_70": 0.82
  }
}
```

### Console Output
```
[Evaluating] eval_dpo_v5_best_temp0.1_samp1 <- runs/offline_dpo/dpo_rna_v5/best.pt

┌─────────────────────────────────────────────────────────────────┐
│  Evaluation Results Summary                                     │
├─────────────────────────────────────────────────────────────────┤
│  Basic Metrics:                                                 │
│    Recovery: 84.7% - sequence identity with native             │
│    Perplexity: 2.13 - model confidence                         │
├─────────────────────────────────────────────────────────────────┤
│  Structure Quality Metrics:                                     │
│    RMSD: 3.45 Å - coordinate deviation                         │
│    TM-score: 0.678 - structural similarity ✓                   │
│    GDT_TS: 0.589 - structural accuracy ✓                       │
│    lDDT: 0.654 - local distance accuracy (success: 83.0%)      │
│    pLDDT: 78.9 - confidence score ✓                            │
│    Clashscore: 23.4 - atomic quality ✓                         │
├─────────────────────────────────────────────────────────────────┤
│  Thermodynamic Metrics:                                         │
│    MFE: -12.34 kcal/mol - stability                            │
│    Ensemble Defect: 2.45 nt - designability                   │
│    Melting Temp: 45.6°C - thermal stability                   │
└─────────────────────────────────────────────────────────────────┘

Results saved to: runs/benchmark_full/eval_dpo_v5_best_temp0.1_samp1_20250918_222208.json
```

### Saved Files (if save_designs=true)
```
runs/benchmark_full/
├── designs_eval_dpo_v5_best_temp0.1_samp1_20250918_222208/
│   ├── sample0/
│   │   ├── all_designs.fasta     # All designed sequences with metrics
│   │   ├── design0.pdb           # Predicted 3D structure
│   │   └── design0.fasta         # Designed sequence
│   └── sample1/
│       └── ...
├── eval_dpo_v5_best_temp0.1_samp1_20250918_222208.json
└── latest.json -> eval_dpo_v5_best_temp0.1_samp1_20250918_222208.json
```

## Usage Examples

### Evaluate Base Model
```bash
python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml
```

### High-Diversity Sampling
```bash
python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml \
    --n_samples 8 --temperature 0.5
```

### Structure-Only Metrics
```bash
python -m dpo.bench.eval_full --config dpo/configs/bench_full.yaml \
    --metrics recovery sc_rhofold --save_designs
```

### Custom Configuration
```yaml
# custom_eval.yaml
eval:
  n_samples: 16
  temperature: 0.1
  use_lddt: true
  use_relax: true
  metrics:
    - recovery
    - perplexity  
    - sc_rhofold
    - vienna_mfe
    - vienna_ED
  save_designs: true
```

```bash
python -m dpo.bench.eval_full --config custom_eval.yaml
```

## Performance Considerations

### Computational Cost
- **n_samples=1**: ~20 seconds per structure (baseline)
- **n_samples=8**: ~2-3 minutes per structure  
- **n_samples=64**: ~15-20 minutes per structure
- **use_lddt=true**: +5-10 seconds per structure (OpenStructure calculation)
- **use_relax=true**: +30-60 seconds per structure (Amber minimization)

### Resource Requirements
- **GPU Memory**: 4-8GB (depends on model size)
- **Disk Space**: 10-100MB per evaluation (depends on save_designs)
- **External Tools**: RhoFold, EternaFold, Vienna RNA, OpenStructure, Phenix

## Troubleshooting

### Common Issues
1. **NetworkX conflicts**: ✅ Resolved via local imports and subprocess isolation
2. **Sequence length mismatches**: Normal for incomplete structures, results in NaN
3. **High clash scores**: Indicates poor structure quality, not evaluation errors
4. **lDDT calculation failures**: ✅ **FIXED**: Chain mapping implemented for multi-chain structures

### Output Interpretation
- **✓ symbols**: Indicate metrics above quality thresholds
- **NaN values**: Missing or failed calculations (normal for some structures)
- **Success rates**: Fraction of successful calculations for each metric
- **Warnings**: Usually structure quality issues, not pipeline errors

## Integration with Weights & Biases

The evaluation automatically logs results to W&B when enabled:

```yaml
eval:
  wandb:
    enable: true
    project: DPO-RNA
    entity: your-username
```

Creates comparison tables and metric tracking across different models and configurations.

## **Current Evaluation Status (✅ All Working)**

### **Validated Metrics (12 total)**
✅ **Basic**: Recovery, Perplexity  
✅ **2D Structure**: EternaFold self-consistency  
✅ **3D Structure**: RMSD, TM-score, GDT, pLDDT, lDDT (chain mapping fixed)  
✅ **Advanced**: INF (all, WC, non-WC, stack), Clash scores (pre/post relax), MCQ  
✅ **Thermodynamics**: Vienna MFE, Ensemble Defect, Shannon entropy, p(S0), diversity, Tm  
✅ **Sequence**: 3-mer diversity, Trimer Profile Novelty  

### **Key Implementation Files**
- **`src/evaluator.py`**: Complete evaluation functions (2304 lines, fully validated)
- **`dpo/bench/eval_full.py`**: Main evaluation pipeline with WandB integration  
- **`dpo/passk.py or multiround/passk.py`**: Pass@k analysis for success rate calculations
- **`tools/usalign_utils.py`**: US-align wrapper for structural alignment
- **`tools/run_phenix.sh`**: Phenix MolProbity wrapper for clash score calculation

### **Working Configurations**
- **`multiround/config/evaluation/01_eval_base_dpo_t05.yaml`**: Multi-checkpoint comparison
- **`dpo/configs/bench_full.yaml`**: Single checkpoint evaluation

### **Recent Fixes Applied (September 2024)**
1. **lDDT Chain Mapping**: ✅ **MAJOR FIX** - Implemented multi-chain structure support in `get_lddt_openstructure_v2()` with automatic chain mapping (e.g., 3B58_1_B-C-A with 3 chains), achieving 100% success rates with valid scores (e.g., lDDT=0.0660)
2. **Dual Clash Scores**: Pre/post relaxation reporting in `self_consistency_score_rhofold_extended()`
3. **WandB Table Consistency**: Column alignment across checkpoints in `eval_full.py`
4. **Metric Name Mapping**: Corrected config metric names to match pipeline expectations (`sc_eternafold` vs `sc_score_eternafold`)
5. **Evaluation Pipeline Reliability**: Now supports both single and multi-checkpoint comparisons with consistent 12-metric output

---

For questions or issues with the evaluation pipeline, please refer to the troubleshooting section or check the debug output in the console.