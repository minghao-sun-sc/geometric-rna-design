# RiboPO Project Structure

## Overview
RiboPO (Ribonucleic acid Preference Optimization) is a multi-round preference optimization framework for RNA inverse folding that uses Direct Preference Optimization (DPO) and SimPO to improve upon the gRNAde base model.

**Latest Update**: RiboPO v2 introduces clean dataset with gap-free structures and enhanced data quality for improved training.

## Directory Structure

### RiboPO v1 (Original)
```
ribopo/
├── 📁 configs/                    # Base model configurations
├── 📁 data/                      # Datasets and preference pairs
│   ├── pairs_margin125/          # Preference pairs (0.125*std)
│   ├── pairs_margin25/           # Preference pairs (0.25*std)
│   ├── raw/                      # Raw PDB structure files
│   └── das_split_raw_data/       # Test set data and analysis
├── 📁 dpo/                       # Single-round DPO implementation
│   ├── bench/                    # Evaluation scripts
│   ├── configs/                  # DPO training configurations
│   ├── debug/archive/            # Archived debug scripts
│   ├── eval_results/archive/     # Archived evaluation results
│   └── hpo/                      # Hyperparameter optimization
├── 📁 multiround/                # Multi-round DPO system ⭐
│   ├── config/
│   │   ├── experiments/          # Training experiment configs (36 files)
│   │   │   ├── debug/           # Debug configurations
│   │   │   ├── 01-12_*.yaml     # Core experiments
│   │   │   └── 13-28_*.yaml     # Ablation studies
│   │   └── evaluation/          # Evaluation configurations
│   ├── docs/                    # Documentation
│   ├── train.py                 # Training entry point
│   ├── trainer.py               # Multi-round trainer
│   └── evaluator.py             # Multi-round evaluator
├── 📁 src/                       # Core model implementation
│   ├── evaluator.py             # Main evaluation module (3000+ lines)
│   ├── models.py                # gRNAde model architecture
│   └── constants.py             # Project constants
├── 📁 tools/                     # External analysis tools
│   ├── phenix-*/                # MolProbity/Phenix installations
│   ├── ViennaRNA/               # RNA thermodynamics
│   ├── rhofold/                 # 3D structure prediction
│   ├── USalign/                 # Structure alignment
│   └── RNA_assessment/          # RNA quality metrics
└── 📁 notebooks/                 # Jupyter analysis notebooks
```

### RiboPO v2 (Clean Dataset) ✨ NEW
```
ribopo/ribopo_v2/
├── 📁 data/
│   └── native_clean/
│       └── native_struct_clean/    # 182 gap-free PDB files ✅
├── 📁 debug/
│   └── data_clean/
│       ├── benchmark_check_gap.py        # Gap analysis script ✅
│       ├── copy_clean_pdbs.py            # Clean PDB copy script ✅
│       ├── exploratory_data_analysis.py  # EDA analysis script ✅
│       ├── pdb_gap_analysis.json         # Gap analysis results ✅
│       ├── gap_summary.txt               # Human-readable summary ✅
│       ├── copy_clean_pdbs.log           # Copy operation log ✅
│       └── eda_results/                  # EDA output directory ✅
│           ├── sequence_length_distribution.png
│           ├── nucleotide_composition.png
│           ├── structural_features.png
│           ├── split_distribution.png
│           ├── eda_analysis_results.json
│           └── eda_summary_report.md     # Comprehensive EDA report ✅
└── [Future: config/, scripts/, analysis/]
```

## Core Components

### 1. Training System
- **Single-round**: `dpo/` - Original DPO implementation
- **Multi-round**: `multiround/` - Advanced curriculum learning system ⭐
  - 36 experiment configurations covering all major parameter studies
  - Dynamic preference pair margins (curriculum learning)
  - Reference model updates between rounds
  - Comprehensive evaluation pipeline

### 2. Evaluation System  
- **Main evaluator**: `src/evaluator.py` - 29 comprehensive metrics
  - Sequence: Recovery, perplexity, edit distance
  - 2D Structure: EternaFold self-consistency  
  - 3D Structure: RMSD, TM-score, GDT, pLDDT, lDDT
  - Interactions: INF (all, WC, non-WC, stack)
  - Quality: Clash scores (pre/post relax), MCQ
  - Thermodynamics: Vienna MFE, ensemble defect, entropy, Tm
  - Diversity: 3-mer correlation, sequence novelty

### 3. Data Management
- **Preference pairs**: Two margin levels (0.125 and 0.25 * std)
- **Test set**: 235 RNA structures with comprehensive mapping
- **Raw structures**: 1000+ PDB files for training/validation

### 4. External Tools Integration
- **Structure prediction**: RhoFold+ (no MSA, optional relax)
- **Thermodynamics**: ViennaRNA ensemble analysis
- **Quality assessment**: Phenix MolProbity, lDDT, INF metrics
- **Alignment**: US-align for structure comparison

## Key Experiments

### Production-Ready Experiments
1. **01_sft_ablation.yaml** - SFT-only baseline
2. **02_dpo_m125.yaml** - DPO with tight margins (0.125*std)
3. **15_dpo_dynamic_margins.yaml** - Main method (curriculum learning) ⭐
4. **16_simpo_dynamic_margins.yaml** - SimPO variant

### Parameter Studies
- **Beta ablation**: 21-27 (β = 0.09-0.25)
- **Training length**: 18-20 (5-20 epochs per round)
- **Reference updates**: 07-10 (no reference vs. dynamic updates)

## Evaluation Workflow

### Quick Evaluation (8 samples)
```bash
python -m dpo.bench.eval_full --config multiround/config/evaluation/01_eval_base_dpo_t05.yaml
```

### Full Evaluation (64 samples)
```bash
python -m dpo.bench.eval_full --config multiround/config/evaluation/01_eval_base_dpo_t05.yaml \
    --max_structures 98 --n_samples 64 --use_relax
```

### Training
```bash
python multiround/train.py --config multiround/config/experiments/15_dpo_dynamic_margins.yaml
```

## Recent Cleanup (Completed)

### Removed/Archived
- ✅ **70+ debug scripts** → `dpo/debug/archive/`
- ✅ **Backup files** (.bak, *~) → Deleted
- ✅ **Test directories** → `eval_results/archive/2025-09/`
- ✅ **Debug configs** → `experiments/debug/`

### Organized
- ✅ **Experiment configs** → Clear numbering and documentation
- ✅ **Evaluation results** → Date-based archiving system
- ✅ **Debug tools** → Keep only essential ones
- ✅ **.gitignore** → Updated with cleanup patterns

## Status

### ✅ Production Ready
- Multi-round training system
- All 29 evaluation metrics working
- Comprehensive experiment suite
- Pass@k analysis pipeline
- WandB integration

### 🔄 Active Development
- Code refactoring (split evaluator.py)
- Configuration system simplification
- Documentation consolidation
- Testing framework implementation

### 📋 Future Work
- Module separation (core/, training/, evaluation/)
- API documentation generation
- Continuous integration setup
- Memory optimization for large-scale training

## Quick Start

1. **Train a model**:
   ```bash
   python multiround/train.py --config multiround/config/experiments/15_dpo_dynamic_margins.yaml
   ```

2. **Evaluate results**:
   ```bash
   python -m dpo.bench.eval_full --config multiround/config/evaluation/01_eval_base_dpo_t05.yaml
   ```

3. **Run ablation study**:
   ```bash
   python multiround/train.py --config multiround/config/experiments/25_no_ref_dpo_b0.1_0.125_10e.yaml
   ```

For detailed usage, see `multiround/README.md` and individual configuration files.

## RiboPO v2 Data Quality Improvements ✅ COMPLETED

### Full Dataset Cleaning Completed (2025-10-04)

#### Comprehensive Gap Analysis Results
**Full DAS Dataset Analysis (Train/Val/Test)**:
- **Total structures processed**: 12,011 PDB files
- **Clean structures (gap-free)**: 6,293 (52.4%) ✅
- **Structures with gaps**: 5,718 (47.6%) - REMOVED
- **Quality improvement**: 100% gap-free training data

#### Split-wise Results
| Split | Total IDs | Found PDBs | Clean PDBs | Gap PDBs | Clean % |
|-------|-----------|------------|------------|----------|---------|
| **Train** | 11,244 | 11,244 | 5,746 | 5,498 | 51.1% |
| **Val** | 532 | 532 | 365 | 167 | 68.6% |
| **Test** | 235 | 235 | 182 | 53 | 77.4% |

#### Clean Dataset Organization
```
ribopo_v2/data/
├── train/          # 5,746 gap-free training PDBs ✅
├── val/            # 365 gap-free validation PDBs ✅
├── test/           # 182 gap-free test PDBs ✅
└── native_clean/   # Original test set analysis
```

#### Gap Distribution Analysis
- **Dominant gap sizes**: 1-residue gaps (35,138 total), 2-residue gaps (6,347 total)
- **Large gaps**: Up to 6,373 residues in worst cases
- **Training set most affected**: 48.9% gap rate vs 31.4% (val) and 22.6% (test)

#### Key Quality Improvements
1. **Training Data**: 5,746 gap-free structures for robust DPO learning ✅
2. **Validation Data**: 365 structures for model selection ✅
3. **Test Data**: 182 structures for final evaluation ✅
4. **Data Integrity**: 100% structurally complete across all splits ✅

#### Analysis Scripts & Reports
- **Full Dataset Analysis**: `ribopo_v2/debug/data_clean/analyze_full_dataset_gaps.py`
- **Gap Analysis Summary**: `ribopo_v2/debug/data_clean/full_dataset_gap_summary.txt`
- **Detailed Results**: `ribopo_v2/debug/data_clean/full_dataset_gap_analysis.json`
- **Copy Logs**: `ribopo_v2/debug/data_clean/full_dataset_copy_log.json`
- **Test EDA Report**: `ribopo_v2/debug/data_clean/eda_results/eda_summary_report.md`

#### Ready for RiboPO v2 Training
1. **Dataset Splits**: ✅ Complete train/val/test assignment
2. **Data Quality**: ✅ 100% gap-free structures
3. **Size Distribution**: ✅ Suitable for batch training
4. **Directory Structure**: ✅ Organized and ready

### Comparison: v1 vs v2
| Aspect | RiboPO v1 | RiboPO v2 |
|--------|-----------|-----------|
| **Total Dataset** | 12,011 (mixed quality) | 6,293 (gap-free) ✅ |
| **Training Set** | 11,244 (with gaps) | 5,746 (gap-free) ✅ |
| **Validation Set** | 532 (with gaps) | 365 (gap-free) ✅ |
| **Test Set** | 235 (with gaps) | 182 (gap-free) ✅ |
| **Data Quality** | Mixed (47.6% gaps) | 100% clean ✅ |
| **Training Ready** | Requires filtering | Ready for training ✅ |
| **Gap Analysis** | Not systematically done | Comprehensive analysis ✅ |