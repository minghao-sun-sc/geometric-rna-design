# RiboPO v2 Project Structure

## Overview
This document outlines the organization of the RiboPO v2 project, which implements a comprehensive RNA preference optimization framework with cleaned, high-quality datasets.

## Directory Structure

```
ribopo_v2/
├── PROJECT_STRUCTURE.md              # This file
├── data_clean/                        # Original cleaned dataset (post-gap removal)
│   ├── train/                         # 5,746 clean training structures
│   ├── val/                           # 365 clean validation structures  
│   ├── test/                          # 182 clean test structures
│   ├── native_seq/                    # Extracted sequences
│   │   ├── train/
│   │   │   └── train_native_sequences.fasta
│   │   ├── val/
│   │   │   └── val_native_sequences.fasta
│   │   └── test/
│   │       └── test_native_sequences.fasta
│   └── eda/                           # Exploratory Data Analysis
│       ├── scripts/                   # Analysis scripts
│       │   ├── comprehensive_eda.py
│       │   ├── extract_native_sequences.py
│       │   ├── outlier_analysis.py
│       │   └── filter_length_outliers.py
│       └── results/                   # Analysis outputs
│           ├── eda_summary_report.md
│           ├── outlier_analysis_report.md
│           ├── filtering_summary_report.md
│           ├── data_cleaning_summary_report.md
│           └── plots/                 # Visualization outputs
├── data_clean_filtered/               # Length-filtered dataset (recommended for training)
│   ├── train/                         # 5,044 filtered training structures  
│   ├── val/                           # 360 filtered validation structures
│   ├── test/                          # 182 filtered test structures
│   └── native_seq/                    # Filtered sequences
│       ├── train/
│       │   └── train_native_sequences.fasta
│       ├── val/
│       │   └── val_native_sequences.fasta
│       └── test/
│           └── test_native_sequences.fasta
└── debug/                            # Development and analysis scripts
    └── data_clean/
        ├── analyze_full_dataset_gaps.py      # Gap analysis script
        ├── gap_analysis_summary.json         # Gap analysis results
        └── benchmark_check_gap.py            # Initial gap checking script
```

## Dataset Statistics

### Original Dataset (post-gap cleaning)
| Split | Structures | Mean Length | Std Length | Mean GC% |
|-------|------------|-------------|------------|----------|
| TRAIN | 5,746      | 212.6 nt    | 518.0 nt   | 56.6%    |
| VAL   | 365        | 27.5 nt     | 31.3 nt    | 29.2%    |
| TEST  | 182        | 77.3 nt     | 39.9 nt    | 52.1%    |

### Filtered Dataset (recommended)
| Split | Structures | Mean Length | Std Length | Mean GC% |
|-------|------------|-------------|------------|----------|
| TRAIN | 5,044      | 87.2 nt     | 73.9 nt    | 49.8%    |
| VAL   | 360        | 27.4 nt     | 31.4 nt    | 29.1%    |
| TEST  | 182        | 77.3 nt     | 39.9 nt    | 52.1%    |

## Data Quality Assurance

### Gap Analysis Results
- **Total PDB files processed:** 12,011
- **Clean structures:** 6,293 (52.4% retention)
- **Files with gaps removed:** 5,718 (47.6%)

### Filtering Results  
- **Length-based filtering:** Removed 707 extreme outliers
- **Standard deviation reduction:** 85.7% in training set
- **Overall retention:** 88.8% of clean data preserved

### Quality Metrics
- ✅ **Structural integrity:** All gaps and discontinuities removed
- ✅ **Sequence extraction:** 100% success rate with validation
- ✅ **Distribution stability:** Training set variance reduced by 85.7%
- ✅ **Biological relevance:** Appropriate length ranges preserved

## Key Files and Scripts

### Data Processing Scripts
1. **`debug/data_clean/analyze_full_dataset_gaps.py`**
   - Comprehensive gap detection across all splits
   - Identifies missing residues and chain breaks
   - Copies clean structures to organized directories

2. **`data_clean/eda/scripts/extract_native_sequences.py`**
   - Extracts sequences from clean PDB structures
   - Applies robust RNA nucleotide mapping
   - Generates FASTA files for each split

3. **`data_clean/eda/scripts/filter_length_outliers.py`**
   - Implements length-based filtering strategy
   - Removes extreme outliers while preserving diversity
   - Creates filtered dataset for stable DPO training

### Analysis and Reports
1. **`data_clean/eda/results/data_cleaning_summary_report.md`**
   - Comprehensive overview of entire cleaning process
   - Statistical analysis and quality metrics
   - Recommendations for DPO training

2. **`debug/data_clean/gap_analysis_summary.json`**
   - Detailed gap analysis results in JSON format
   - Per-file gap detection outcomes
   - Summary statistics and retention rates

## Usage Recommendations

### For DPO Training
- **Primary dataset:** Use `data_clean_filtered/` 
- **Rationale:** Stable length distributions essential for DPO convergence
- **Benefit:** 85.7% reduction in training variance improves learning

### For Analysis and Validation
- **Sequence files:** Use FASTA files in `native_seq/` subdirectories
- **Structure validation:** Original PDB files in respective split directories
- **Quality metrics:** Refer to analysis reports in `eda/results/`

### For Development
- **Scripts:** All processing scripts preserved in `debug/` and `eda/scripts/`
- **Reproducibility:** Complete pipeline documented and executable
- **Verification:** Gap analysis and filtering results fully validated

## Integration with RiboPO v1

### Inherited Components
- **Base model:** gRNAde architecture and checkpoints
- **Evaluation pipeline:** `src/evaluator.py` and metric calculations
- **Training framework:** DPO/SimPO implementation from `dpo/` module
- **Multi-round setup:** Configuration and workflow from `multiround/`

### v2 Enhancements  
- **Clean dataset:** Gap-free, length-filtered structures
- **Improved distributions:** Stable variance for reliable training
- **Comprehensive analysis:** Full EDA and quality validation
- **Organized structure:** Clear separation of clean vs filtered data

## Next Steps

### Immediate Actions
1. **Validate dataset integrity** before training begins
2. **Configure training parameters** for filtered dataset characteristics  
3. **Set up monitoring** for length distribution stability during training

### DPO Training Preparation
1. **Use filtered dataset** (`data_clean_filtered/`) as primary training data
2. **Adjust batch sizes** based on new length distributions
3. **Monitor preference pair generation** with stable length ranges
4. **Validate model outputs** against expected sequence characteristics

### Quality Monitoring
1. **Track generated sequence lengths** during training
2. **Validate structural folding** of generated sequences
3. **Monitor training stability** with reduced variance
4. **Compare results** against v1 baseline performance

---

**Status:** Data cleaning complete, ready for DPO training  
**Quality:** Production-ready, fully validated dataset  
**Recommendation:** Proceed with multi-round preference optimization using filtered dataset