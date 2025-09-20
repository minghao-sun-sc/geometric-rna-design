# RiboPO Multi-Round Training & Evaluation

## **Overview**

The `multiround/` directory contains the **complete multi-round DPO/SimPO training system** and **comprehensive evaluation pipeline** for RNA inverse folding. This system supports curriculum learning with dynamic margins and full evaluation with 12 validated metrics.

## **🗂️ Directory Structure**

```
multiround/
├── config/
│   ├── evaluation/              # Evaluation configurations
│   │   └── 01_eval_base_dpo_t05.yaml  # Multi-checkpoint comparison
│   └── experiments/            # Training experiment configurations (1-12)
│       ├── 01_sft_ablation.yaml       # SFT-only baseline
│       ├── 02_dpo_margin125.yaml      # DPO with 0.125*std margins
│       ├── 03_simpo_margin125.yaml    # SimPO with 0.125*std margins
│       ├── 04_dpo_margin25.yaml       # DPO with 0.25*std margins
│       ├── 05_simpo_margin25.yaml     # SimPO with 0.25*std margins
│       ├── 06_dpo_beta_ablation.yaml  # Beta parameter study
│       ├── 07_dpo_epochs_ablation.yaml # Epoch count study
│       ├── 08_simpo_beta_ablation.yaml # SimPO beta study
│       ├── 09_sft_lambda_ablation.yaml # SFT regularization study
│       ├── 10_full_evaluation.yaml    # Comprehensive evaluation
│       ├── 11_dpo_dynamic_margins.yaml # Dynamic curriculum (25→125)
│       └── 12_simpo_dynamic_margins.yaml # SimPO dynamic curriculum
├── debug/
│   └── integrative/            # Essential debugging tools
│       ├── test_fixed_lddt.py         # Validates lDDT chain mapping
│       ├── analyze_chain_structure.py # PDB chain analysis
│       └── find_test_structures.py    # Dataset structure mapping
├── docs/                      # Documentation
│   ├── evaluate.md            # Evaluation usage guide
│   └── training.md            # Training configuration guide
├── eval_multiround/           # Evaluation outputs
├── trainer.py                # Core multi-round trainer
└── README.md                 # This file
```

## **🚀 Quick Start**

### **Training**
```bash
# Run multi-round DPO training
python multiround/trainer.py --config multiround/config/experiments/02_dpo_margin125.yaml

# Run dynamic curriculum learning
python multiround/trainer.py --config multiround/config/experiments/11_dpo_dynamic_margins.yaml
```

### **Evaluation**
```bash
# Multi-checkpoint comparison (recommended)
python -m dpo.bench.eval_full --config multiround/config/evaluation/01_eval_base_dpo_t05.yaml

# Full test set evaluation
python -m dpo.bench.eval_full --config multiround/config/evaluation/01_eval_base_dpo_t05.yaml \
    --max_structures 98 --n_samples 8 --use_relax
```

## **📊 Evaluation System**

### **Complete Metrics (12 total, all working)**
✅ **Basic**: Recovery, Perplexity  
✅ **2D Structure**: EternaFold self-consistency  
✅ **3D Structure**: RMSD, TM-score, GDT, pLDDT, lDDT (chain mapping fixed)  
✅ **Advanced**: INF (all, WC, non-WC, stack), Clash scores (pre/post relax), MCQ  
✅ **Thermodynamics**: Vienna MFE, ED, entropy, p(S0), diversity, Tm  
✅ **Sequence**: 3-mer diversity  

### **Key Features**
- **Multi-checkpoint comparison** in single run
- **WandB integration** with consistent tables
- **Pass@k analysis** for success rate evaluation
- **Dual clash scores** (pre/post Amber relaxation)
- **Chain mapping** for multi-chain RNA structures

## **🧪 Experiment Configurations**

### **Core Comparisons (1-5)**
1. **SFT Ablation**: SFT-only baseline for comparison
2. **DPO Margin 125**: DPO with 0.125*std preference margins  
3. **SimPO Margin 125**: SimPO with 0.125*std preference margins
4. **DPO Margin 25**: DPO with 0.25*std preference margins
5. **SimPO Margin 25**: SimPO with 0.25*std preference margins

### **Parameter Studies (6-9)**
6. **DPO Beta**: β parameter ablation (0.08, 0.12, 0.16)
7. **DPO Epochs**: Training length study (10, 20, 30)
8. **SimPO Beta**: SimPO β parameter study (1.5, 2.0, 2.5)
9. **SFT Lambda**: SFT regularization study (0.05, 0.10, 0.20)

### **Advanced Methods (10-12)**
10. **Full Evaluation**: Comprehensive metrics and pass@k analysis
11. **DPO Dynamic**: Curriculum learning (margin 25→125 over rounds)
12. **SimPO Dynamic**: SimPO curriculum learning

### **Configuration Inheritance**
All experiments inherit from `dpo/configs/defaults.yaml`:
```yaml
inherit_from: ../dpo/configs/defaults.yaml  # ✅ Ensures consistent base settings
```

## **🔧 Configuration Options**

### **Training Parameters**
```yaml
training:
  epochs: 20              # Training epochs per round
  rounds: 5               # Multi-round iterations
  dynamic_margins: true   # Enable curriculum learning
  margin_schedule:        # Dynamic margin progression
    - round: [1, 2]
      pair_margin: "25"
    - round: [3, 4, 5]  
      pair_margin: "125"
```

### **Evaluation Parameters**
```yaml
eval:
  max_structures: 98      # Full test set (or 8 for debugging)
  n_samples: 8           # Samples per structure for robust statistics
  temperature: 0.1       # Conservative sampling for quality
  use_relax: true        # Enable Amber relaxation (dual clash scores)
  use_lddt: true         # Enable lDDT calculation (fixed)
  save_designs: true     # Save all designed sequences
```

### **Metrics Selection**
```yaml
metrics:
  - recovery             # Sequence recovery (basic)
  - perplexity          # Model confidence
  - sc_eternafold       # 2D self-consistency
  - sc_rhofold          # 3D self-consistency (all structural metrics)
  - sc_vienna           # Thermodynamic stability
  - diversity_3mer      # Sequence diversity
```

## **🐛 Debug Tools**

### **Validation Scripts**
- **`test_fixed_lddt.py`**: Validates lDDT chain mapping fix
- **`analyze_chain_structure.py`**: Analyzes PDB chain structure for debugging
- **`find_test_structures.py`**: Maps dataset indices to structure IDs

### **Usage**
```bash
# Test lDDT functionality
python multiround/debug/integrative/test_fixed_lddt.py

# Analyze PDB chain structure
python multiround/debug/integrative/analyze_chain_structure.py

# Find test structure details
python multiround/debug/integrative/find_test_structures.py
```

## **📈 Monitoring & Logging**

### **WandB Integration**
All experiments automatically log to Weights & Biases:
- **Training metrics**: Loss, accuracy, KL divergence
- **Evaluation metrics**: All 12 validation metrics
- **Comparison tables**: Multi-checkpoint performance
- **Progress tracking**: Round-by-round improvement

### **Output Files**
```
runs/multiround_{experiment_name}/
├── checkpoints/
│   ├── round_1_best.pt
│   ├── round_2_best.pt
│   └── ...
├── logs/
├── evaluation/
│   ├── round_1_eval.json
│   └── ...
└── wandb/
```

## **🔗 Related Documentation**

- **[Evaluation Guide](../docs/evaluation_guide.md)**: Complete evaluation system documentation
- **[Evaluation Modules](../docs/evaluation_modules.md)**: Technical implementation details  
- **[Cleanup Proposal](../docs/cleanup_proposal.md)**: Codebase maintenance recommendations

## **✅ Verification**

### **Test Training Configuration**
```bash
# Verify config inheritance works
python -c "
import yaml
with open('multiround/config/experiments/02_dpo_margin125.yaml') as f:
    config = yaml.safe_load(f)
print('✅ Inheritance:', config.get('inherit_from'))
"
```

### **Test Evaluation Pipeline**
```bash
# Quick evaluation test
python -m dpo.bench.eval_full --config multiround/config/evaluation/01_eval_base_dpo_t05.yaml

# Should output: All 12 metrics with real values (no NaN for lDDT)
```

---

## **🎯 Current Status**

✅ **All 12 evaluation metrics working**  
✅ **lDDT chain mapping fixed**  
✅ **12 experiment configurations ready**  
✅ **Multi-checkpoint evaluation working**  
✅ **WandB integration complete**  
✅ **Pass@k analysis implemented**  
✅ **Dual clash score reporting**  
✅ **Configuration inheritance working**  

**The system is production-ready for comprehensive RNA inverse folding research.**