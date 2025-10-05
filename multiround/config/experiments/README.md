# RiboPO Multi-Round Experiment Configurations

This directory contains experiment configurations for multi-round preference optimization training.

## Core Experiments (Production Ready)

### Baseline Comparisons
- **01_sft_ablation.yaml** - SFT-only baseline (no preference optimization)
- **02_dpo_m125.yaml** - DPO with 0.125*std preference margins
- **03_simpo_m125.yaml** - SimPO with 0.125*std preference margins  
- **04_dpo_m25.yaml** - DPO with 0.25*std preference margins
- **05_simpo_m25.yaml** - SimPO with 0.25*std preference margins

### Training Variants
- **06_dpo_m25_3r.yaml** - DPO with 3 rounds instead of 5
- **11_dpo_dynamic_margins.yaml** - Dynamic curriculum learning (0.25→0.125)
- **12_simpo_dynamic_margins.yaml** - SimPO dynamic curriculum learning

### Advanced Methods
- **15_dpo_dynamic_margins.yaml** - Main dynamic margin experiment
- **16_simpo_dynamic_margins.yaml** - SimPO dynamic margin experiment
- **17_dpo_m25_ema_ref.yaml** - DPO with exponential moving average reference

## Reference Update Ablations
- **07_no_ref_dpo_m25_3r.yaml** - No reference model updates
- **08_no_ref_dpo_dynamic_margins.yaml** - No ref updates + dynamic margins
- **09_no_ref_dpo_m25.yaml** - No ref updates, full training
- **10_no_ref_dpo_m125.yaml** - No ref updates with tight margins

## Parameter Studies
- **14_pure_dpo_m25.yaml** - DPO without SFT regularization
- **18_dpo_dynamic_ref_10e.yaml** - 10 epochs per round
- **19_dpo_dynamic_ema_10e.yaml** - EMA with 10 epochs
- **20_dpo_ema_dynamic_10e.yaml** - EMA dynamic training

## Beta Parameter Ablations
- **21_no_ref_dpo_0.13_dynamic_10e.yaml** - β=0.13
- **22_no_ref_dpo_0.12_dnm_20e.yaml** - β=0.12, 20 epochs
- **23_no_ref_dpo_0.125_10e.yaml** - β=0.125
- **24_no_ref_dpo_0.25_10e.yaml** - β=0.25
- **25_no_ref_dpo_b0.1_0.125_10e.yaml** - β=0.1
- **26_no_ref_dpo_b0.11_0.125_10e.yaml** - β=0.11
- **27_no_ref_dpo_b0.09_0.125_10e.yaml** - β=0.09

## Reduced Training Studies
- **13_no_ref_dpo_m25_10e_3r.yaml** - 3 rounds, 10 epochs each
- **28_no_ref_dpo_dnm_b0.10_5e.yaml** - 5 epochs per round

## Directory Structure

```
experiments/
├── README.md                    # This file
├── debug/                      # Debug configurations (archived)
│   ├── 00_debug_a100_80g.yaml
│   ├── 00_debug_a40.yaml
│   └── ...
├── 01_sft_ablation.yaml        # Core experiments (01-12)
├── 15_dpo_dynamic_margins.yaml # Main production configs
└── 21-28_*.yaml               # Parameter studies
```

## Usage

```bash
# Run a core experiment
python multiround/train.py --config multiround/config/experiments/02_dpo_m125.yaml

# Run dynamic curriculum learning  
python multiround/train.py --config multiround/config/experiments/15_dpo_dynamic_margins.yaml

# Run ablation study
python multiround/train.py --config multiround/config/experiments/25_no_ref_dpo_b0.1_0.125_10e.yaml
```

## Configuration Inheritance

All experiments inherit base settings from:
- `multiround/config/multiround_defaults.yaml`
- `dpo/configs/defaults.yaml`

Override specific parameters as needed in each experiment file.

## Recommended Experiments for Papers

### Main Results
1. **15_dpo_dynamic_margins.yaml** - Primary method
2. **02_dpo_m125.yaml** - Static margin baseline
3. **01_sft_ablation.yaml** - SFT-only baseline

### Ablation Studies  
4. **14_pure_dpo_m25.yaml** - No SFT regularization
5. **09_no_ref_dpo_m25.yaml** - No reference updates
6. **25_no_ref_dpo_b0.1_0.125_10e.yaml** - Beta parameter study

Use `multiround/config/evaluation/` configs for comprehensive evaluation of trained models.