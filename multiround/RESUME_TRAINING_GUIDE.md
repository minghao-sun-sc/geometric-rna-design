# Resume Training Guide

## ✅ **Yes, training is now resumable from previous checkpoints!**

### 🔧 **How Resume Works**

The multi-round training system saves checkpoints after each round and can resume from any completed round.

**Checkpoint Structure:**
```
runs/{experiment_name}/
├── round_01/
│   ├── round_1_best.pt          # ← Resume point for round 2
│   └── eval_results/
├── round_02/
│   ├── round_2_best.pt          # ← Resume point for round 3
│   └── eval_results/
└── round_03/
    ├── round_3_best.pt          # ← Resume point for round 4
    └── eval_results/
```

### 🏃 **How to Resume Training**

#### **Option 1: Command Line Resume**
```bash
# Resume from round 3 (if rounds 1-2 completed)
python multiround/train.py multiround/config/experiments/04_dpo_m25.yaml --start_from_round 3

# Resume from round 5 (if rounds 1-4 completed)  
python multiround/train.py multiround/config/experiments/04_dpo_m25.yaml --start_from_round 5
```

#### **Option 2: Config File Resume**
```yaml
# Add to your config file:
multiround:
  current_round: 3              # Resume from round 3
  num_rounds: 5
  epochs_per_round: 20
```

### 📝 **Resume Examples**

#### **Example 1: Training Interrupted After Round 2**
```bash
# Original training (got interrupted after round 2)
python multiround/train.py multiround/config/experiments/04_dpo_m25.yaml

# Resume from round 3
python multiround/train.py multiround/config/experiments/04_dpo_m25.yaml --start_from_round 3
```

#### **Example 2: Skip to Final Round for Quick Testing**
```bash
# Skip to round 5 to test final evaluation (if rounds 1-4 exist)
python multiround/train.py multiround/config/experiments/04_dpo_m25.yaml --start_from_round 5
```

#### **Example 3: Extend Training (Add More Rounds)**
```bash
# Original: 5 rounds completed
# Extend to 7 rounds starting from round 6
python multiround/train.py multiround/config/experiments/04_dpo_m25.yaml \
  --start_from_round 6 \
  --num_rounds 7
```

### 🔍 **What Happens During Resume**

1. **Checkpoint Loading**: Loads the best checkpoint from `round_{N-1}_best.pt`
2. **Model State**: Restores policy model, optimizer, and scheduler states
3. **Training Context**: Continues with correct round numbering and logging
4. **Evaluation**: Continues from the specified round with full evaluation

### ⚠️ **Important Notes**

#### **Checkpoint Requirements**
- Resume from round N requires `round_{N-1}_best.pt` to exist
- If checkpoint is missing, training starts from the base model with a warning

#### **Configuration Consistency**  
- Model architecture must match (node_dims, edge_dims, etc.)
- Optimizer settings can be changed but may affect convergence
- Evaluation settings will use current config (can be modified)

#### **WandB Logging**
- Resume will continue logging to the same run if run_name matches
- Use different run_name to create a new WandB run for resumed training

### 🛠️ **Troubleshooting**

#### **Issue: "No checkpoint found for round X"**
```bash
# Check if checkpoint exists
ls runs/04_dpo_m25/round_02/round_2_best.pt

# If missing, start from an earlier round or round 1
python multiround/train.py config.yaml --start_from_round 1
```

#### **Issue: "Model architecture mismatch"**
- Ensure config model settings match the original training
- Check node_in_dim, edge_in_dim, num_layers, etc.

#### **Issue: "CUDA out of memory during resume"**
- Resume uses the same batch_size as original training
- Reduce batch_size or use gradient accumulation if needed

### 🎯 **Use Cases**

#### **1. Interrupted Training Recovery**
Most common use case - training gets interrupted due to:
- System shutdown/reboot
- SLURM job time limits  
- GPU memory issues
- Network interruptions

#### **2. Hyperparameter Adjustment**
- Change learning rate for later rounds
- Adjust evaluation settings
- Modify batch size or accumulation

#### **3. Extended Training**
- Add more rounds beyond original plan
- Test different round schedules

#### **4. Debugging/Analysis**
- Jump to specific rounds to test evaluation
- Re-run final round with different metrics
- Test checkpoint loading functionality

### ✅ **Resume Capabilities Summary**

- ✅ **Model State**: Policy, optimizer, scheduler fully restored
- ✅ **Round Tracking**: Correct round numbering and progress
- ✅ **Evaluation Continuity**: Full evaluation pipeline continues
- ✅ **Checkpoint Chain**: Automatic loading from previous round
- ✅ **Flexible Starting Point**: Start from any completed round
- ✅ **Error Handling**: Graceful fallback if checkpoints missing
- ✅ **CLI Support**: Easy command-line resume options

**Training is now fully resumable and robust to interruptions!** 🎉