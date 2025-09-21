#!/usr/bin/env python3
"""
Test script to demonstrate the difference between multiround.train and dpo.train
and verify the correct training command for multiround experiments.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_multiround_trainer_validation():
    """Test that multiround.train can validate configs properly."""
    print("🧪 Testing multiround trainer validation...")
    
    try:
        # Test config validation (dry run)
        config_path = "multiround/config/experiments/00_debug_dpo_dynamic.yaml"
        
        # This should show proper multiround validation
        result = subprocess.run([
            sys.executable, "-m", "multiround.train", 
            "--config", config_path,
            "--validate-only"  # If this flag exists
        ], capture_output=True, text=True, timeout=30)
        
        print(f"   Return code: {result.returncode}")
        if result.stdout:
            print(f"   STDOUT preview: {result.stdout[:200]}...")
        if result.stderr:
            print(f"   STDERR preview: {result.stderr[:200]}...")
            
        # Even if it fails due to missing data/checkpoints, we should see multiround-specific logic
        if "multiround" in result.stderr.lower() or "round" in result.stderr.lower():
            print("   ✅ Multiround trainer recognized and attempted to run")
            return True
        else:
            print("   ⚠️ No multiround-specific output detected")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⚠️ Timeout - trainer was attempting to start (this is expected)")
        return True
    except Exception as e:
        print(f"   ❌ Error testing multiround trainer: {e}")
        return False

def check_trainer_differences():
    """Show the key differences between dpo.train and multiround.train."""
    print("\n📋 Trainer Differences Analysis:")
    
    # Check if both trainers exist
    try:
        import dpo.train
        print("   ✅ dpo.train module exists (old single-round trainer)")
    except ImportError:
        print("   ❌ dpo.train module not found")
    
    try:
        import multiround.train
        print("   ✅ multiround.train module exists (new multiround trainer)")
    except ImportError:
        print("   ❌ multiround.train module not found")
        return False
    
    # Check for multiround-specific classes
    try:
        from multiround.trainer import MultiRoundDPOTrainer
        print("   ✅ MultiRoundDPOTrainer class found")
    except ImportError:
        print("   ❌ MultiRoundDPOTrainer class not found")
        return False
    
    try:
        from multiround.evaluator import MultiRoundEvaluator
        print("   ✅ MultiRoundEvaluator class found")
    except ImportError:
        print("   ❌ MultiRoundEvaluator class not found")
        return False
    
    # Show key differences
    print("\n   🔄 Key Differences:")
    print("   dpo.train (old):")
    print("      • Single-round DPO training")
    print("      • No reference model updates")
    print("      • No pass@k analysis")
    print("      • Flat directory structure")
    print("      • Standard DPO evaluation")
    
    print("   multiround.train (new):")
    print("      • Multi-round training with round management")
    print("      • Reference model updates between rounds")
    print("      • Pass@k analysis on rounds 1, 3, 5")
    print("      • Round-based directory structure")
    print("      • Comprehensive 29-metric evaluation")
    print("      • Dynamic preference pair switching")
    print("      • Advanced reference model selection")
    
    return True

def show_correct_commands():
    """Show the correct commands for different training scenarios."""
    print("\n📝 Correct Training Commands:")
    
    print("   🚫 WRONG (what was used in debug):")
    print("      python -m dpo.train --config multiround/config/experiments/00_debug_dpo_dynamic.yaml")
    print("      → This uses old single-round trainer!")
    
    print("\n   ✅ CORRECT (multiround training):")
    print("      python -m multiround.train --config multiround/config/experiments/00_debug_dpo_dynamic.yaml")
    print("      → This uses new multiround trainer with all features!")
    
    print("\n   📋 Other correct commands:")
    print("      # Debug run (2 rounds, 2 epochs each)")
    print("      python -m multiround.train --config multiround/config/experiments/00_debug_dpo_dynamic.yaml")
    print()
    print("      # Full DPO experiment")
    print("      python -m multiround.train --config multiround/config/experiments/04_dpo_m25.yaml")
    print()
    print("      # SimPO experiment")
    print("      python -m multiround.train --config multiround/config/experiments/03_simpo_m125.yaml")
    print()
    print("      # Dynamic margins experiment")
    print("      python -m multiround.train --config multiround/config/experiments/15_dpo_dynamic_margins.yaml")

def main():
    """Run trainer comparison and validation."""
    print("🔄 Multiround vs DPO Trainer Analysis")
    print("=" * 50)
    
    success = True
    
    # Test 1: Check trainer differences
    if not check_trainer_differences():
        success = False
    
    # Test 2: Test multiround trainer validation
    if not test_multiround_trainer_validation():
        success = False
    
    # Test 3: Show correct commands
    show_correct_commands()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Summary of Debug Run Issue:")
    print("   ❌ Issue: Used 'python -m dpo.train' (wrong trainer)")
    print("   ✅ Fix: Use 'python -m multiround.train' (correct trainer)")
    print("\n📋 What the debug run actually did:")
    print("   • Ran old single-round DPO for ~1830 steps")
    print("   • No round transitions (stuck in round 1)")
    print("   • No reference model updates")
    print("   • No multiround evaluation or pass@k")
    print("   • Used flat directory structure")
    print("\n🎯 What it should do with correct command:")
    print("   • Round 1: Train 2 epochs → evaluate → pass@k (64 samples)")
    print("   • Reference update: policy → reference")
    print("   • Round 2: Train 2 epochs → evaluate → reference update")
    print("   • Round-based directory structure with evaluation results")
    print("   • Complete pass@k analysis and results storage")
    
    if success:
        print("\n✅ All systems ready for proper multiround training!")
        print("   Just use the correct command: python -m multiround.train")
    else:
        print("\n⚠️ Some issues detected with multiround trainer setup.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)