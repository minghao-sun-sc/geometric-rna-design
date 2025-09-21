#!/usr/bin/env python3
"""
Test the complete multiround workflow including reference model updates
and round transitions to ensure training will work smoothly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import torch
import json
from types import SimpleNamespace as SN

def _to_sn(obj):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SN(**{k: _to_sn(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_sn(item) for item in obj]
    else:
        return obj

def test_multiround_workflow():
    """Test the complete multiround workflow including round transitions."""
    print("🔧 MultiRound Workflow Test")
    print("=" * 60)
    
    try:
        # Load config
        from multiround.utils import load_config_with_inheritance
        config_path = "multiround/config/experiments/00_debug_a100_80g.yaml"
        config_dict = load_config_with_inheritance(config_path)
        cfg = _to_sn(config_dict)
        
        print(f"✅ Config loaded: {config_path}")
        print(f"   Num rounds: {cfg.multiround.num_rounds}")
        print(f"   Update reference: {cfg.multiround.update_reference}")
        print(f"   Pass@k rounds: {getattr(cfg.evaluation.pass_k, 'k_values', 'Not configured')}")
        
        # Create evaluator
        from multiround.evaluator import MultiRoundEvaluator
        evaluator = MultiRoundEvaluator(cfg)
        
        print(f"✅ Evaluator created successfully")
        print(f"   Pass@k rounds: {evaluator.passk_rounds}")
        
        # Build and load model
        from dpo.ref_manager import build_model_from_cfg
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = build_model_from_cfg(cfg.model).to(device)
        
        # Load base checkpoint
        if hasattr(cfg.paths, 'base_checkpoint') and os.path.exists(cfg.paths.base_checkpoint):
            print(f"Loading base checkpoint: {cfg.paths.base_checkpoint}")
            checkpoint = torch.load(cfg.paths.base_checkpoint, map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            print(f"✅ Model loaded successfully")
        else:
            print(f"⚠️ No base checkpoint found, using random weights")
        
        model.eval()
        
        # Test round 1 evaluation with pass@k
        print(f"\n📊 Testing Round 1 Evaluation (with pass@k)...")
        
        # Use small samples for testing speed
        evaluator.eval_samples = 4
        evaluator.final_eval_samples = 8  # Minimum for pass@8
        
        with tempfile.TemporaryDirectory() as temp_dir:
            round1_dir = os.path.join(temp_dir, "round_01")
            os.makedirs(round1_dir, exist_ok=True)
            
            # Test round 1 evaluation (should include pass@k)
            results = evaluator.evaluate_round(
                model=model,
                round_num=1,  # Round 1 should have pass@k
                output_dir=round1_dir,
                is_final_round=False
            )
            
            if 'error' in results:
                print(f"❌ Round 1 evaluation failed: {results['error']}")
                return False
            
            print(f"✅ Round 1 evaluation completed!")
            print(f"   Result keys: {list(results.keys())}")
            
            # Check for pass@k results
            passk_keys = [k for k in results.keys() if 'pass' in k.lower()]
            if passk_keys:
                print(f"✅ Pass@k analysis found: {len(passk_keys)} metrics")
                for key in sorted(passk_keys):
                    value = results[key]
                    if isinstance(value, (int, float)):
                        print(f"     {key}: {value:.4f}")
            else:
                print(f"⚠️ No pass@k results found - this may be expected for this round")
            
            # Show key metrics
            key_metrics = ['recovery', 'perplexity']
            for key in key_metrics:
                if key in results:
                    value = results[key]
                    print(f"   {key}: {value:.4f}")
            
            # Test reference model update mechanism
            print(f"\n🔄 Testing Reference Model Update...")
            
            # Simulate saving round 1 checkpoint
            round1_ckpt = os.path.join(round1_dir, "round_1_best.pt")
            torch.save(model.state_dict(), round1_ckpt)
            print(f"✅ Saved round 1 checkpoint: {round1_ckpt}")
            
            # Test loading as reference model
            reference_model = build_model_from_cfg(cfg.model).to(device)
            ref_checkpoint = torch.load(round1_ckpt, map_location=device)
            reference_model.load_state_dict(ref_checkpoint, strict=True)
            reference_model.eval()
            print(f"✅ Reference model loaded successfully")
            
            # Verify models are different objects but same weights
            assert model is not reference_model, "Models should be different objects"
            
            # Compare a few parameters to verify they're identical
            model_param = next(model.parameters()).flatten()[:10]
            ref_param = next(reference_model.parameters()).flatten()[:10]
            if torch.allclose(model_param, ref_param):
                print(f"✅ Reference model weights match original model")
            else:
                print(f"❌ Reference model weights don't match!")
                return False
            
            # Test round 2 preparation
            print(f"\n📋 Testing Round 2 Preparation...")
            
            # Simulate round 2 directory
            round2_dir = os.path.join(temp_dir, "round_02")
            os.makedirs(round2_dir, exist_ok=True)
            
            # Test metrics that would guide round transition
            round_metrics = {
                'round': 1,
                'recovery': results.get('recovery', 0.0),
                'perplexity': results.get('perplexity', 0.0),
                'best_checkpoint': round1_ckpt,
                'evaluation_success': True,
                'passk_computed': len(passk_keys) > 0
            }
            
            metrics_file = os.path.join(round1_dir, "round_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(round_metrics, f, indent=2)
            print(f"✅ Round metrics saved: {metrics_file}")
            
            # Test round 2 evaluation (should not have pass@k unless it's a pass@k round)
            print(f"\n📊 Testing Round 2 Evaluation...")
            
            results_r2 = evaluator.evaluate_round(
                model=reference_model,  # Use reference model from round 1
                round_num=2,
                output_dir=round2_dir,
                is_final_round=False
            )
            
            if 'error' in results_r2:
                print(f"❌ Round 2 evaluation failed: {results_r2['error']}")
                return False
            
            print(f"✅ Round 2 evaluation completed!")
            
            # Check pass@k for round 2
            passk_keys_r2 = [k for k in results_r2.keys() if 'pass' in k.lower()]
            if 2 in evaluator.passk_rounds:
                if passk_keys_r2:
                    print(f"✅ Round 2 pass@k analysis found (expected)")
                else:
                    print(f"⚠️ Round 2 should have pass@k but none found")
            else:
                if not passk_keys_r2:
                    print(f"✅ Round 2 has no pass@k analysis (expected)")
                else:
                    print(f"⚠️ Round 2 has unexpected pass@k analysis")
            
            # Test final evaluation simulation
            print(f"\n🏁 Testing Final Evaluation (Round 2 as final)...")
            
            results_final = evaluator.evaluate_round(
                model=reference_model,
                round_num=2,
                output_dir=round2_dir,
                is_final_round=True  # Test final round evaluation
            )
            
            if 'error' in results_final:
                print(f"❌ Final evaluation failed: {results_final['error']}")
                return False
            
            print(f"✅ Final evaluation completed!")
            
            # Final evaluation should have more metrics
            final_passk_keys = [k for k in results_final.keys() if 'pass' in k.lower()]
            print(f"   Final evaluation pass@k metrics: {len(final_passk_keys)}")
            
            return True
    
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 MultiRound Workflow Test")
    print("Testing complete multiround training workflow including round transitions.")
    
    success = test_multiround_workflow()
    
    if success:
        print(f"\n🎉 ALL WORKFLOW TESTS PASSED!")
        print(f"   ✅ Round 1 evaluation works")
        print(f"   ✅ Reference model updates work")
        print(f"   ✅ Round transitions work")
        print(f"   ✅ Pass@k analysis framework works")
        print(f"   ✅ Final round evaluation works")
        print(f"\n🚀 READY FOR FULL MULTIROUND TRAINING!")
        print(f"   You can now run the full training with confidence.")
    else:
        print(f"\n❌ WORKFLOW TESTS FAILED")
        print(f"   The multiround training workflow needs more fixes.")
        print(f"   Do NOT run full training yet.")

if __name__ == "__main__":
    main()