#!/usr/bin/env python3
"""
Test multiround training initialization without full training.
"""

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

import torch
import sys
import os
sys.path.append('/mnt/rna01/smh/projects/ribopo')

from multiround.train import load_multiround_config, validate_config
from multiround.trainer import MultiRoundDPOTrainer

def test_training_init():
    """Test multiround training initialization."""
    config_path = "multiround/config/experiments/multiround_dynamic_margins_v1.yaml"
    
    print("🚀 Testing multiround training initialization...")
    
    try:
        # Load and validate config
        cfg = load_multiround_config(config_path)
        if not validate_config(cfg):
            return False
        
        print("✅ Configuration loaded and validated")
        
        # Initialize trainer
        print("\n🔧 Initializing multiround trainer...")
        trainer = MultiRoundDPOTrainer(cfg)
        print("✅ Trainer initialized successfully")
        
        # Test trainer setup for round 1
        print("\n📊 Testing trainer setup for round 1...")
        
        # This will test if the data loaders can be created without errors
        try:
            # Call _setup_trainer_for_round which is the internal method
            round_dir = os.path.join(trainer.output_root, f"round_{1:02d}")
            os.makedirs(round_dir, exist_ok=True)
            trainer._setup_trainer_for_round(1, round_dir)
            print("✅ Round 1 trainer setup completed successfully")
            
            # Check if we can create a small batch  
            if trainer.trainer and trainer.trainer.train_loader:
                train_loader = trainer.trainer.train_loader
                print(f"✅ Train loader created with {len(train_loader.dataset)} pairs")
                
                # Try to get one batch (this is where the original error occurred)
                print("🔍 Testing batch loading...")
                batch_iter = iter(train_loader)
                batch = next(batch_iter)
                print(f"✅ Successfully loaded batch with CID: {batch.cid}")
                
            else:
                print("❌ Train loader not available")
                return False
                
        except Exception as e:
            print(f"❌ Error during round setup or batch loading: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n✅ All initialization tests passed!")
        print("🎉 The multiround training should now work correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_training_init()
    sys.exit(0 if success else 1)