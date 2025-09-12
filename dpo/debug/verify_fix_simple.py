#!/usr/bin/env python
"""
Simple verification that collate functions don't use CUDA operations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def verify_collate_functions():
    """Verify collate functions don't have device transfer operations."""
    
    print("Verifying collate functions are CUDA-free...")
    
    # Read the data.py file
    data_file = "/mnt/rna01/smh/projects/offline-dpo/dpo/data.py"
    with open(data_file, 'r') as f:
        content = f.read()
    
    # Check for problematic patterns in collate functions
    issues = []
    
    # Find collate_batch_pairs function
    import re
    collate_pattern = r'def collate_batch_pairs.*?(?=\ndef |\Z)'
    collate_match = re.search(collate_pattern, content, re.DOTALL)
    
    if collate_match:
        collate_code = collate_match.group()
        print("\n1. Checking collate_batch_pairs function...")
        
        # Check for .to(device) or .to(target_device) or .cuda()
        if '.to(target_device)' in collate_code or '.to(device)' in collate_code or '.cuda()' in collate_code:
            issues.append("collate_batch_pairs contains device transfer operations!")
        else:
            print("   ✅ No device transfers found in collate_batch_pairs")
    
    # Find _collate_identity function
    identity_pattern = r'def _collate_identity.*?(?=\ndef |\Z)'
    identity_match = re.search(identity_pattern, content, re.DOTALL)
    
    if identity_match:
        identity_code = identity_match.group()
        print("\n2. Checking _collate_identity function...")
        
        if '.to(target_device)' in identity_code or '.to(device)' in identity_code or '.cuda()' in identity_code:
            issues.append("_collate_identity contains device transfer operations!")
        else:
            print("   ✅ No device transfers found in _collate_identity")
    
    # Check trainer for device transfers
    trainer_file = "/mnt/rna01/smh/projects/offline-dpo/dpo/trainer.py"
    with open(trainer_file, 'r') as f:
        trainer_content = f.read()
    
    print("\n3. Checking trainer for device transfers...")
    
    # Look for batch device transfers in train loops
    if 'batch.graph = batch.graph.to(self.device)' in trainer_content:
        print("   ✅ Found batch.graph device transfer in trainer")
    else:
        issues.append("Missing batch.graph device transfer in trainer!")
    
    if 'batch.winner_seq = batch.winner_seq.to(self.device)' in trainer_content:
        print("   ✅ Found batch.winner_seq device transfer in trainer")
    else:
        issues.append("Missing batch.winner_seq device transfer in trainer!")
    
    if 'batch.loser_seq = batch.loser_seq.to(self.device)' in trainer_content:
        print("   ✅ Found batch.loser_seq device transfer in trainer")
    else:
        issues.append("Missing batch.loser_seq device transfer in trainer!")
    
    # Summary
    print("\n" + "="*50)
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ ALL CHECKS PASSED!")
        print("\nThe fix is correctly implemented:")
        print("1. Collate functions keep data on CPU")
        print("2. Trainer moves batch to device after DataLoader")
        print("3. This avoids CUDA multiprocessing errors")
        return True

if __name__ == "__main__":
    success = verify_collate_functions()
    sys.exit(0 if success else 1)