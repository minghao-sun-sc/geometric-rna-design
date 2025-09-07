#!/usr/bin/env python
"""
Inspect the model architecture to understand what linear layers exist
and how LoRA target patterns should be configured.
"""

import sys
import yaml
from pathlib import Path

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

sys.path.append(str(Path(__file__).parent.parent.parent))

from hydra.utils import instantiate

def main():
    cfg = yaml.safe_load(open("dpo/configs/default.yaml"))
    
    print("Loading model...")
    model = instantiate(cfg["model"])
    
    print(f"\n=== MODEL ARCHITECTURE ===")
    print(f"Model: {type(model).__name__}")
    
    print(f"\n=== ALL NAMED MODULES ===")
    linear_count = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'numel'):
            params = module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                params += module.bias.numel()
            total_params += params
            
            if 'Linear' in str(type(module)):
                linear_count += 1
                print(f"  {name}: {type(module).__name__} -> {params:,} params")
                if hasattr(module, 'in_features'):
                    print(f"    Shape: {module.in_features} -> {module.out_features}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Linear layers found: {linear_count}")
    
    print(f"\n=== TESTING CURRENT LORA CONFIG ===")
    # Test what the current config would match
    target_modules = cfg["lora"]["target_modules"]
    print(f"Current target_modules: {target_modules}")
    
    matched_modules = []
    for name, module in model.named_modules():
        if 'Linear' in str(type(module)):
            if any(target in name for target in target_modules):
                matched_modules.append(name)
    
    print(f"Modules that would be LoRA-wrapped: {len(matched_modules)}")
    for name in matched_modules:
        print(f"  - {name}")
    
    if not matched_modules:
        print("WARNING: No modules match current target patterns!")
        print("\nSuggested target_modules based on actual layer names:")
        all_linear_names = [name for name, module in model.named_modules() 
                          if 'Linear' in str(type(module))]
        # Extract common substrings
        common_patterns = set()
        for name in all_linear_names:
            parts = name.split('.')
            for part in parts:
                if part and not part.isdigit():  # Skip numeric indices
                    common_patterns.add(part)
        
        print(f"  Potential patterns: {sorted(common_patterns)}")

if __name__ == "__main__":
    main()