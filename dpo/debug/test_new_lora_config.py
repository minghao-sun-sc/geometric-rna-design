#!/usr/bin/env python
"""
Test the new LoRA configuration to see how many parameters become trainable.
"""

import sys
import yaml
import torch
from pathlib import Path

from dpo.env_bootstrap import bootstrap_env
bootstrap_env()

sys.path.append(str(Path(__file__).parent.parent.parent))

from hydra.utils import instantiate
from dpo.lora import apply_lora, summarize_lora

def main():
    cfg = yaml.safe_load(open("dpo/configs/default.yaml"))
    
    print("Loading model...")
    model = instantiate(cfg["model"])
    
    print(f"\n=== BEFORE LORA ===")
    total_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_before:,}")
    print(f"Trainable params: {trainable_before:,}")
    
    print(f"\n=== APPLYING LORA ===")
    lora_cfg = cfg["lora"]
    target_modules = lora_cfg["target_modules"]
    print(f"Target modules: {target_modules}")
    
    # Count what will be matched
    matched_modules = []
    matched_params = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and any(target in name for target in target_modules):
            if hasattr(module, 'weight'):
                params = module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    params += module.bias.numel()
                matched_modules.append((name, params))
                matched_params += params
    
    print(f"Will apply LoRA to {len(matched_modules)} modules with {matched_params:,} total params")
    
    # Apply LoRA
    r = lora_cfg["r"]
    alpha = lora_cfg["alpha"]
    dropout = lora_cfg["dropout"]
    
    apply_lora(
        model, 
        r=r, 
        alpha=alpha, 
        dropout=dropout,
        target_modules=target_modules,
        train_bias="none"
    )
    
    print(f"\n=== AFTER LORA ===")
    summary = summarize_lora(model)
    print(f"Total params: {summary['total']:,}")
    print(f"Trainable params: {summary['trainable']:,}")
    print(f"LoRA-wrapped modules: {summary['wrapped_linear']}")
    print(f"Trainable ratio: {100 * summary['trainable'] / summary['total']:.2f}%")
    
    # Calculate expected LoRA params
    expected_lora_params = 0
    for name, orig_params in matched_modules:
        # For each Linear layer with in_features x out_features
        # LoRA adds: A (in_features x r) + B (r x out_features)  
        module = model.get_submodule(name)
        if hasattr(module, 'linear'):  # LoRALinear wrapper
            in_feat = module.in_features
            out_feat = module.out_features
            lora_params = in_feat * r + r * out_feat
            expected_lora_params += lora_params
    
    print(f"Expected LoRA params: {expected_lora_params:,}")
    
    print(f"\n=== TOP 10 LARGEST WRAPPED MODULES ===")
    wrapped_info = []
    for name, module in model.named_modules():
        if hasattr(module, 'A') and module.A is not None:  # LoRALinear
            lora_params = module.A.numel() + module.B.numel()
            orig_params = module.linear.weight.numel()
            if hasattr(module.linear, 'bias') and module.linear.bias is not None:
                orig_params += module.linear.bias.numel()
            wrapped_info.append((name, orig_params, lora_params))
    
    wrapped_info.sort(key=lambda x: x[1], reverse=True)
    for name, orig, lora in wrapped_info[:10]:
        print(f"  {name}: {orig:,} -> +{lora:,} LoRA params")

if __name__ == "__main__":
    main()