# dpo/ref_manager.py
import torch
import torch.nn as nn

def clone_as_reference(build_model_fn, policy_state_dict, device: torch.device) -> nn.Module:
    """
    Build a fresh module and load a fully detached/cloned subset of the policy
    state_dict to avoid any storage aliasing between policy and reference.

    If the policy uses LoRA (extra adapter weights), we only load the keys
    that also exist in the reference and ignore the rest.
    """
    ref = build_model_fn().to("cpu")  # create on CPU first
    ref_sd = ref.state_dict()
    # keep only intersecting keys; detach & clone to avoid aliasing
    filtered = {
        k: v.detach().clone()
        for k, v in policy_state_dict.items()
        if k in ref_sd and isinstance(v, torch.Tensor)
    }
    # load non-strict so missing/extra keys (e.g., LoRA) don't error out
    ref.load_state_dict(filtered, strict=False)
    ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref
