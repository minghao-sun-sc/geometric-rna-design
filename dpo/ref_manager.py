# dpo/ref_manager.py
import torch

def clone_as_reference(build_model_fn, policy_state_dict, device):
    """
    Build a fresh module and load a fully detached/cloned state_dict
    to avoid any storage aliasing between policy and reference.
    """
    ref = build_model_fn().to(device)
    with torch.no_grad():
        ref.load_state_dict({k: v.detach().clone() for k, v in policy_state_dict.items()})
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref
