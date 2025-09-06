# dpo/ref_manager.py
from __future__ import annotations
import copy
import torch
from typing import Optional

# ---- keep your original utility ----
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


class RefManager:
    """
    Holds a frozen reference model for DPO.
    - ref_model: a frozen copy of the policy (or built from cfg/model_path)
    - update_from_policy(policy): refresh ref weights (for multi-round DPO)
    """

    def __init__(self, cfg: dict, device: torch.device, policy_model: Optional[torch.nn.Module] = None):
        self.cfg = cfg
        self.device = device
        self.ref_model: torch.nn.Module = self._init_reference(policy_model)

    def _build_model(self) -> torch.nn.Module:
        # Uses your existing model builder (loads architecture + weights from cfg.model_path)
        from src.models import build_model
        m = build_model(self.cfg).to(self.device)
        return m

    def _init_reference(self, policy_model: Optional[torch.nn.Module]) -> torch.nn.Module:
        if policy_model is not None:
            # Deep-clone from the current policy (recommended)
            return clone_as_reference(self._build_model, policy_model.state_dict(), self.device)
        else:
            # Fallback: build from cfg/model_path and freeze as-is
            ref = self._build_model()
            ref.eval()
            for p in ref.parameters():
                p.requires_grad_(False)
            return ref

    @torch.no_grad()
    def update_from_policy(self, policy_model: torch.nn.Module):
        """
        Called at the end of a DPO round to make the new policy the next reference.
        """
        state = {k: v.detach().clone() for k, v in policy_model.state_dict().items()}
        self.ref_model.load_state_dict(state)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
