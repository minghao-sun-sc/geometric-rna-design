# dpo/ref_manager.py
from __future__ import annotations
import torch
from typing import Optional, Dict, Any
from dpo.model_factory import build_model  # use our factory, not src/


def clone_as_reference(build_model_fn, policy_state_dict, device):
    """
    Build a fresh module and load a fully detached/cloned state_dict
    to avoid any storage aliasing between policy and reference.
    """
    ref = build_model_fn().to(device)
    with torch.no_grad():
        ref.load_state_dict({k: v.detach().clone() for k, v in policy_state_dict.items()}, strict=False)
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

    def __init__(self, cfg: Dict[str, Any], device: torch.device, policy_model: Optional[torch.nn.Module] = None):
        self.cfg = cfg
        self.device = device
        self.ref_model: torch.nn.Module = self._init_reference(policy_model)

    def _build_model(self) -> torch.nn.Module:
        return build_model(self.cfg).to(self.device)

    def _init_reference(self, policy_model: Optional[torch.nn.Module]) -> torch.nn.Module:
        if policy_model is not None:
            return clone_as_reference(self._build_model, policy_model.state_dict(), self.device)
        else:
            ref = self._build_model()
            ref.eval()
            for p in ref.parameters():
                p.requires_grad_(False)
            return ref

    @torch.no_grad()
    def update_from_policy(self, policy_model: torch.nn.Module):
        state = {k: v.detach().clone() for k, v in policy_model.state_dict().items()}
        self.ref_model.load_state_dict(state, strict=False)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
