# dpo/lora.py
from __future__ import annotations
import math
from typing import Sequence, Tuple, Set, List
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0, train_bias: str = "none"):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        in_f, out_f = base.in_features, base.out_features
        self.r = int(r)
        self.scaling = float(alpha) / float(max(1, r))
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        # Freeze base weights
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None and train_bias != "all":
            self.base.bias.requires_grad_(False)

        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(in_f, self.r))
            self.lora_B = nn.Parameter(torch.zeros(self.r, out_f))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x):
        y = self.base(x)
        if self.r and self.lora_A is not None:
            x_d = self.dropout(x)
            update = (x_d @ self.lora_A) @ self.lora_B
            y = y + self.scaling * update
        return y


def _name_matches(name: str, patterns: Sequence[str]) -> bool:
    # empty string "" matches all; if patterns contains "", match all linear modules
    return ("" in patterns) or any(p in name for p in patterns)


def _iter_module_tree(root: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Iterative DFS over the module tree, yielding (full_name, module).
    Avoid recursion and protect against cycles via a visited set.
    """
    stack: List[Tuple[str, nn.Module]] = [("", root)]
    visited: Set[int] = set()
    out: List[Tuple[str, nn.Module]] = []

    while stack:
        name, mod = stack.pop()
        mid = id(mod)
        if mid in visited:
            continue
        visited.add(mid)
        out.append((name, mod))

        # Push children
        for child_name, child in mod.named_children():
            full = f"{name}.{child_name}" if name else child_name
            stack.append((full, child))
    return out


def apply_lora(model: nn.Module,
               enabled: bool = True,
               r: int = 8,
               alpha: int = 16,
               dropout: float = 0.0,
               target_modules: Sequence[str] = ("linear", "proj", "fc", "out_proj"),
               train_bias: str = "none") -> int:
    """
    Replace selected nn.Linear modules with LoRALinear wrappers.
    Returns number of modules wrapped.
    If target_modules contains "", wrap ALL Linear modules.
    """
    if not enabled:
        return 0

    patterns = list(target_modules) if target_modules is not None else [""]
    if len(patterns) == 0:
        patterns = [""]  # wrap all

    wrapped = 0

    # Iterate parents so we can assign back into them safely
    for parent_name, parent in _iter_module_tree(model):
        # work on a snapshot to avoid mutation during iteration
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name

            # Skip if already LoRA-wrapped
            if isinstance(child, LoRALinear):
                continue

            if isinstance(child, nn.Linear) and _name_matches(full_name, patterns):
                lora = LoRALinear(child, r=r, alpha=alpha, dropout=dropout, train_bias=train_bias)
                setattr(parent, child_name, lora)
                wrapped += 1

    return wrapped


def mark_only_lora_as_trainable(model: nn.Module, train_bias: str = "none"):
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad_(True)
        elif n.endswith(".bias") and train_bias == "all":
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
