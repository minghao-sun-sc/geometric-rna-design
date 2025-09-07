# dpo/lora.py
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Drop-in LoRA wrapper for nn.Linear.
    y = Linear(x) + scale * Dropout(x) @ A @ B
    where A in R^{in_features x r}, B in R^{r x out_features}, scale = alpha / r.
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        train_bias: str = "none",  # "none" or "all"
    ):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = (alpha / r) if r > 0 else 0.0

        # Base (frozen) linear
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(train_bias == "all")

        # LoRA adapters
        if self.r > 0:
            # A: in_features x r ; B: r x out_features
            self.A = nn.Parameter(torch.zeros(self.in_features, self.r))
            self.B = nn.Parameter(torch.zeros(self.r, self.out_features))
            nn.init.kaiming_uniform_(self.A, a=5**0.5)
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        if self.r > 0:
            # (.., in) @ (in, r) @ (r, out) -> (.., out)
            y = y + self.scaling * (self.drop(x) @ self.A @ self.B)
        return y


def _name_matches_any(name: str, substrings: Sequence[str]) -> bool:
    # If user passes [""] treat as "match all Linear"
    if len(substrings) == 0:
        return False
    if len(substrings) == 1 and substrings[0] == "":
        return True
    return any(s in name for s in substrings)


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Optional[Sequence[str]] = None,
    train_bias: str = "none",
) -> nn.Module:
    """
    Wrap matching nn.Linear modules in LoRALinear and freeze non-adapter weights.
    Returns the *same* model instance with modules replaced in-place.
    """
    if target_modules is None:
        target_modules = [""]

    # Replace in-place
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _name_matches_any(name, target_modules):
            # Find the parent to replace attribute
            parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            wrapped = LoRALinear(module, r=r, alpha=alpha, dropout=dropout, train_bias=train_bias)
            setattr(parent, attr, wrapped)

    # Freeze everything except LoRA params (and bias if requested)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            # LoRA params trainable
            if m.A is not None:
                m.A.requires_grad_(True)
            if m.B is not None:
                m.B.requires_grad_(True)
            # bias flag handled inside LoRALinear
        elif isinstance(m, nn.Linear):
            # Any remaining Linear (not wrapped) is fully frozen unless user wants bias
            m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(train_bias == "all")
        else:
            # leave other modules as-is; the upstream ARv1 has most non-linear params frozen anyway
            pass

    return model


def summarize_lora(model: nn.Module) -> dict:
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    n_wrapped = sum(1 for m in model.modules() if isinstance(m, LoRALinear))
    return {"trainable": n_train, "total": n_total, "wrapped_linear": n_wrapped}
