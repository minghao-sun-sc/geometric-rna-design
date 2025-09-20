# multiround/ema.py
from __future__ import annotations
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, Iterable, Optional
import torch
import torch.nn as nn

def _unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m

@dataclass
class EMAConfig:
    decay: float = 0.999
    warmup_updates: int = 0          # optional warmup for decay scheduling
    device: Optional[str] = None     # None = same device as model; "cpu" to offload
    include_buffers: bool = False    # rarely needed; usually False
    skip_keys: Iterable[str] = ()    # substrings to skip (e.g., ["bias", "bn"])

class ModelEMA:
    """
    Exponential Moving Average of model parameters (float32 shadows).
    Call `update(model)` AFTER each optimizer step.
    """
    def __init__(self, model: nn.Module, cfg: EMAConfig):
        self.cfg = cfg
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}  # for temporary swap

        model = _unwrap_model(model)
        device = cfg.device
        for n, p in model.named_parameters():
            if (not p.requires_grad) or (not p.dtype.is_floating_point):
                continue
            if any(s in n for s in cfg.skip_keys):
                continue
            t = p.detach().float().clone()
            if device is not None:
                t = t.to(device)
            self.shadow[n] = t

        if cfg.include_buffers:
            for n, b in model.named_buffers():
                if b.dtype.is_floating_point:
                    t = b.detach().float().clone()
                    if device is not None:
                        t = t.to(device)
                    self.shadow[n] = t

    def _decay(self) -> float:
        # Optional warmup: ramps the *effective* decay up with updates
        if self.cfg.warmup_updates and self.num_updates < self.cfg.warmup_updates:
            # keep EMA closer to current weights during early steps
            warm = float(self.num_updates) / float(self.cfg.warmup_updates)
            return 1.0 - (1.0 - self.cfg.decay) * warm
        return self.cfg.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Call after each optimizer.step()."""
        self.num_updates += 1
        d = self._decay()
        model = _unwrap_model(model)
        for n, p in model.named_parameters():
            if n not in self.shadow:
                continue
            if (not p.requires_grad) or (not p.dtype.is_floating_point):
                continue
            s = self.shadow[n]
            # cast to float32 for numerically stable EMA
            s.mul_(d).add_(p.detach().float().to(s.device), alpha=(1.0 - d))

        if self.cfg.include_buffers:
            for n, b in model.named_buffers():
                if n in self.shadow and b.dtype.is_floating_point:
                    s = self.shadow[n]
                    s.mul_(d).add_(b.detach().float().to(s.device), alpha=(1.0 - d))

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite `model` params with EMA weights (in-place)."""
        model = _unwrap_model(model)
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.device, dtype=p.dtype))
        if self.cfg.include_buffers:
            for n, b in model.named_buffers():
                if n in self.shadow:
                    b.data.copy_(self.shadow[n].to(b.device, dtype=b.dtype))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Save current model params so we can restore after temporary EMA swap."""
        self.backup.clear()
        model = _unwrap_model(model)
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.detach().clone()
        if self.cfg.include_buffers:
            for n, b in model.named_buffers():
                if n in self.shadow:
                    self.backup[n] = b.data.detach().clone()

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore params saved by `store()`."""
        model = _unwrap_model(model)
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        if self.cfg.include_buffers:
            for n, b in model.named_buffers():
                if n in self.backup:
                    b.data.copy_(self.backup[n])
        self.backup.clear()

    @contextmanager
    def apply_to(self, model: nn.Module):
        """
        Context manager for "evaluate with EMA":
            with ema.apply_to(model):
                evaluate(...)
        """
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> Dict:
        return {
            "decay": self.cfg.decay,
            "warmup_updates": self.cfg.warmup_updates,
            "num_updates": self.num_updates,
            "device": self.cfg.device,
            "include_buffers": self.cfg.include_buffers,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict) -> None:
        self.num_updates = int(state.get("num_updates", 0))
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
