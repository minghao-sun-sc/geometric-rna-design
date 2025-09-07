# dpo/lightning_module.py

from __future__ import annotations

import copy
import importlib
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl

from dpo.losses import dpo_sft_step
from dpo.lora import apply_lora


# ---------------------- small utils ---------------------- #

def _omega_to_container(x: Any) -> Any:
    """Safely convert OmegaConf containers to plain python containers."""
    try:
        from omegaconf import OmegaConf, DictConfig, ListConfig  # type: ignore
        if isinstance(x, (DictConfig, ListConfig)):
            return OmegaConf.to_container(x, resolve=True)
    except Exception:
        pass
    return x


def _to_dict(x: Any) -> Dict[str, Any]:
    """Best-effort: turn config-like objects into plain dicts."""
    x = _omega_to_container(x)
    if x is None:
        return {}
    if isinstance(x, dict):
        # Also convert nested OmegaConf containers
        return {k: _omega_to_container(v) for k, v in x.items()}
    if is_dataclass(x):
        return asdict(x)
    # Some structured configs present attributes (dataclasses, SimpleNamespace, etc.)
    if hasattr(x, "__dict__") and not isinstance(x, str):
        try:
            d = vars(x)
            # ensure nested OmegaConf nodes are resolved
            return {k: _omega_to_container(v) for k, v in d.items()}
        except Exception:
            pass
    # If it's already a string/primitive, don't wrap it into {"value": x} here.
    if isinstance(x, (str, int, float, bool)):
        return {"__primitive__": x}
    try:
        return dict(x)
    except Exception:
        return {"__primitive__": x}


def _import_from_string(target: str):
    """Hydra-style 'package.module:Class' or 'package.module.Class' importer."""
    if ":" in target:
        module_path, cls_name = target.split(":")
    else:
        last_dot = target.rfind(".")
        if last_dot < 0:
            raise ValueError(f"Import target '{target}' must be 'pkg.mod:Class' or 'pkg.mod.Class'")
        module_path, cls_name = target[:last_dot], target[last_dot + 1 :]
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()


def _count_params(m: nn.Module) -> Tuple[int, int]:
    n_total = sum(p.numel() for p in m.parameters())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return n_train, n_total


def _unwrap_batch(batch: Any) -> Any:
    # Lightning + some DataLoaders may wrap batches as tuples/lists. Keep the first element.
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        print("[batch] Unwrapped a list/tuple batch to a single batch object.")
        return batch[0]
    return batch


def _pluck(d: Dict[str, Any], names: Tuple[str, ...]) -> Any:
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return None


# ---------------------- config discovery helpers ---------------------- #

def _find_model_cfgs(cfg: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """
    Return (policy_cfg, ref_cfg, ref_ckpt_path) from a flexible set of locations.
    Accepts aliases and nested dictionaries like cfg['models']['policy'] etc.
    """
    # direct top-level first
    policy = _pluck(cfg, ("model", "policy", "policy_model", "student", "policy_cfg"))
    ref = _pluck(cfg, ("ref_model", "reference_model", "reference", "ref", "teacher", "baseline", "ref_cfg"))

    # nested containers
    for container_key in ("models", "modules", "networks", "arch", "architecture"):
        container = cfg.get(container_key)
        if isinstance(container, dict):
            policy = policy or _pluck(container, ("policy", "model", "policy_model", "student"))
            ref = ref or _pluck(container, ("ref_model", "reference", "reference_model", "ref", "teacher", "baseline"))

    # optional checkpoint path for ref
    ref_ckpt = _pluck(
        cfg,
        ("ref_ckpt", "ref_checkpoint", "reference_ckpt", "reference_checkpoint", "ref_path"),
    )
    if ref_ckpt is None:
        ckpt_section = cfg.get("checkpoints") or cfg.get("checkpoint") or {}
        if isinstance(ckpt_section, dict):
            ref_ckpt = _pluck(
                ckpt_section,
                ("ref_ckpt", "ref_checkpoint", "reference_ckpt", "reference_checkpoint", "ref_path"),
            )

    return policy, ref, ref_ckpt


def _guess_params_for(root_cfg: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Find a params dict for 'policy' or 'ref' in common places."""
    if role == "policy":
        names = ("model_params", "policy_params", "student_params", "policy_kwargs", "params")
        nested_names = ("policy_params", "model_params", "student_params", "policy_kwargs", "params")
    else:
        names = ("ref_params", "reference_params", "teacher_params", "ref_kwargs", "params")
        nested_names = ("ref_params", "reference_params", "teacher_params", "ref_kwargs", "params")

    # top-level
    for n in names:
        val = root_cfg.get(n)
        if isinstance(val, dict):
            return _to_dict(val)

    # nested containers
    for container_key in ("models", "modules", "networks", "arch", "architecture"):
        cont = root_cfg.get(container_key)
        if isinstance(cont, dict):
            for n in nested_names:
                val = cont.get(n)
                if isinstance(val, dict):
                    return _to_dict(val)

    return {}


def _maybe_instantiate(model_cfg: Any, *, root_cfg: Dict[str, Any], role: str) -> nn.Module:
    """
    Instantiate a model from common config patterns:
      - already a torch.nn.Module   -> returned as-is
      - STRING: 'package.module:Class' or 'package.module.Class' (+ guessed params)
      - {'_target_': '<pkg.Class>', ...}                    (Hydra)
      - {'target': '<pkg.Class>', 'params': {...}}          (custom)
      - {'class_path'|'cls'|'model_class': '<pkg.Class>', ...}
    """
    if isinstance(model_cfg, nn.Module):
        return model_cfg

    # If OmegaConf/structured -> dict; tolerate primitives
    cfg = _to_dict(model_cfg)

    # STRING form (stored under "__primitive__")
    if "__primitive__" in cfg and isinstance(cfg["__primitive__"], str):
        target = cfg["__primitive__"]
        params = _guess_params_for(root_cfg, role)
        cls = _import_from_string(target)
        try:
            return cls(**params)
        except TypeError as e:
            raise ValueError(
                f"Failed to instantiate {target} with params {list(params.keys())}. "
                f"Please provide a params dict (e.g., '{role}_params:' or 'model_params:'). "
                f"Original error: {e}"
            )

    # Hydra style
    if "_target_" in cfg:
        target = cfg["_target_"]
        params = {k: v for k, v in cfg.items() if k != "_target_"}
        cls = _import_from_string(target)
        return cls(**params)

    # Custom 'target' + 'params'
    if "target" in cfg:
        target = cfg["target"]
        params = cfg.get("params", {})
        if not isinstance(params, dict):
            params = _to_dict(params)
        cls = _import_from_string(target)
        return cls(**params)

    # Alternative key spellings
    for key in ("class_path", "cls", "model_class"):
        if key in cfg:
            cls = _import_from_string(cfg[key])
            kwargs = {k: v for k, v in cfg.items() if k not in (key,)}
            return cls(**kwargs)

    # Last resort: if we only got a primitive that isn't a string
    if "__primitive__" in cfg:
        raise ValueError(
            f"Unsupported primitive for {role} model config: {cfg['__primitive__']!r}. "
            f"Provide a string import path or a dict with '_target_' or 'target'."
        )

    raise ValueError(
        "Unsupported model config format. Expected a torch.nn.Module instance, a string "
        "('pkg.mod:Class'), a dict with '_target_' (Hydra), or 'target'/'params'. "
        f"Got keys: {list(cfg.keys())}"
    )


def _load_state_dict_loosely(model: nn.Module, sd: Dict[str, torch.Tensor], label: str = "ref") -> None:
    """Load a state dict while stripping common prefixes and being tolerant to mismatches."""
    state = sd.get("state_dict", sd)
    cleaned = {}
    for k, v in state.items():
        k2 = k
        for pref in ("model.", "module.", "ref_model."):
            if k2.startswith(pref):
                k2 = k2[len(pref) :]
        cleaned[k2] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[{label}] loaded state_dict: missing={len(missing)}, unexpected={len(unexpected)}")


# ---------------------- Lightning module ---------------------- #

class DpoLightningModule(pl.LightningModule):
    """
    Lightning wrapper for policy/ref models trained via the combined DPO+SFT step.

    Flexible config:
      - Top-level keys: 'model' and 'ref_model' (preferred), or aliases:
        'policy', 'policy_model', 'reference', 'reference_model', 'ref', 'teacher', etc.
      - Nested under: 'models' / 'modules' / 'networks' also supported.
      - Model specs can be strings or dicts (Hydra/custom).
      - If no explicit 'ref_model' is provided, we clone the policy *before LoRA* and
        optionally load weights from 'ref_ckpt' / 'reference_ckpt' / 'ref_checkpoint'.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        cfg = _to_dict(config)

        # ---------------- build policy + ref ---------------- #
        policy_cfg, ref_cfg, ref_ckpt = _find_model_cfgs(cfg)

        if policy_cfg is None and ref_cfg is None:
            # Give a clearer error with visible keys
            raise ValueError(
                "Could not locate model configs. Looked for keys like "
                "'model', 'policy', 'ref_model', 'reference', possibly under "
                "'models'/'modules'/'networks'. Top-level keys found: "
                f"{list(cfg.keys())}"
            )

        # Instantiate policy first (base, no LoRA yet)
        if policy_cfg is None and ref_cfg is not None:
            policy_cfg = ref_cfg

        self.model: nn.Module = _maybe_instantiate(policy_cfg, root_cfg=cfg, role="policy")

        # Build reference:
        if ref_cfg is not None:
            self.ref_model: nn.Module = _maybe_instantiate(ref_cfg, root_cfg=cfg, role="ref")
        else:
            # No explicit ref config: clone policy BEFORE LoRA
            self.ref_model = copy.deepcopy(self.model)
            print("[ref] No explicit ref_model in config; cloned policy as reference.")

        # Optionally load a ref checkpoint
        if ref_ckpt:
            try:
                ckpt = torch.load(ref_ckpt, map_location="cpu")
                _load_state_dict_loosely(self.ref_model, ckpt, label="ref")
            except Exception as e:
                print(f"[ref] WARNING: failed to load ref checkpoint '{ref_ckpt}': {e}")

        # Freeze reference; used only under no_grad inside loss
        _freeze_module(self.ref_model)

        # Apply LoRA to policy only (after ref is built/frozen)
        lora_cfg = cfg.get("lora") or cfg.get("policy_lora")
        if lora_cfg:
            lora_cfg = _to_dict(lora_cfg)
            # Check if LoRA is enabled (default to True if not specified)
            if lora_cfg.get("enabled", True):
                # Remove 'enabled' key before passing to apply_lora
                lora_params = {k: v for k, v in lora_cfg.items() if k != "enabled"}
                apply_lora(self.model, **lora_params)

        # ---------------- hyperparams ---------------- #
        loss_cfg = cfg.get("loss", {})
        if not isinstance(loss_cfg, dict):
            loss_cfg = _to_dict(loss_cfg)

        self.beta: float = (
            cfg.get("beta")
            or loss_cfg.get("beta")
            or 0.1
        )

        # Accept legacy names lam_sft / sft_weight but normalize to lambda_sft
        self.lambda_sft: float = (
            cfg.get("lambda_sft")
            or loss_cfg.get("lambda_sft")
            or cfg.get("lam_sft")
            or loss_cfg.get("lam_sft")
            or cfg.get("sft_weight")
            or loss_cfg.get("sft_weight")
            or 0.0
        )

        self.length_norm: bool = bool(
            cfg.get("length_norm", loss_cfg.get("length_norm", True))
        )

        # Print trainable param counts (policy only)
        n_train, n_total = _count_params(self.model)
        print(f"[init] Trainable params (policy): {n_train:,} / {n_total:,}")

        # Optimizer config
        opt_cfg = cfg.get("optimizer", {})
        if not isinstance(opt_cfg, dict):
            opt_cfg = _to_dict(opt_cfg)
        self._lr: float = float(opt_cfg.get("lr", cfg.get("lr", 1e-4)))
        self._weight_decay: float = float(opt_cfg.get("weight_decay", cfg.get("weight_decay", 0.0)))
        betas = opt_cfg.get("betas", cfg.get("betas", (0.9, 0.999)))
        self._betas = tuple(betas) if isinstance(betas, (list, tuple)) else (0.9, 0.999)
        self._eps: float = float(opt_cfg.get("eps", cfg.get("eps", 1e-8)))

    # ---------------- Lightning plumbing ---------------- #

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optim = AdamW(params, lr=self._lr, weight_decay=self._weight_decay, betas=self._betas, eps=self._eps)
        return optim

    def forward(self, batch: Any) -> Any:
        return self.model(batch)

    # ---------------- Train / Val shared step ---------------- #

    def _shared_step(self, batch: Any, batch_idx: int, *, train: bool) -> Dict[str, torch.Tensor]:
        batch = _unwrap_batch(batch)

        # Exact signature from your printout:
        # dpo_sft_step(model, data_batch, *, beta, lambda_sft, length_norm, ref_model=None, train=True)
        out = dpo_sft_step(
            self.model,
            batch,
            beta=self.beta,
            lambda_sft=self.lambda_sft,
            length_norm=self.length_norm,
            ref_model=self.ref_model,
            train=bool(train),
        )

        # Log aux metrics
        to_log = {k: v for k, v in out.items() if k != "loss"}
        if to_log:
            self.log_dict(
                to_log,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return out

    # ---------------- Lightning hooks ---------------- #

    def training_step(self, batch: Any, batch_idx: int):
        out = self._shared_step(batch, batch_idx, train=True)
        return out["loss"]

    def validation_step(self, batch: Any, batch_idx: int):
        out = self._shared_step(batch, batch_idx, train=False)
        return out["loss"]
