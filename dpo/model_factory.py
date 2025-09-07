# dpo/model_factory.py
from __future__ import annotations
import os
import torch
from typing import Dict, Any

# We do NOT modify src/, we just import the classes.
from src.models import AutoregressiveMultiGNNv1, NonAutoregressiveMultiGNNv1


def _to_tuple(x):
    return tuple(x) if isinstance(x, (list, tuple)) else x


def _strip_module_prefix(state_dict: dict) -> dict:
    # Handle DDP checkpoints
    if not state_dict:
        return state_dict
    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _maybe_extract_state_dict(obj: dict) -> dict:
    # Common patterns: {'state_dict': {...}} or direct state dict
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    """
    Build a gRNAde model from cfg without modifying src/.
    Loads weights from cfg['model_path'] if present.
    """
    name = str(cfg.get("model", "ARv1")).upper()

    node_in_dim = _to_tuple(cfg.get("node_in_dim", (15, 4)))
    node_h_dim  = _to_tuple(cfg.get("node_h_dim",  (128, 16)))
    edge_in_dim = _to_tuple(cfg.get("edge_in_dim", (131, 3)))
    edge_h_dim  = _to_tuple(cfg.get("edge_h_dim",  (64, 4)))
    num_layers  = int(cfg.get("num_layers", 4))
    drop_rate   = float(cfg.get("drop_rate", 0.5))
    out_dim     = int(cfg.get("out_dim", 4))

    if name == "ARV1":
        model = AutoregressiveMultiGNNv1(
            node_in_dim=node_in_dim,
            node_h_dim=node_h_dim,
            edge_in_dim=edge_in_dim,
            edge_h_dim=edge_h_dim,
            num_layers=num_layers,
            drop_rate=drop_rate,
            out_dim=out_dim,
        )
    elif name == "NARV1":
        model = NonAutoregressiveMultiGNNv1(
            node_in_dim=node_in_dim,
            node_h_dim=node_h_dim,
            edge_in_dim=edge_in_dim,
            edge_h_dim=edge_h_dim,
            num_layers=num_layers,
            drop_rate=drop_rate,
            out_dim=out_dim,
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Expected ARv1 or NARv1.")

    # Load weights if available
    ckpt_path = cfg.get("model_path", None)
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            raw = torch.load(ckpt_path, map_location="cpu")
            state = _maybe_extract_state_dict(raw)
            state = _strip_module_prefix(state)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[model_factory] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[model_factory] Warning: failed to load checkpoint '{ckpt_path}': {e}")

    return model
