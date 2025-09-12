import os, copy, torch
from src.models import AutoregressiveMultiGNNv1


def build_model_from_cfg(model_cfg):
    return AutoregressiveMultiGNNv1(
        node_in_dim=tuple(model_cfg.node_in_dim),
        node_h_dim=tuple(model_cfg.node_h_dim),
        edge_in_dim=tuple(model_cfg.edge_in_dim),
        edge_h_dim=tuple(model_cfg.edge_h_dim),
        num_layers=model_cfg.num_layers,
        drop_rate=model_cfg.drop_rate,
        out_dim=model_cfg.out_dim,
    )


def smart_load(model, ckpt_path, map_location="cpu"):
    sd = torch.load(ckpt_path, map_location=map_location)
    # allow checkpoints saved with torch.save(model.state_dict()) AND plain dicts
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=True)  # strict=True now that dims match
    return missing, unexpected


def build_policy_and_reference(cfg, device):
    # build two identical nets
    policy = build_model_from_cfg(cfg.model).to(device)
    reference = build_model_from_cfg(cfg.model).to(device)
    # init from base checkpoint
    ckpt = cfg.paths.base_checkpoint
    smart_load(policy, ckpt, map_location="cpu")
    smart_load(reference, ckpt, map_location="cpu")
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    return policy, reference


def build_policy_only(cfg, device):
    """Build only policy model (for SimPO training)."""
    policy = build_model_from_cfg(cfg.model).to(device)
    ckpt = cfg.paths.base_checkpoint
    smart_load(policy, ckpt, map_location="cpu")
    return policy


def save_checkpoint(root, name, model, optimizer, scheduler, step, best_metric, cfg):
    path = os.path.join(root, f"{name}.pt")
    # Handle custom schedulers that don't have state_dict
    scheduler_state = None
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        scheduler_state = scheduler.state_dict()
    
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler_state,
        "step": step,
        "best_metric": best_metric,
        "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)
    # also write a pointer for latest/best
    if name in ["best", "latest"]:
        link_dir = os.path.join(root, "latest" if name=="latest" else "best")
        os.makedirs(os.path.dirname(os.path.join(root, name)), exist_ok=True)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    obj = torch.load(path, map_location=device)
    model.load_state_dict(obj["model"], strict=True)
    if optimizer is not None and obj.get("optimizer") is not None:
        optimizer.load_state_dict(obj["optimizer"])
    if scheduler is not None and obj.get("scheduler") is not None:
        scheduler.load_state_dict(obj["scheduler"])
    return obj.get("step", 0), obj.get("best_metric", None)
