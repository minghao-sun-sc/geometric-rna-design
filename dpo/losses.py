# dpo/losses.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F


# -------------------------------
# Utilities
# -------------------------------

def _as_logits(out) -> torch.Tensor:
    """
    Some models return (logits, extra), some return logits directly.
    """
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


@torch.no_grad()
def _get_ref_model(ref_manager) -> torch.nn.Module:
    """
    Try a few common attribute names to find the frozen reference model.
    """
    for name in ("ref_model", "reference_model", "model_ref", "model"):
        if hasattr(ref_manager, name):
            return getattr(ref_manager, name)
    raise AttributeError("RefManager does not expose a reference model attribute.")


def _gather_logps_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    logits: [N, V]
    y     : [N] (long)
    returns per-token log-prob: [N]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, y.view(-1, 1)).squeeze(-1)


def _aggregate_per_graph(
    node_values: torch.Tensor,          # [N] (e.g., per-token log p or per-token loss)
    node_mask: Optional[torch.Tensor],  # [N] of {0,1} or None
    batch_index: torch.Tensor,          # [N] int graph ids
    length_norm: bool,                  # normalize by #supervised tokens if True
) -> torch.Tensor:
    """
    Reduce node-wise values to per-graph scalars.
    If length_norm: mean over supervised tokens; else: sum.
    """
    if node_mask is None:
        node_mask = torch.ones_like(node_values, dtype=node_values.dtype)
    else:
        node_mask = node_mask.to(dtype=node_values.dtype)

    Ngraphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0

    # Sum of masked values per graph
    sum_vals = torch.zeros(Ngraphs, dtype=node_values.dtype, device=node_values.device)
    sum_vals.scatter_add_(0, batch_index, node_values * node_mask)

    # Denominator = # supervised tokens per graph
    denom = torch.zeros(Ngraphs, dtype=node_values.dtype, device=node_values.device)
    denom.scatter_add_(0, batch_index, node_mask)

    if length_norm:
        denom = denom.clamp_min(1.0)
        per_graph = sum_vals / denom
    else:
        per_graph = sum_vals  # unnormalized sum

    return per_graph


def _compute_per_graph_logp(
    model: torch.nn.Module,
    data_batch,          # PyG Batch
    y_tokens: torch.Tensor,           # [N]
    node_mask: Optional[torch.Tensor],
    length_norm: bool,
    *,
    no_grad: bool,
) -> torch.Tensor:
    """
    Returns per-graph log p_theta(y|x) as [B] (B = #graphs in batch).
    Sets batch.seq to y_tokens (teacher-forcing for AR model) before forward.
    """
    # Set the sequence that the AR model should condition on
    # (teacher forcing). This is safe since we operate on the batched copy.
    data_batch.seq = y_tokens

    if no_grad:
        with torch.no_grad():
            logits = _as_logits(model(data_batch))
    else:
        logits = _as_logits(model(data_batch))

    # Node-wise log p for the provided tokens
    node_logps = _gather_logps_from_logits(logits, y_tokens)  # [N]
    # Reduce to per-graph values
    per_graph = _aggregate_per_graph(node_logps, node_mask, data_batch.batch, length_norm)
    return per_graph  # [B]


def _nll_sft(
    model: torch.nn.Module,
    data_batch,
    y_tokens: torch.Tensor,
    node_mask: Optional[torch.Tensor],
    length_norm: bool,
) -> torch.Tensor:
    """
    Winner-only SFT: masked CE reduced per-graph then mean over batch.
    Returns a scalar loss (higher = worse).
    """
    data_batch.seq = y_tokens
    logits = _as_logits(model(data_batch))  # [N, V]

    # Per-token CE (negative log-likelihood)
    # NLL per token: -log p(y)
    nll_tok = -_gather_logps_from_logits(logits, y_tokens)  # [N]

    # Reduce per graph
    nll_per_graph = _aggregate_per_graph(nll_tok, node_mask, data_batch.batch, length_norm)  # [B]
    # Mean over graphs
    return nll_per_graph.mean()


# -------------------------------
# Public API: one training/eval step
# -------------------------------

def dpo_sft_step(
    model: torch.nn.Module,
    ref_manager,
    data_batch,                  # PyG Batch (CPU or already on device; trainer moves it)
    y_w: torch.Tensor,           # winner tokens [N]
    y_l: torch.Tensor,           # loser  tokens [N]
    node_mask: Optional[torch.Tensor],  # [N] or None (if None, supervise all tokens)
    weight: Optional[torch.Tensor] = None,  # [B] per-graph weights or None
    *,
    beta: float = 0.1,
    lambda_sft: float = 0.0,
    length_norm: bool = True,
    train: bool = True,
) -> Dict[str, Any]:
    """
    Compute DPO loss (with frozen reference) + optional SFT, return metrics.

    Returns:
      {
        "loss": scalar,
        "loss_dpo": scalar,
        "loss_sft": scalar or None,
        "pref_acc": scalar in [0,1],
      }
    """
    # 1) Policy per-graph logps
    logp_pol_w = _compute_per_graph_logp(model, data_batch, y_w, node_mask, length_norm, no_grad=not train)
    logp_pol_l = _compute_per_graph_logp(model, data_batch, y_l, node_mask, length_norm, no_grad=not train)

    # 2) Reference per-graph logps (no grad)
    ref_model = _get_ref_model(ref_manager)
    logp_ref_w = _compute_per_graph_logp(ref_model, data_batch, y_w, node_mask, length_norm, no_grad=True)
    logp_ref_l = _compute_per_graph_logp(ref_model, data_batch, y_l, node_mask, length_norm, no_grad=True)

    # 3) DPO preference loss
    #    L_dpo = - E[ log σ( β * ( (logπθ(w)-logπθ(l)) - (logπref(w)-logπref(l)) ) ) ]
    margin_pol = logp_pol_w - logp_pol_l
    margin_ref = logp_ref_w - logp_ref_l
    dpo_arg = beta * (margin_pol - margin_ref)
    loss_dpo_vec = -F.logsigmoid(dpo_arg)  # [B]

    # 4) Optional SFT (winner-only CE)
    loss_sft = None
    if lambda_sft and lambda_sft > 0.0:
        loss_sft = _nll_sft(model, data_batch, y_w, node_mask, length_norm)  # scalar

    # 5) Weighting and aggregation over batch
    if weight is not None:
        # weight is per-graph [B]
        weight = weight.to(dtype=loss_dpo_vec.dtype, device=loss_dpo_vec.device)
        wsum = weight.sum().clamp_min(1.0)
        loss_dpo = (loss_dpo_vec * weight).sum() / wsum
    else:
        loss_dpo = loss_dpo_vec.mean()

    loss = loss_dpo + (lambda_sft * loss_sft if loss_sft is not None else 0.0)

    # 6) Metrics
    # Preference accuracy (policy-only): does the policy assign higher prob to the winner?
    pref_acc = (margin_pol > 0).float().mean()

    out = {
        "loss": loss,
        "loss_dpo": loss_dpo,
        "loss_sft": loss_sft,
        "pref_acc": pref_acc,
        # (optional: expose raw means for debugging)
        # "logp_pol_w": logp_pol_w.mean(),
        # "logp_pol_l": logp_pol_l.mean(),
        # "logp_ref_w": logp_ref_w.mean(),
        # "logp_ref_l": logp_ref_l.mean(),
    }
    return out
