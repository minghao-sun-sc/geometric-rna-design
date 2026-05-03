"""Pareto-DPO Stage-2: weight-conditioned policy via FiLM.

Wraps `AutoregressiveMultiGNNv1` (gRNAde backbone) with a small FiLM head that
conditions the scalar features of the encoder embeddings on a scalarization
weight w ∈ Δ^{M-1}. M = 3 by default (pLDDT / RMSD / MFE axes).

Design intent (see appendix `app:hetero_derivation` and `app:trust_region` of
manuscript/RiboPO_revision):
- Train: sample w ~ Dirichlet(α=1) per batch; the BTL loss applied to a Pareto-
  dominant pair set is consistent with every w ⪰ 0, so the policy learns a
  Pareto-monotonic implicit reward.
- Inference: query π_θ(s | G, w) for any w to traverse the achievable Pareto
  front (e.g., w = (1,0,0) emphasises pLDDT, w = (0,0,1) emphasises MFE).

This is the methodological centerpiece of the ICLR 2027 reframe.
Do **not** import `src/models.py`-level changes here; we wrap, not modify.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import AutoregressiveMultiGNNv1


class FiLMHead(nn.Module):
    """Per-dimension γ, β from a low-dim scalarization weight w."""

    def __init__(self, w_dim: int, h_dim: int, init_residual: bool = True):
        super().__init__()
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.gamma = nn.Linear(w_dim, h_dim, bias=True)
        self.beta  = nn.Linear(w_dim, h_dim, bias=True)
        if init_residual:
            # init γ→0, β→0 so wrapped model = base model at init (residual safety)
            nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, h_scalar: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """h_scalar : [n_nodes, h_dim] ; w : [w_dim]; broadcasts w over nodes."""
        gamma = self.gamma(w).unsqueeze(0)   # [1, h_dim]
        beta  = self.beta(w).unsqueeze(0)    # [1, h_dim]
        return h_scalar * (1.0 + gamma) + beta  # residual: w=0 → identity at init


class WeightConditionedAutoregressiveGNN(nn.Module):
    """Wraps `AutoregressiveMultiGNNv1` with a FiLM head on scalar encoder features.

    The forward pass takes an additional `w` argument; if w is None, the wrapper
    is a no-op (identity for scalar features), reproducing the base model exactly.

    Loading from a base checkpoint:
        wrapped = WeightConditionedAutoregressiveGNN(base_kwargs={...})
        wrapped.load_base_state_dict(torch.load("gRNAde_ARv1_1state_das.h5"))
        # FiLM head is zero-initialised so wrapped(g) ≡ base(g) at init.
    """

    def __init__(self, base_kwargs: dict, w_dim: int = 3, film_init_residual: bool = True):
        super().__init__()
        self.w_dim = w_dim
        self.base = AutoregressiveMultiGNNv1(**base_kwargs)
        h_scalar = base_kwargs.get("node_h_dim", (128, 16))[0]
        self.film = FiLMHead(w_dim=w_dim, h_dim=h_scalar, init_residual=film_init_residual)

    # --- weight-injection helpers ---

    def _modulate(self, encoder_embeddings, w):
        """Apply FiLM to scalar features only (vector features unchanged)."""
        h_s, h_v = encoder_embeddings
        h_s = self.film(h_s, w)
        return (h_s, h_v)

    # --- forward (mirrors `AutoregressiveMultiGNNv1.forward`) ---

    def _default_uniform_w(self, device, dtype):
        """The simplex centroid w = (1/M, ..., 1/M) — the canonical operating point."""
        return torch.full((self.w_dim,), 1.0 / self.w_dim, device=device, dtype=dtype)

    def forward(self, batch, w: Optional[torch.Tensor] = None):
        b = self.base
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        seq = batch.seq

        h_V = b.W_v(h_V)
        h_E = b.W_e(h_E)
        for layer in b.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)

        h_V, h_E = b.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)

        encoder_embeddings = h_V

        # Default w to the simplex centroid (1/M, ..., 1/M) when not provided.
        # This makes evaluation without an explicit w correspond to the canonical
        # uniform-weight Pareto operating point, NOT to a "FiLM-stripped" policy.
        # During training the trainer always passes an explicit w (sampled from
        # Dirichlet); this default is only relevant for inference / eval.
        if w is None:
            w = self._default_uniform_w(encoder_embeddings[0].device, encoder_embeddings[0].dtype)
        else:
            w = w.to(encoder_embeddings[0].device).to(encoder_embeddings[0].dtype)
        encoder_embeddings = self._modulate(encoder_embeddings, w)

        h_S = b.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])

        h_V = encoder_embeddings
        for layer in b.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)

        logits = b.W_out(h_V)
        return logits

    @torch.no_grad()
    def sample(self, batch, n_samples, temperature: float = 0.1, w: Optional[torch.Tensor] = None,
               logit_bias=None, return_logits: bool = False):
        """Sample at a given scalarization weight; defaults to the simplex centroid.

        Implementation: monkey-patch the base model's pool_multi_conf to apply FiLM
        on the way out, then delegate to base.sample. Restored on exit. This avoids
        having to re-implement the entire AR sampler and works for both w=None
        (defaults to uniform centroid via FiLM) and any specific w on the simplex.
        """
        # Resolve w to a concrete tensor (default to centroid for inference parity
        # with `forward`).
        device = next(self.base.parameters()).device
        dtype = next(self.base.parameters()).dtype
        if w is None:
            w = self._default_uniform_w(device, dtype)
        else:
            w = w.to(device).to(dtype)

        orig = self.base.pool_multi_conf

        def patched(h_V, h_E, mask_confs, edge_index):
            h_V, h_E = orig(h_V, h_E, mask_confs, edge_index)
            h_V = self._modulate(h_V, w)
            return h_V, h_E

        self.base.pool_multi_conf = patched
        try:
            return self.base.sample(batch, n_samples, temperature=temperature,
                                    logit_bias=logit_bias, return_logits=return_logits)
        finally:
            self.base.pool_multi_conf = orig

    # --- checkpoint compatibility ---

    def load_base_state_dict(self, state: dict, strict: bool = True):
        """Load a vanilla AutoregressiveMultiGNNv1 checkpoint into the base model.

        Keeps FiLM weights at residual init (γ=β=0), so the wrapper is initially
        equivalent to the base.
        """
        return self.base.load_state_dict(state, strict=strict)


def pareto_stage2_step_losses(
    policy_w,            # WeightConditionedAutoregressiveGNN
    ref_model,           # base AutoregressiveMultiGNNv1 (NOT weight-conditioned)
    batch,
    w: torch.Tensor,     # scalarization weight, shape [w_dim]
    beta: float = 0.12,
    max_len: Optional[int] = None,
):
    """Stage-2 DPO step: policy is weight-conditioned, reference is the frozen base.

    The implicit reward is r_θ(s|G,w) = β · log(π_θ(s|G,w) / π_ref(s|G)). Sampling w
    per batch exposes the BTL likelihood to many scalarizations; because the
    preference set is ε-Pareto-dominant, every w ⪰ 0 is a consistent target,
    so the policy learns a Pareto-monotonic implicit reward over the cone.

    Returns the same dict shape as `dpo_step_losses` for trainer compatibility.
    """
    from dpo.losses import seq_logprob
    # Policy: pass w through forward
    lpw = seq_logprob(policy_w, batch.graph, batch.winner_seq, max_len=max_len, w=w)
    lpl = seq_logprob(policy_w, batch.graph, batch.loser_seq,  max_len=max_len, w=w)
    # Reference: standard base model, no w
    with torch.no_grad():
        lrw = seq_logprob(ref_model, batch.graph, batch.winner_seq, max_len=max_len)
        lrl = seq_logprob(ref_model, batch.graph, batch.loser_seq,  max_len=max_len)

    pi_diff  = lpw - lpl
    ref_diff = lrw - lrl
    loss = -F.logsigmoid(beta * (pi_diff - ref_diff))
    if loss.dim() > 0:
        loss = loss.mean()
    pref_acc = (pi_diff > 0).float().mean()
    margin = pi_diff.detach().mean() if pi_diff.dim() > 0 else pi_diff.detach()
    return {
        "loss_pareto2":  loss,
        "loss_dpo":      loss,         # alias for trainer meter compatibility
        "pref_acc":      pref_acc,
        "margin":        margin,
        "w":             w.detach(),
        "w_max":         torch.tensor(float(w.max()), device=lpw.device),
    }


def sample_dirichlet_w(batch_size: int, w_dim: int = 3, alpha: float = 1.0,
                       device: Optional[torch.device] = None) -> torch.Tensor:
    """Sample one weight vector per batch (or per-pair) from Dirichlet(alpha).

    Returns shape [w_dim] when batch_size==1 (default scalar weight per batch),
    or [batch_size, w_dim] for per-example weighting.
    """
    a = torch.full((w_dim,), alpha, dtype=torch.float32, device=device)
    if batch_size == 1:
        return torch.distributions.Dirichlet(a).sample()
    return torch.distributions.Dirichlet(a).sample((batch_size,))
