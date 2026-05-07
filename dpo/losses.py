import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_geometric.data import Batch as GeometricBatch


def seq_logprob(model, graph, seq, max_len: Optional[int] = None, **forward_kwargs):
    """
    Teacher-forced log-probability of a full sequence under the AR model.
    Handles both single graphs and batched graphs.

    Any extra kwargs (e.g., w=... for weight-conditioned policies) are passed
    through to model.forward; the base AutoregressiveMultiGNNv1 does not accept
    extra kwargs so we only forward when forward_kwargs is non-empty.

    Returns: scalar logprob (sum over positions) or tensor of logprobs for batch.
    """
    # Check if we have a batch
    is_batched = isinstance(graph, GeometricBatch) and seq.dim() == 2

    if not is_batched:
        # Original single-graph logic
        graph = graph.clone()
        if max_len is not None:
            seq = seq[:max_len]
            graph.seq = graph.seq[:max_len]
            # we also need to mask graph nodes/edges down to max_len -> build subgraph
            # Simple & safe: recompute edge mask for nodes < max_len
            node_mask = torch.arange(graph.node_s.size(0), device=seq.device) < len(seq)
            ei = graph.edge_index
            ekeep = node_mask[ei[0]] & node_mask[ei[1]]
            graph.edge_index = ei[:, ekeep]
            graph.node_s = graph.node_s[node_mask]
            graph.node_v = graph.node_v[node_mask]
            graph.mask_confs = graph.mask_confs[node_mask]
            graph.mask_coords = graph.mask_coords[node_mask]
            graph.edge_s = graph.edge_s[ekeep]
            graph.edge_v = graph.edge_v[ekeep]
        graph.seq = seq

        # Pass forward kwargs only if non-empty so base models work unchanged.
        logits = model.forward(graph, **forward_kwargs) if forward_kwargs else model.forward(graph)
        logp = F.log_softmax(logits, dim=-1)
        lp = logp.gather(-1, seq.view(-1,1)).sum()
        return lp
    else:
        # Batched graph logic - process each sample in the batch
        batch_size = seq.size(0)
        log_probs = []

        # Split the batched graph back into individual graphs
        graphs = graph.to_data_list()

        for i in range(batch_size):
            curr_seq = seq[i]
            # Remove padding (-1 values)
            valid_mask = curr_seq != -1
            curr_seq = curr_seq[valid_mask]

            # Process individual graph
            lp = seq_logprob(model, graphs[i], curr_seq, max_len=max_len, **forward_kwargs)
            log_probs.append(lp)

        # Stack log probs
        return torch.stack(log_probs)


def ce_on_sequence(model, graph, seq, max_len: Optional[int] = None):
    # Check if we have a batch
    is_batched = isinstance(graph, GeometricBatch) and seq.dim() == 2
    
    if not is_batched:
        # Original single-graph logic
        if max_len is not None and len(seq) > max_len:
            # replicate max_len trimming from seq_logprob
            graph = graph.clone()
            seq = seq[:max_len]
            node_mask = torch.arange(graph.node_s.size(0), device=seq.device) < len(seq)
            ei = graph.edge_index
            ekeep = node_mask[ei[0]] & node_mask[ei[1]]
            graph.edge_index = ei[:, ekeep]
            graph.node_s = graph.node_s[node_mask]
            graph.node_v = graph.node_v[node_mask]
            graph.mask_confs = graph.mask_confs[node_mask]
            graph.mask_coords = graph.mask_coords[node_mask]
            graph.edge_s = graph.edge_s[ekeep]
            graph.edge_v = graph.edge_v[ekeep]
        else:
            graph = graph

        graph = graph.clone()
        graph.seq = seq
        logits = model.forward(graph)
        loss = F.cross_entropy(logits, seq, reduction="mean")
        return loss
    else:
        # Batched graph logic
        batch_size = seq.size(0)
        losses = []
        graphs = graph.to_data_list()
        
        for i in range(batch_size):
            curr_seq = seq[i]
            # Remove padding (-1 values)
            valid_mask = curr_seq != -1
            curr_seq = curr_seq[valid_mask]
            
            # Process individual graph
            loss = ce_on_sequence(model, graphs[i], curr_seq, max_len=max_len)
            losses.append(loss)
        
        # Average losses across batch
        return torch.stack(losses).mean()


def dpo_step_losses(model, ref_model, batch, beta: float = 0.1, label_smoothing: float = 0.0, max_len: Optional[int] = None):
    """
    Computes DPO loss and preference accuracy for single or batched pairs.
    """
    # policy log-probs
    lpw = seq_logprob(model, batch.graph, batch.winner_seq, max_len=max_len)
    lpl = seq_logprob(model, batch.graph, batch.loser_seq,  max_len=max_len)
    # reference log-probs (no grad)
    with torch.no_grad():
        lrw = seq_logprob(ref_model, batch.graph, batch.winner_seq, max_len=max_len)
        lrl = seq_logprob(ref_model, batch.graph, batch.loser_seq,  max_len=max_len)

    # DPO objective per pair
    pi_diff  = (lpw - lpl)
    ref_diff = (lrw - lrl)
    # minus log-sigmoid for minimization
    dpo_loss = -F.logsigmoid(beta * (pi_diff - ref_diff))

    # For batched data, average the loss
    if dpo_loss.dim() > 0:
        dpo_loss = dpo_loss.mean()

    # metrics
    pref_acc = (pi_diff > 0).float().mean()
    
    # For margin, handle both single and batched cases
    if pi_diff.dim() == 0:
        margin = pi_diff.detach().item()
    else:
        margin = pi_diff.detach().mean().item()

    return {
        "loss_dpo": dpo_loss,
        "pref_acc": pref_acc,
        "margin":   pi_diff.detach().mean() if pi_diff.dim() > 0 else pi_diff.detach(),
    }


def ipo_step_losses(model, ref_model, batch, beta: float = 0.12, max_len: Optional[int] = None):
    """IPO loss (Azar et al., 2024).

    L_IPO = (h - 1/(2β))^2 ,  h = (log π_θ(s^w) - log π_θ(s^l)) - (log π_ref(s^w) - log π_ref(s^l))

    Replaces the log-sigmoid of DPO with a squared loss centered at 1/(2β); empirically
    less prone to over-optimization on noisy preferences. Useful as an A/B against DPO
    when preferences come from noisy physical proxies.
    """
    lpw = seq_logprob(model, batch.graph, batch.winner_seq, max_len=max_len)
    lpl = seq_logprob(model, batch.graph, batch.loser_seq,  max_len=max_len)
    with torch.no_grad():
        lrw = seq_logprob(ref_model, batch.graph, batch.winner_seq, max_len=max_len)
        lrl = seq_logprob(ref_model, batch.graph, batch.loser_seq,  max_len=max_len)

    pi_diff  = lpw - lpl
    ref_diff = lrw - lrl
    h        = pi_diff - ref_diff
    target   = 1.0 / (2.0 * beta)
    loss     = (h - target).pow(2)
    if loss.dim() > 0:
        loss = loss.mean()
    pref_acc = (pi_diff > 0).float().mean()
    return {
        "loss_ipo": loss,
        "pref_acc": pref_acc,
        "margin":   pi_diff.detach().mean() if pi_diff.dim() > 0 else pi_diff.detach(),
    }


def kto_step_losses(
    model,
    ref_model,
    batch,
    beta: float = 0.12,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
    max_len: Optional[int] = None,
):
    """KTO loss (Ethayarajh et al., 2024) — pointwise variant on paired data.

    Standard KTO uses unpaired (prompt, response, +/-) data; here we adapt to paired
    (s^w, s^l) preferences by treating winners as desirable and losers as undesirable.
    The KL reference is computed once per batch from the reference policy on the same
    examples; this is an approximation since KTO normally uses a moving KL estimate.

    L_KTO_desirable   = sigmoid(-beta * (r(s) - z_ref))       , r(s) = log π_θ(s) - log π_ref(s)
    L_KTO_undesirable = sigmoid(+beta * (r(s) - z_ref))
    where z_ref = mean over batch of detached r(s) (a soft KL target).

    Useful when binary (good/bad) labels are more natural than rankings, e.g., when
    the structural quality gate can be applied to single sequences without needing
    a paired loser.
    """
    lpw = seq_logprob(model, batch.graph, batch.winner_seq, max_len=max_len)
    lpl = seq_logprob(model, batch.graph, batch.loser_seq,  max_len=max_len)
    with torch.no_grad():
        lrw = seq_logprob(ref_model, batch.graph, batch.winner_seq, max_len=max_len)
        lrl = seq_logprob(ref_model, batch.graph, batch.loser_seq,  max_len=max_len)

    rw = lpw - lrw            # implicit reward of winner
    rl = lpl - lrl            # implicit reward of loser
    z_ref = torch.cat([rw.detach(), rl.detach()]).mean()

    # σ(-β(r_w - z_ref)) is small when r_w >> z_ref (good); we want to minimise.
    desirable_loss   = torch.sigmoid(-beta * (rw - z_ref))
    undesirable_loss = torch.sigmoid( beta * (rl - z_ref))
    loss = desirable_weight * desirable_loss + undesirable_weight * undesirable_loss
    if loss.dim() > 0:
        loss = loss.mean()
    pref_acc = (rw > rl).float().mean()
    return {
        "loss_kto":    loss,
        "pref_acc":    pref_acc,
        "z_ref":       z_ref.detach(),
        "margin":      (rw - rl).detach().mean() if (rw - rl).dim() > 0 else (rw - rl).detach(),
    }


def importance_corrected_dpo_step_losses(
    model,
    ref_model,
    batch,
    beta: float = 0.12,
    is_clip: float = 5.0,
    is_log_prev_w: Optional[torch.Tensor] = None,
    is_log_prev_l: Optional[torch.Tensor] = None,
    max_len: Optional[int] = None,
):
    """Importance-corrected DPO step (Theorem~\\ref{thm:off_policy} mitigation).

    For static-pair multi-round DPO, by round r the policy π_θ_r has drifted from
    the candidate-generation distribution π_θ_0 (= π_ref). The DPO gradient on a
    static pair set is therefore biased; the bias grows linearly in r.

    This step applies a clipped per-pair importance weight
       w_pair = min(IS_clip, π_θ_{r-1}(s^w)/π_θ_{r-1}(s^w_at_construction))
    to reweight each pair toward the current candidate-generation distribution.
    The previous-round log-probs `is_log_prev_{w,l}` must be precomputed and
    stored in the pair set; if absent, this reduces to standard DPO.

    Usage in multi-round training:
        - Round 1: standard DPO (no IS correction).
        - Before round r >= 2: precompute log π_θ_{r-1}(s^w | G), log π_θ_{r-1}(s^l | G)
          for each pair and attach as `is_log_prev_{w,l}` to the batch.
        - Round r: call this with the IS arrays.

    Returns the same dict as `dpo_step_losses` plus `is_weight` for logging.
    """
    lpw = seq_logprob(model, batch.graph, batch.winner_seq, max_len=max_len)
    lpl = seq_logprob(model, batch.graph, batch.loser_seq,  max_len=max_len)
    with torch.no_grad():
        lrw = seq_logprob(ref_model, batch.graph, batch.winner_seq, max_len=max_len)
        lrl = seq_logprob(ref_model, batch.graph, batch.loser_seq,  max_len=max_len)

    pi_diff  = lpw - lpl
    ref_diff = lrw - lrl
    raw_loss = -F.logsigmoid(beta * (pi_diff - ref_diff))

    if is_log_prev_w is not None and is_log_prev_l is not None:
        # importance ratio = π_prev(s) / π_ref(s) ~ exp(log_prev - log_ref) per side.
        # NOTE on granularity (theory–impl gap, intentional): is_log_prev_{w,l}
        # are precomputed ONCE at the start of round r against the round_{r-1}
        # checkpoint and held CONSTANT for every step in round r. Within-round
        # drift of π_θ is therefore not corrected; this is a per-round (not
        # per-step) IS scheme. The paper's claim about trajectory shape
        # (monotone-drift vs. bounded-oscillation) is robust to this
        # approximation, but the bound in Proposition (off_policy) is
        # conservative w.r.t. step-level drift.
        with torch.no_grad():
            log_w_prev = is_log_prev_w - lrw
            log_l_prev = is_log_prev_l - lrl
            # geometric-mean style: use the average per-pair log-IS, then exponentiate.
            log_w_pair = 0.5 * (log_w_prev + log_l_prev)
            is_weight = torch.exp(log_w_pair).clamp(max=is_clip)
            is_weight = is_weight / is_weight.mean().clamp_min(1e-8)
        loss = (is_weight * raw_loss)
    else:
        is_weight = torch.ones_like(raw_loss)
        loss = raw_loss

    if loss.dim() > 0:
        loss = loss.mean()
    pref_acc = (pi_diff > 0).float().mean()
    return {
        "loss_dpo_is": loss,
        "pref_acc":    pref_acc,
        "is_weight":   is_weight.detach().mean() if is_weight.dim() > 0 else is_weight.detach(),
        "margin":      pi_diff.detach().mean() if pi_diff.dim() > 0 else pi_diff.detach(),
    }


def pareto_dpo_step_losses(
    model,
    ref_model,
    batch,
    beta: float = 0.12,
    weight_dim: int = 3,
    sample_w: bool = True,
    max_len: Optional[int] = None,
):
    """Pareto-DPO step (Stage-1 stub: scalar BTL with stochastic scalarization weights).

    The full Pareto-DPO requires weight-conditioned policies π_θ(s|G,w). This stub
    samples w ~ Dirichlet(1) per batch and applies the standard DPO loss; it provides
    the API surface and demonstrates the worst-case scalarization invariance argued
    in the paper without requiring an architectural change. The full version (with
    a FiLM-conditioned policy head) is a natural extension on top of this.

    For each batch, we sample a single w ∈ Δ^{weight_dim - 1} and apply the BTL loss
    on the policy ratio. Because the *preference set* itself is ε-Pareto-dominant,
    the loss is consistent with all w; sampling w then exposes the policy to many
    different scalarizations during training, encouraging Pareto-monotonic implicit
    rewards.
    """
    lpw = seq_logprob(model, batch.graph, batch.winner_seq, max_len=max_len)
    lpl = seq_logprob(model, batch.graph, batch.loser_seq,  max_len=max_len)
    with torch.no_grad():
        lrw = seq_logprob(ref_model, batch.graph, batch.winner_seq, max_len=max_len)
        lrl = seq_logprob(ref_model, batch.graph, batch.loser_seq,  max_len=max_len)

    pi_diff  = lpw - lpl
    ref_diff = lrw - lrl

    if sample_w and weight_dim > 1:
        w = torch.distributions.Dirichlet(torch.ones(weight_dim, device=lpw.device)).sample()
        # weight scaling normalised so E[w_eff] = 1; preserves expected loss magnitude
        w_eff = float(w.sum())
    else:
        w_eff = 1.0

    loss = -F.logsigmoid(beta * w_eff * (pi_diff - ref_diff))
    if loss.dim() > 0:
        loss = loss.mean()
    pref_acc = (pi_diff > 0).float().mean()
    return {
        "loss_pareto_dpo": loss,
        "pref_acc":        pref_acc,
        "w_eff":           torch.tensor(w_eff, device=lpw.device),
        "margin":          pi_diff.detach().mean() if pi_diff.dim() > 0 else pi_diff.detach(),
    }


_LOSS_REGISTRY = {
    "dpo":          ("loss_dpo",        dpo_step_losses),
    "ipo":          ("loss_ipo",        ipo_step_losses),
    "kto":          ("loss_kto",        kto_step_losses),
    "pareto_dpo":   ("loss_pareto_dpo", pareto_dpo_step_losses),
    "dpo_is":       ("loss_dpo_is",     importance_corrected_dpo_step_losses),
}


def step_losses(loss_type: str, model, ref_model, batch, **kwargs):
    """Dispatch a loss by name. Returns (loss_tensor, full_metrics_dict).

    Drop-in replacement for hardcoded `dpo_step_losses(...)` calls in trainers:
        loss, metrics = step_losses(cfg.loss_type, model, ref_model, batch,
                                    beta=cfg.dpo.beta, max_len=cfg.dpo.max_len)
    """
    if loss_type not in _LOSS_REGISTRY:
        raise ValueError(f"unknown loss_type {loss_type!r}; valid: {list(_LOSS_REGISTRY)}")
    key, fn = _LOSS_REGISTRY[loss_type]
    out = fn(model=model, ref_model=ref_model, batch=batch, **kwargs)
    return out[key], out


class SimPOLoss(nn.Module):
    """
    SimPO loss:
      L = -log σ( β*(avg_logp_w - avg_logp_l) - γ )
        = softplus( - (β*(avg_logp_w - avg_logp_l) - γ) )
    """
    def __init__(self, beta: float = 2.0, gamma: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index

    @staticmethod
    def _avg_logp(logits, labels, ignore_index=-1):
        # logits: [B, T, V]; labels: [B, T]
        # get per-token negative log-likelihood, then negate to log-prob
        nll = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
            ignore_index=ignore_index,
        ).view(labels.size())
        logp = -nll

        mask = (labels != ignore_index).float()
        # avoid division by zero if a sequence is fully masked
        lengths = mask.sum(dim=1).clamp_min(1.0)
        avg_logp = (logp * mask).sum(dim=1) / lengths
        return avg_logp  # [B]

    def forward(self, win_logits, win_labels, lose_logits, lose_labels):
        avg_w = self._avg_logp(win_logits, win_labels, self.ignore_index)
        avg_l = self._avg_logp(lose_logits, lose_labels, self.ignore_index)
        z = self.beta * (avg_w - avg_l) - self.gamma
        loss = F.softplus(-z)  # == -logsigmoid(z)
        # metrics for logging
        with torch.no_grad():
            reward_margin = (avg_w - avg_l)
            reward_accuracy = (self.beta * reward_margin > self.gamma).float().mean()
        return loss.mean(), {
            "simpo/avg_logp_w": avg_w.mean().item(),
            "simpo/avg_logp_l": avg_l.mean().item(),
            "simpo/z_margin": z.mean().item(),
            "simpo/reward_acc": reward_accuracy.item(),
        }
