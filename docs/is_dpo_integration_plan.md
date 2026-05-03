# Importance-Corrected Iterative DPO — Integration Plan

**Status:** Step function `importance_corrected_dpo_step_losses` is in `dpo/losses.py` and registered in the loss dispatcher. Full multi-round integration into `multiround/trainer.py` requires the steps below; estimated 1–2 days of careful implementation + testing.

## Why this matters

Theorem~\ref{thm:off_policy} (manuscript Appendix A.11) predicts that static-pair multi-round DPO has gradient bias growing linearly in round count `R`. Empirically, this matches the rebuttal R5+ degradation. Importance-corrected iterative DPO is the principled mitigation: at round `r >= 2`, reweight each pair by the policy density ratio between the current policy `π_θ_r` and the previous round's policy `π_θ_{r-1}` (truncated for stability).

## Required changes to `multiround/trainer.py`

### Step 1: extend the round driver

In `train_round(round_num)` (currently around line 167):

```python
def train_round(self, round_num: int) -> Dict:
    # ... existing setup ...

    # NEW: precompute importance-weight log-probs at the START of round r >= 2
    if round_num >= 2 and getattr(self.cfg, "use_is_correction", False):
        prev_round_ckpt = self._load_previous_round_policy(round_num - 1)
        self._precompute_pair_log_probs(prev_round_ckpt)
```

### Step 2: precompute per-pair log-probs

Add a helper method to `MultiRoundDPOTrainer`:

```python
def _precompute_pair_log_probs(self, prev_policy):
    """For each pair, compute log π_θ_{r-1}(s^w | G) and log π_θ_{r-1}(s^l | G).
    Store on the dataset items so the dataloader yields them in the batch.
    """
    from dpo.losses import seq_logprob
    prev_policy.eval()
    with torch.no_grad():
        for i, item in enumerate(self.train_loader.dataset):
            graph = item.graph.to(self.device)
            wseq = torch.as_tensor(item.winner_seq, device=self.device)
            lseq = torch.as_tensor(item.loser_seq,  device=self.device)
            log_prev_w = seq_logprob(prev_policy, graph, wseq).item()
            log_prev_l = seq_logprob(prev_policy, graph, lseq).item()
            self.train_loader.dataset.set_log_prev(i, log_prev_w, log_prev_l)
    print(f"[round {self.current_round}] precomputed {len(self.train_loader.dataset)} IS log-probs", flush=True)
```

### Step 3: modify `DPOPairDataset` to carry log-prev

In `dpo/data.py` `DPOPairDataset`:

```python
def __init__(self, ...):
    # existing ...
    self.log_prev_w = [None] * len(self.pairs)
    self.log_prev_l = [None] * len(self.pairs)

def set_log_prev(self, i, lw, ll):
    self.log_prev_w[i] = lw
    self.log_prev_l[i] = ll

def __getitem__(self, idx):
    # existing fields ...
    sample.log_prev_w = self.log_prev_w[idx]
    sample.log_prev_l = self.log_prev_l[idx]
    return sample
```

The `PairBatch` dataclass and collate function need the same fields.

### Step 4: swap loss in the inner step

Replace lines 739–747 of `multiround/trainer.py`:

```python
# old:
out = dpo_step_losses(
    model=self.trainer.policy, ref_model=self.trainer.reference,
    batch=batch, beta=cfg.dpo.beta,
    label_smoothing=cfg.dpo.label_smoothing,
    max_len=cfg.dpo.max_len,
)
loss = out["loss_dpo"]

# new:
if getattr(cfg, "use_is_correction", False) and self.current_round >= 2:
    from dpo.losses import importance_corrected_dpo_step_losses
    is_log_prev_w = batch.log_prev_w  # filled by collate; tensor [B]
    is_log_prev_l = batch.log_prev_l
    out = importance_corrected_dpo_step_losses(
        model=self.trainer.policy, ref_model=self.trainer.reference,
        batch=batch, beta=cfg.dpo.beta,
        is_clip=getattr(cfg, "is_clip", 5.0),
        is_log_prev_w=is_log_prev_w, is_log_prev_l=is_log_prev_l,
        max_len=cfg.dpo.max_len,
    )
    loss = out["loss_dpo_is"]
else:
    out = dpo_step_losses(...)  # round 1 or IS off
    loss = out["loss_dpo"]
```

### Step 5: optional — fully on-policy data resampling

For a stronger version, replace the static pair set after each round with fresh candidates sampled from `π_θ_{r-1}`:

```python
def _resample_pairs_from_policy(self, prev_policy, n_per_target=8, T=0.5):
    """For each backbone, sample n candidates from prev_policy at temp T, evaluate
    structural metrics with RhoFold+/Vienna, and rebuild the preference set with
    the same ε-Pareto filter as the original construction.
    """
    # ... wraps the existing pair-construction pipeline; see scripts/build_pairs.py
```

This is a heavier change because it requires running the full pair-construction (~3 days CPU per round on 4K targets). Not necessary for a first IS-DPO experiment — Step 4 alone (with the static pair set re-weighted) already partially mitigates the off-policy bias.

## Empirical validation plan

After integration:
1. Train multi-round DPO with `use_is_correction: false` for R=8 rounds → reproduce the rebuttal R5+ degradation (scMCC peaks at R2, drifts down to ~0.66 at R8).
2. Train with `use_is_correction: true` for R=8 rounds → expected: scMCC plateau extends, no degradation through R5–R8.
3. Plot KL(π_θ_r ‖ π_ref) over rounds for both — IS correction should keep KL bounded vs.\ linear growth in static-pair training.

This experiment is the empirical validation of Theorem~\ref{thm:off_policy} and would be a flagship result in §4 of the manuscript.

## Estimated effort

- Step 1–4 (loss swap + log-prev plumbing): ~1 day
- Step 5 (on-policy resampling): ~1 week (requires re-running RhoFold+ pair construction)
- Validation experiment (R=1–8 with both): ~2 days GPU
