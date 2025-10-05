# RiboPO v2 Workflow Summary

**Generated**: 2025-01-04
**Framework**: Winner-Focused DPO for RNA Inverse Folding

---

## Executive Summary

RiboPO v2 represents a paradigm shift in RNA inverse folding optimization, moving from complex multi-objective optimization to a **winner-focused approach**. The key insight: DPO learning signal is dominated by the quality of chosen (winner) sequences, not the complexity of negative mining.

### Key Changes from v1
1. **Larger candidate pools** (200-500 vs 50-100) → Higher chance of excellent winners
2. **Simple pairing** (best-vs-random) → High-yield pairs with minimal complexity  
3. **Compact reward** (TM + INF + ED/L) → Focused on core RNA properties
4. **Offline-first** → Stable, reproducible baseline before online refinement

---

## Completed Work (Phase 0) ✅

### Data Cleaning & Filtering
- **Two-stage filtering**: Length-based + GC content
- **Achievement**: 85.9% reduction in training set variance (518→73 nt std)
- **Final dataset**: 
  - TRAIN: 4,865 sequences (84.7% retention)
  - VAL: 192 sequences (improved GC: 29.2%→55.1%)
  - TEST: 182 sequences (100% retention)

### Key Artifacts
- `data_clean_filtered/` - Clean PDB structures
- `native_seq/` - Extracted FASTA sequences
- `eda/results/` - Complete statistical analysis

---

## Implementation Pipeline

### Phase 1: Candidate Generation (Week 1)
**Goal**: Generate 200-500 diverse candidates per backbone

**Strategy**:
- Temperature diversity: {0.1, 0.5, 1.0}
- Decoding strategies: greedy, top-p, top-k
- Baseline anchors: native + GC-shuffle

**Key Code**:
```python
generator = CandidateGenerator(model_checkpoint)
candidates = generator.generate_diverse_candidates(
    backbone_pdb, n_candidates=200, temperatures=[0.1, 0.5, 1.0]
)
```

### Phase 2: Metric Computation (Week 1)
**Goal**: Comprehensive evaluation with efficient caching

**Metrics Stack**:
- **3D**: RhoFold+ → TM/RMSD/pLDDT, US-align → lDDT, DSSR → INF
- **2D**: ViennaRNA → MFE/ED/Entropy/P(target)
- **Quality**: Clash scores, MCQ angles

**Caching Strategy**:
```python
cache_file = f"cache/metrics/{backbone_id}_{seq_hash}.json"
```

### Phase 3: Normalization (Week 2)
**Goal**: Fair comparison across variable-length RNAs

**Method**: Per-backbone percentile ranking
```python
TM_norm = percentile_rank(TM, per_backbone=True)
INF_norm = percentile_rank(INF, per_backbone=True)  
ED_norm = percentile_rank(1 - ED/L, per_backbone=True)
```

### Phase 4: Reward & Winners (Week 2)
**Goal**: Select top 5-10 winners per backbone

**Formula**:
```python
S_total = 0.50 * TM_norm + 0.30 * INF_norm + 0.20 * ED_norm
```

**Gates**:
- Loose TM gate: ≥0.20 (or RMSD ≤12Å)
- pLDDT down-weighting if <0.6

### Phase 5: Pair Construction (Week 2)
**Goal**: High-yield training pairs

**Strategy**: Best-vs-Random
- Each winner → 3 random non-winners
- Margin: 0.8 * MAD(ΔS_total)
- Light veto: drop if winner worse by >0.75σ

---

## Technical Architecture

```
Input: Clean PDBs (4,865 train backbones)
   ↓
[Candidate Generation]
   - 200 sequences/backbone
   - 970K total candidates
   ↓
[Metric Computation]
   - ~30 metrics per sequence
   - Cached for efficiency
   ↓
[Normalization]
   - Per-backbone percentiles
   - Robust to length variation
   ↓
[Winner Selection]  
   - Top 5-10 per backbone
   - ~25-50K winners
   ↓
[Pair Construction]
   - 3 negatives per winner
   - ~75-150K pairs
   ↓
Output: pairs_offline.jsonl, winners_only.jsonl
```

---

## Resource Requirements

### Computational
- **GPU**: 1x A100 for candidate generation
- **CPU**: 32 cores for parallel metric computation
- **RAM**: 64GB for caching
- **Storage**: ~100GB for candidates + metrics

### Time Estimates
- **Candidate generation**: 2-3 days (200 seq/backbone)
- **Metric computation**: 3-4 days (with caching)
- **Normalization + Winners**: 1 day
- **Pair construction**: 4 hours

**Total**: ~1 week for complete pipeline

---

## Quality Control Gates

### Phase 1 (Candidates)
- [ ] Unique sequences >95%
- [ ] 3-mer diversity >0.5
- [ ] Coverage: all backbones have candidates

### Phase 2 (Metrics)
- [ ] Cache hit rate >90% on re-runs
- [ ] No failed evaluations
- [ ] Metric distributions reasonable

### Phase 3-4 (Winners)
- [ ] Every backbone has ≥1 winner
- [ ] Winner S_total >0.5
- [ ] TM scores show improvement over random

### Phase 5 (Pairs)
- [ ] ≥500 valid pairs
- [ ] Margin consistency across backbones
- [ ] Delta_S distribution centered >0

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Low winner quality | Increase candidates to 500, add local search |
| Metric computation bottleneck | Implement better caching, use parallel processing |
| Normalization issues | Use robust statistics (MAD), winsorize outliers |
| Insufficient pairs | Lower margin, use semi-hard negatives |

---

## Next Critical Decisions

1. **Local search implementation**: Worth the complexity?
2. **Online refinement timing**: When to transition from offline?
3. **SFT baseline importance**: How much effort on baseline?
4. **Evaluation metrics**: Which subset for fast iteration?

---

## Success Metrics

### Short-term (2 weeks)
- Generate first 1000 preference pairs
- Winners show TM >0.25 average
- SFT baseline training converges

### Medium-term (1 month)  
- DPO model shows improvement over SFT
- Validation metrics: TM >0.35, INF >0.5
- Online refinement shows gains

### Long-term (2 months)
- **Target**: TM ≥0.45, INF ≥0.7, ED/L ≤0.3
- Pass@k metrics competitive with baselines
- Reproducible pipeline with documentation

---

## Commands for Quick Start

```bash
# 1. Generate candidates for one backbone (test)
python ribopo_v2/candidate_generation/generate_candidates.py \
    --backbone data_clean_filtered/train/1CSL_1_B.pdb \
    --n_candidates 50 \
    --output test_candidates.json

# 2. Compute metrics for candidates
python ribopo_v2/metric_computation/run_metrics.py \
    --candidates test_candidates.json \
    --cache_dir cache/metrics/

# 3. Full pipeline for subset
python ribopo_v2/run_pipeline.py \
    --phase all \
    --backbone_dir data_clean_filtered/train/ \
    --limit 10 \  # First 10 backbones only
    --n_candidates 100
```

---

## Documentation Structure

```
ribopo_v2/
├── RIBOPO_V2_PROGRESS.md      # Progress tracker (UPDATE DAILY)
├── IMPLEMENTATION_WORKFLOW.md  # Detailed technical workflow
├── ACTION_ITEMS.md            # Immediate tasks (CHECK DAILY)
├── WORKFLOW_SUMMARY.md        # This document
└── logs/
    └── daily/                 # Daily progress logs
```

---

## Contact & Resources

- **Project Lead**: [Your name]
- **gRNAde Repo**: [Link to original gRNAde]
- **RhoFold+ Model**: `tools/rhofold/model_20221010_params.pt`
- **Dataset**: `/mnt/rna01/smh/projects/ribopo/ribopo_v2/data_clean_filtered/`

---

*This workflow is designed for systematic execution with clear checkpoints and quality gates at each phase.*