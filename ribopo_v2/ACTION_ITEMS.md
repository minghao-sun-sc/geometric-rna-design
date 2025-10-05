# RiboPO v2 Immediate Action Items

**Priority**: Critical path to first preference pairs
**Timeline**: Next 2 weeks focus

---

## 🚨 Week 1: Foundation (Jan 4-11, 2025)

### Day 1-2: Environment Setup
- [ ] **Verify gRNAde checkpoint access**
  ```bash
  ls -la checkpoints/grnade_base.pt
  # If missing, download from original repo
  ```

- [ ] **Check evaluation tool availability**
  ```bash
  which USalign  # Should point to tools/USalign/USalign
  python -c "import RNA"  # ViennaRNA bindings
  ls tools/rhofold/  # RhoFold+ model files
  ```

- [ ] **Create directory structure**
  ```bash
  cd ribopo_v2
  mkdir -p {candidate_generation,metric_computation,normalization,winner_selection,pair_construction}
  mkdir -p {output/candidates,cache/metrics,results}
  ```

### Day 3-4: Candidate Generation Testing
- [ ] **Test gRNAde sampling**
  ```python
  # Quick test script
  from src.models import RNADesignModel
  model = RNADesignModel.from_pretrained('checkpoints/grnade_base.pt')
  
  # Test on one backbone
  test_backbone = 'data_clean_filtered/train/1CSL_1_B.pdb'
  candidates = model.sample(test_backbone, n=10, temperature=0.5)
  print(f"Generated {len(candidates)} sequences")
  ```

- [ ] **Implement diversity sampling**
  - Temperature variation: {0.1, 0.5, 1.0}
  - Top-p sampling with p={0.9, 0.95}
  - Add native sequence as anchor

### Day 5-7: Metric Pipeline
- [ ] **Set up metric caching system**
  ```python
  # Simple JSON cache per backbone
  cache_file = f"cache/metrics/{backbone_id}.json"
  ```

- [ ] **Test metric computation on 5 sequences**
  - RhoFold+ → TM/RMSD
  - ViennaRNA → MFE, ED
  - Time each metric, identify bottlenecks

---

## 🎯 Week 2: Integration (Jan 11-18, 2025)

### Day 8-9: Normalization Development
- [ ] **Analyze metric distributions**
  ```python
  # On 100 test sequences
  import pandas as pd
  df = pd.read_csv('test_metrics.csv')
  print(df[['tm_score', 'inf_all', 'ed_per_nt']].describe())
  ```

- [ ] **Implement percentile ranking**
  - Per-backbone normalization
  - Check std deviation ∈ [0.25, 0.35]

### Day 10-11: Winner Selection
- [ ] **Test reward formula**
  ```python
  S_total = 0.50 * tm_norm + 0.30 * inf_norm + 0.20 * ed_norm
  ```

- [ ] **Verify winner quality**
  - Each backbone has ≥1 winner
  - Winners have S_total > 0.5

### Day 12-14: Pair Construction
- [ ] **Generate first batch of pairs**
  - Best-vs-random strategy
  - Calculate margins (0.8 * MAD)
  - Output to `pairs_offline.jsonl`

- [ ] **Quality check pairs**
  ```python
  # Verify pair quality
  pairs = load_pairs('pairs_offline.jsonl')
  print(f"Total pairs: {len(pairs)}")
  print(f"Avg delta_S: {np.mean([p['delta_s'] for p in pairs])}")
  ```

---

## 📊 Quick Validation Checklist

### After Week 1:
- [ ] Can generate 200 candidates/backbone in <5 minutes
- [ ] Metrics computed for 100 test sequences
- [ ] Cache system working (90% hit rate on re-runs)

### After Week 2:
- [ ] 500+ preference pairs generated
- [ ] Winners show TM > 0.20, reasonable INF scores
- [ ] Ready for SFT baseline training

---

## 🔧 Debug Commands

```bash
# Monitor GPU usage during generation
watch -n 1 nvidia-smi

# Check cache effectiveness
du -sh cache/metrics/
ls cache/metrics/ | wc -l

# Validate pair file format
head -n 5 pairs_offline.jsonl | python -m json.tool

# Quick metric stats
python -c "
import json
with open('winners.csv') as f:
    import pandas as pd
    df = pd.read_csv(f)
    print(df[['tm_score', 's_total']].describe())
"
```

---

## 🚀 Parallel Execution Opportunities

Run these in parallel to save time:

1. **Candidate generation** per backbone (parallel across backbones)
2. **RhoFold+ predictions** (batch by 10 sequences)
3. **ViennaRNA calculations** (embarrassingly parallel)
4. **Metric caching** (parallel read/write with file locks)

```bash
# Example parallel execution
parallel -j 8 python generate_candidates.py --backbone {} ::: data_clean_filtered/train/*.pdb
```

---

## 📈 Success Criteria for Week 2

| Metric | Target | Acceptable |
|--------|--------|------------|
| Candidates per backbone | 200 | 100+ |
| Unique sequences % | >95% | >90% |
| Winners per backbone | 5-10 | ≥1 |
| Total preference pairs | 1000+ | 500+ |
| Avg winner TM-score | >0.30 | >0.20 |
| Avg winner S_total | >0.6 | >0.5 |

---

## 🔴 Potential Blockers & Solutions

| Blocker | Solution |
|---------|----------|
| gRNAde OOM on large backbones | Reduce batch size, use gradient checkpointing |
| RhoFold+ too slow | Use smaller test set, implement better caching |
| Low winner quality | Increase candidates to 500, add local search |
| Insufficient pairs | Lower margin threshold, use semi-hard negatives |

---

## 📝 Daily Log Template

```markdown
### Date: [YYYY-MM-DD]
**Completed**:
- [ ] Task 1
- [ ] Task 2

**Metrics**:
- Candidates generated: X
- Winners identified: Y  
- Pairs created: Z

**Issues**:
- Issue 1: [Description] → [Solution]

**Next**:
- Priority task for tomorrow
```

---

*Update this document daily with progress checkmarks and metric results.*