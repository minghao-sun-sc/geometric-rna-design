# Evaluation Modules Documentation

## **Core Evaluation System**

### **Primary Files (Essential - Keep)**

#### **`src/evaluator.py` (2304 lines) - CORE EVALUATION ENGINE**
**Status**: ✅ **Production Ready - All 12 metrics working**

**Key Functions**:
- `evaluate()` - Main evaluation pipeline (lines 47-488)
- `self_consistency_score_rhofold_extended()` - 3D metrics with chain mapping (lines 690-850)
- `get_lddt_openstructure_v2()` - lDDT with multi-chain support (lines 1558-1736) ✅ **FIXED**
- `vienna_ensemble_metrics()` - Thermodynamic analysis (lines 2050-2120)
- `mcq_pseudotorsion_stats()` - Torsion angle analysis (lines 2200-2250)

**Metrics Implemented**:
1. Recovery/Perplexity (basic)
2. EternaFold 2D self-consistency  
3. RhoFold 3D: RMSD, TM-score, GDT, pLDDT, lDDT
4. INF: all, WC, non-WC, stack interactions
5. Clash scores: pre/post relaxation
6. MCQ: circular torsion analysis
7. Vienna: MFE, ED, entropy, p(S0), diversity, Tm
8. 3-mer sequence diversity

#### **`dpo/bench/eval_full.py` - MAIN EVALUATION PIPELINE**
**Status**: ✅ **Production Ready**

**Key Features**:
- Multi-checkpoint comparison
- WandB integration with consistent tables
- Command-line interface
- Comprehensive error handling
- JSON/CSV output formats

#### **`dpo/passk.py` - PASS@K ANALYSIS**
**Status**: ✅ **Production Ready**

**Functions**:
- `pass_at_k_unbiased()` - HumanEval-style estimator
- `pass_at_k_topk()` - Ranked selection analysis
- `Rule` class for threshold definitions

#### **`tools/usalign_utils.py` - STRUCTURAL ALIGNMENT**
**Status**: ✅ **Production Ready**

**Functions**:
- `run_rna_usalign()` - US-align wrapper for RNA
- `aggregate_tm()` - TM-score aggregation strategies

#### **`tools/run_phenix.sh` - CLASH SCORE CALCULATION**
**Status**: ✅ **Production Ready**

**Purpose**: Phenix MolProbity wrapper for dual clash score reporting

---

## **Configuration Files (Essential - Keep)**

### **Working Configurations**
- **`multiround/config/evaluation/01_eval_base_dpo_t05.yaml`** ✅ **Multi-checkpoint**
- **`dpo/configs/bench_full.yaml`** ✅ **Single checkpoint**

### **Experiment Configurations (12 total)**
- **`multiround/config/experiments/01_sft_ablation.yaml`** through **`12_simpo_dynamic_margins.yaml`**
- **Status**: ✅ All have proper inheritance and checkpoint saving

---

## **Debug/Development Files**

### **Currently Useful Debug Scripts**

#### **`dpo/debug/test_fixed_lddt.py`** ✅ **Keep - Validates lDDT fix**
- Tests chain mapping functionality
- Verifies lDDT calculation works with multi-chain structures

#### **`dpo/debug/analyze_chain_structure.py`** ✅ **Keep - Chain analysis tool**
- Analyzes PDB chain structure for debugging
- Useful for understanding structure complexity

#### **`dpo/debug/find_test_structures.py`** ✅ **Keep - Dataset inspection**
- Maps test indices to structure IDs
- Useful for debugging specific evaluation cases

### **Obsolete Debug Scripts (Propose for Deletion)**

#### **Multiple lDDT Debug Scripts (No longer needed)**
- `dpo/debug/debug_lddt_actual_files.py` ❌ **DELETE - Issue resolved**
- `dpo/debug/debug_lddt_detailed.py` ❌ **DELETE - Issue resolved**  
- `dpo/debug/simple_lddt_debug.py` ❌ **DELETE - Issue resolved**
- `dpo/debug/test_lddt_correct_pair.py` ❌ **DELETE - Issue resolved**
- `dpo/debug/test_lddt_robustness.py` ❌ **DELETE - Issue resolved**

#### **Old Evaluation Development Scripts**
- `dpo/debug/test_eval_fixes.py` ❌ **DELETE - Issues fixed**
- `dpo/debug/debug_full_pipeline.py` ❌ **DELETE - Pipeline working**
- `dpo/debug/debug_structure_metrics.py` ❌ **DELETE - Metrics working**

---

## **Redundant/Obsolete Modules**

### **Duplicate Evaluation Implementations**

#### **`multiround/eval_proper.py`** ❌ **DELETE - Redundant**
- **Issue**: Attempts to reimplement parts of `src/evaluator.py`
- **Problem**: Coordinate format mismatches, incomplete implementation
- **Solution**: Use `dpo/bench/eval_full.py` instead

#### **`dpo/bench/eval_benchmark.py`** ❌ **CHECK FOR DELETION**
- **Status**: Likely superseded by `eval_full.py`
- **Recommendation**: Compare functionality and delete if redundant

#### **Old Evaluation Scripts in `dpo/`**
- `dpo/benchmark_gpu.py` ❌ **DELETE - Old implementation**
- Any other `eval_*.py` files that predate the current working pipeline

---

## **File Organization Recommendations**

### **Keep (Essential)**
```
src/evaluator.py                    # Core evaluation engine
dpo/bench/eval_full.py              # Main pipeline  
dpo/passk.py                        # Pass@k analysis
tools/usalign_utils.py              # Structural alignment
tools/run_phenix.sh                 # Clash scores
multiround/config/evaluation/       # Working configs
multiround/config/experiments/      # Training configs (1-12)
```

### **Move to Archive (multiround/debug/integrative/)**
```
dpo/debug/test_fixed_lddt.py        # lDDT validation
dpo/debug/analyze_chain_structure.py # Chain analysis
dpo/debug/find_test_structures.py   # Dataset inspection
```

### **Propose for Deletion**
```
dpo/debug/debug_lddt_*.py           # lDDT debugging (5+ files)
dpo/debug/test_eval_*.py            # Evaluation debugging
dpo/debug/debug_*_pipeline.py      # Pipeline debugging  
multiround/eval_proper.py           # Redundant implementation
dpo/benchmark_gpu.py                # Old benchmark
```

---

## **Summary**

✅ **Core system is complete and working** (12 metrics, all validated)  
✅ **Main pipeline handles all use cases** (`eval_full.py`)  
✅ **Configuration system is clean** (inheritance working)  
❌ **~15+ debug files can be deleted** (issues resolved)  
❌ **~3 redundant evaluation modules** need cleanup  

The evaluation system is **production-ready** with comprehensive metrics and robust error handling.