#!/usr/bin/env python3
"""
Debug script to test evaluation dataset loading for multiround training.
This script mimics the evaluation dataset loading to identify the exact issue.
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import yaml
from types import SimpleNamespace as SN

def _to_sn(obj):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SN(**{k: _to_sn(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_sn(item) for item in obj]
    else:
        return obj

def debug_config_loading():
    """Test loading the multiround config and checking dataset paths."""
    print("=" * 60)
    print("🔍 DEBUG: Configuration Loading")
    print("=" * 60)
    
    config_path = "multiround/config/experiments/00_debug_a100_80g.yaml"
    print(f"Loading config: {config_path}")
    
    try:
        # Load config with inheritance (like train.py does)
        from multiround.utils import load_config_with_inheritance
        config_dict = load_config_with_inheritance(config_path)
        cfg = _to_sn(config_dict)
        
        print("✅ Config loaded successfully")
        
        # Check paths structure
        print(f"\nConfig structure check:")
        print(f"  cfg.paths exists: {hasattr(cfg, 'paths')}")
        if hasattr(cfg, 'paths'):
            paths_cfg = getattr(cfg, 'paths')
            print(f"  cfg.paths.pairs exists: {hasattr(paths_cfg, 'pairs')}")
            if hasattr(paths_cfg, 'pairs'):
                pairs_cfg = getattr(paths_cfg, 'pairs')
                print(f"  cfg.paths.pairs.test exists: {hasattr(pairs_cfg, 'test')}")
                if hasattr(pairs_cfg, 'test'):
                    test_path = getattr(pairs_cfg, 'test')
                    print(f"  cfg.paths.pairs.test = {test_path}")
                    print(f"  Test file exists: {os.path.exists(test_path)}")
                    
                    # Check file size and first few lines
                    if os.path.exists(test_path):
                        file_size = os.path.getsize(test_path)
                        print(f"  Test file size: {file_size} bytes")
                        
                        with open(test_path, 'r') as f:
                            first_line = f.readline().strip()
                            print(f"  First line preview: {first_line[:100]}...")
            
            # Check other required paths
            print(f"\nOther required paths:")
            required_paths = ['processed_pt', 'split_pt']
            for path_key in required_paths:
                if hasattr(paths_cfg, path_key):
                    path_value = getattr(paths_cfg, path_key)
                    exists = os.path.exists(path_value)
                    print(f"  {path_key}: {path_value} (exists: {exists})")
                else:
                    print(f"  {path_key}: MISSING")
        
        return cfg
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        traceback.print_exc()
        return None

def debug_dpo_dataset_loading(cfg):
    """Test DPO dataset loading (current approach that fails)."""
    print("\n" + "=" * 60)
    print("🔍 DEBUG: DPO Dataset Loading (Current Approach)")
    print("=" * 60)
    
    try:
        # This is what the current evaluator tries to do
        from dpo.data import DPOPairDataset
        
        paths_cfg = getattr(cfg, 'paths', None)
        pairs_cfg = getattr(paths_cfg, 'pairs', None) if paths_cfg else None
        
        if pairs_cfg and hasattr(pairs_cfg, 'test'):
            test_path = pairs_cfg.test
            print(f"Attempting to load DPOPairDataset from: {test_path}")
            
            # Try loading DPO dataset
            test_dataset = DPOPairDataset(
                pairs_path=test_path,
                processed_pt=cfg.paths.processed_pt,
                split_pt=cfg.paths.split_pt,
                # Add other required parameters
                split='test',
                subsample_pairs=None,
                device='cpu'
            )
            
            print(f"✅ DPOPairDataset loaded successfully")
            print(f"  Dataset length: {len(test_dataset)}")
            print(f"  Underlying dataset length: {len(test_dataset.dataset)}")
            
            # Try to access the underlying dataset
            eval_dataset = test_dataset.dataset
            print(f"✅ Evaluation dataset extracted: {len(eval_dataset)} structures")
            
            return eval_dataset
            
        else:
            print("❌ Config missing pairs.test path")
            return None
            
    except Exception as e:
        print(f"❌ DPO dataset loading failed: {e}")
        traceback.print_exc()
        return None

def debug_direct_dataset_loading(cfg):
    """Test direct dataset loading (like dpo/bench/eval_full.py)."""
    print("\n" + "=" * 60)  
    print("🔍 DEBUG: Direct Dataset Loading (Working Approach)")
    print("=" * 60)
    
    try:
        # This is how dpo/bench/eval_full.py loads the dataset
        from dpo.utils import load_processed_pt
        
        # Load processed data directly
        print(f"Loading processed data from: {cfg.paths.processed_pt}")
        all_items = load_processed_pt(cfg.paths.processed_pt)
        print(f"✅ Processed data loaded: {len(all_items)} total items")
        
        # Load split indices
        print(f"Loading split indices from: {cfg.paths.split_pt}")
        import torch
        tr, va, te = torch.load(cfg.paths.split_pt, map_location="cpu")
        tr, va, te = list(map(int, tr)), list(map(int, va)), list(map(int, te))
        print(f"✅ Split indices loaded:")
        print(f"  Train: {len(tr)} indices")
        print(f"  Val: {len(va)} indices") 
        print(f"  Test: {len(te)} indices")
        
        # Test accessing test split items
        print(f"\nTesting access to test split items...")
        test_items = [all_items[i] for i in te[:5]]  # Just first 5 for testing
        print(f"✅ Successfully accessed {len(test_items)} test items")
        
        # Check structure of first item
        if test_items:
            item = test_items[0]
            print(f"\nFirst test item structure:")
            print(f"  Keys: {list(item.keys())}")
            print(f"  Sequence length: {len(item['sequence'])}")
            print(f"  Number of conformers: {len(item['coords_list'])}")
            if 'id_list' in item:
                print(f"  ID: {item['id_list'][0] if item['id_list'] else 'N/A'}")
        
        return all_items, te
        
    except Exception as e:
        print(f"❌ Direct dataset loading failed: {e}")
        traceback.print_exc()
        return None, None

def debug_featurizer(cfg, all_items, test_indices):
    """Test featurizer with test data."""
    print("\n" + "=" * 60)
    print("🔍 DEBUG: Featurizer Testing")
    print("=" * 60)
    
    try:
        # Import featurizer
        from src.data.featurizer import RNAGraphFeaturizer
        
        # Create featurizer (using CPU to avoid device issues)
        featurizer = RNAGraphFeaturizer(
            split='test',
            radius=0.0,
            top_k=32,
            num_rbf=32,
            num_posenc=32,
            max_num_conformers=1,
            noise_scale=0.0,
            distance_eps=0.001,
            device='cpu'
        )
        
        print("✅ Featurizer created successfully")
        
        # Test featurizing one item
        test_idx = test_indices[0]
        item = all_items[test_idx]
        
        print(f"Testing featurization of item {test_idx}...")
        
        # Prepare data for featurizer (needs numpy arrays)
        from src.data.data_utils import get_backbone_coords
        import torch
        
        coords_list = []
        for coords in item["coords_list"]:
            if isinstance(coords, torch.Tensor):
                coords_list.append(get_backbone_coords(coords.clone().detach(), item["sequence"]))
            else:
                coords_list.append(get_backbone_coords(torch.tensor(coords), item["sequence"]))
        
        raw_for_featurizer = {
            "sequence": item["sequence"],
            "coords_list": [coords.numpy() for coords in coords_list],
            "sec_struct_list": item.get("sec_struct_list", [("." * len(item["sequence"])) for _ in coords_list])
        }
        
        # Featurize
        graph = featurizer.featurize(raw_for_featurizer)
        print(f"✅ Featurization successful")
        print(f"  Graph sequence length: {len(graph.seq)}")
        print(f"  Graph has mask_coords: {hasattr(graph, 'mask_coords')}")
        if hasattr(graph, 'mask_coords'):
            print(f"  Mask coords sum: {graph.mask_coords.sum()}/{len(graph.mask_coords)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Featurizer test failed: {e}")
        traceback.print_exc()
        return False

def test_evaluator_class(cfg):
    """Test the current MultiRoundEvaluator class."""
    print("\n" + "=" * 60)
    print("🔍 DEBUG: MultiRoundEvaluator Class")
    print("=" * 60)
    
    try:
        from multiround.evaluator import MultiRoundEvaluator
        
        print("Creating MultiRoundEvaluator...")
        evaluator = MultiRoundEvaluator(cfg, device='cpu')
        
        print(f"✅ Evaluator created")
        print(f"  eval_dataset is None: {evaluator.eval_dataset is None}")
        
        if evaluator.eval_dataset is None:
            print("❌ eval_dataset is None - this is the problem!")
        else:
            print(f"  eval_dataset length: {len(evaluator.eval_dataset)}")
            
        return evaluator
        
    except Exception as e:
        print(f"❌ Evaluator creation failed: {e}")
        traceback.print_exc()
        return None

def main():
    """Main debug function."""
    print("🔧 MultiRound Evaluation Dataset Debug Script")
    print("This script tests the evaluation dataset loading pipeline.")
    
    # Test 1: Config loading
    cfg = debug_config_loading()
    if cfg is None:
        print("❌ Cannot proceed - config loading failed")
        return
    
    # Test 2: Current DPO dataset approach (fails)
    eval_dataset_dpo = debug_dpo_dataset_loading(cfg)
    
    # Test 3: Direct dataset loading (should work)
    all_items, test_indices = debug_direct_dataset_loading(cfg)
    
    # Test 4: Featurizer
    if all_items and test_indices:
        featurizer_works = debug_featurizer(cfg, all_items, test_indices)
    
    # Test 5: Current evaluator class
    evaluator = test_evaluator_class(cfg)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DEBUG SUMMARY")
    print("=" * 60)
    print(f"✅ Config loading: {'SUCCESS' if cfg else 'FAILED'}")
    print(f"❌ DPO dataset approach: {'SUCCESS' if eval_dataset_dpo else 'FAILED'}")
    print(f"✅ Direct dataset approach: {'SUCCESS' if all_items else 'FAILED'}")
    print(f"✅ Featurizer: {'SUCCESS' if all_items and featurizer_works else 'FAILED'}")
    print(f"❌ Current evaluator: {'SUCCESS' if evaluator and evaluator.eval_dataset else 'FAILED'}")
    
    print("\n🔧 SOLUTION:")
    print("The multiround evaluator should use direct dataset loading like dpo/bench/eval_full.py")
    print("instead of trying to extract from DPOPairDataset.")

if __name__ == "__main__":
    main()