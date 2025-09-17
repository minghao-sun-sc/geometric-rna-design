#!/usr/bin/env python3
"""
Analyze why we have 235 IDs from 98 test structures.
Investigate multiple chains per structure and completeness.
"""

import sys
import os
import json
import torch
from pathlib import Path
from collections import defaultdict, Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()
from dpo.common_id import canonicalize_id

def analyze_test_structure_composition():
    """Analyze the composition of test structures to understand 235 IDs vs 98 structures"""
    print("🔍 Analyzing Test Structure Composition")
    print("=" * 50)
    
    # Load our detailed mapping
    mapping_file = "/mnt/rna01/smh/projects/ribopo/dpo/debug/test_set_detailed_mapping.json"
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # Load processed data to examine structure details
    processed_pt = "/mnt/rna01/smh/projects/ribopo/data/processed.pt"
    all_raws = list(torch.load(processed_pt, map_location='cpu').values())
    
    # Get test indices
    test_indices = list(set(mapping_data['test_indices']))
    test_raws = [all_raws[i] for i in test_indices]
    
    print(f"📊 Basic Statistics:")
    print(f"  Test structure items: {len(test_raws)}")
    print(f"  Total structure IDs: {len(mapping_data['mapped_ids'])}")
    
    # Analyze ID composition per structure
    structure_analysis = []
    chains_per_structure = Counter()
    length_issues = []
    
    for idx, raw_item in enumerate(test_raws):
        global_idx = test_indices[idx]
        
        # Get all IDs for this structure
        id_list = raw_item.get('id_list', [])
        canonical_ids = [canonicalize_id(str(id_val)) for id_val in id_list if canonicalize_id(str(id_val))]
        
        # Extract base PDB ID (before chain)
        base_pdbs = set()
        for canonical_id in canonical_ids:
            parts = canonical_id.split('_')
            if len(parts) >= 2:
                base_pdb = f"{parts[0]}_{parts[1]}"  # PDBID_MODEL
                base_pdbs.add(base_pdb)
        
        # Get sequence info
        sequence = raw_item.get('sequence', '')
        seq_len = len(sequence)
        
        # Check for potential length issues
        coords_list = raw_item.get('coords_list', [])
        if coords_list:
            coord_lens = [len(coords) for coords in coords_list]
            coord_len_avg = sum(coord_lens) / len(coord_lens)
            if abs(seq_len - coord_len_avg) > 2:  # Allow small tolerance
                length_issues.append({
                    'global_idx': global_idx,
                    'seq_len': seq_len,
                    'coord_len_avg': coord_len_avg,
                    'canonical_ids': canonical_ids[:3]  # First 3 for display
                })
        
        chains_per_structure[len(canonical_ids)] += 1
        
        structure_analysis.append({
            'global_idx': global_idx,
            'test_idx': idx,
            'base_pdbs': list(base_pdbs),
            'canonical_ids': canonical_ids,
            'num_chains': len(canonical_ids),
            'sequence_length': seq_len,
            'sample_sequence': sequence[:50] + '...' if len(sequence) > 50 else sequence
        })
    
    # Analysis results
    print(f"\n🔗 Chain Distribution Analysis:")
    for num_chains, count in sorted(chains_per_structure.items()):
        total_ids = num_chains * count
        print(f"  {num_chains} chain(s): {count} structures → {total_ids} IDs")
    
    print(f"\n📋 Multi-chain Structure Examples:")
    multi_chain_examples = [s for s in structure_analysis if s['num_chains'] > 1]
    for example in multi_chain_examples[:5]:
        print(f"  Structure {example['test_idx']} (global {example['global_idx']}):")
        print(f"    Base PDB(s): {example['base_pdbs']}")
        print(f"    Chains: {example['canonical_ids']}")
        print(f"    Sequence length: {example['sequence_length']}")
    
    print(f"\n⚠️ Potential Length Issues:")
    if length_issues:
        print(f"  Found {len(length_issues)} structures with seq/coord length mismatches:")
        for issue in length_issues[:5]:
            print(f"    Global {issue['global_idx']}: seq_len={issue['seq_len']}, coord_len≈{issue['coord_len_avg']:.1f}")
            print(f"      IDs: {issue['canonical_ids']}")
    else:
        print(f"  ✅ No significant length mismatches detected")
    
    # Summary explanation
    total_expected_ids = sum(num_chains * count for num_chains, count in chains_per_structure.items())
    print(f"\n💡 Explanation:")
    print(f"  • {len(test_raws)} RNA structures in test set")
    print(f"  • Each structure can have multiple chains/conformers")
    print(f"  • Expected total IDs: {total_expected_ids}")
    print(f"  • Actual extracted IDs: {len(mapping_data['mapped_ids'])}")
    print(f"  • This is normal - RNA complexes often have multiple chains")
    
    return {
        'structure_analysis': structure_analysis,
        'chains_per_structure': dict(chains_per_structure),
        'length_issues': length_issues,
        'multi_chain_count': len(multi_chain_examples),
        'total_structures': len(test_raws),
        'total_ids': len(mapping_data['mapped_ids'])
    }

def main():
    try:
        results = analyze_test_structure_composition()
        
        # Save analysis results
        output_file = "/mnt/rna01/smh/projects/ribopo/dpo/debug/structure_composition_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Analysis saved to: {output_file}")
        return 0
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())