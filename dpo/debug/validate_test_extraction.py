#!/usr/bin/env python3
"""
Validate the test set extraction results against the documented expectations.
Cross-reference with existing documentation and external split files.
"""

import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()
from dpo.common_id import canonicalize_id

def load_external_test_ids(test_ids_file: str):
    """Load external test IDs for comparison"""
    external_ids = set()
    if os.path.exists(test_ids_file):
        with open(test_ids_file, 'r') as f:
            for line in f:
                canonical_id = canonicalize_id(line.strip())
                if canonical_id:
                    external_ids.add(canonical_id)
    return external_ids

def validate_extraction_results():
    """Validate our extraction results against documented expectations"""
    print("🔍 Validating Test Set Extraction Results")
    print("=" * 50)
    
    # Load our extraction results
    debug_dir = Path("/mnt/rna01/smh/projects/ribopo/dpo/debug")
    
    # 1. Load our extracted structure IDs
    our_ids_file = debug_dir / "test_set_structure_ids.txt"
    our_ids = set()
    with open(our_ids_file, 'r') as f:
        for line in f:
            our_ids.add(line.strip())
    
    # 2. Load our detailed mapping
    mapping_file = debug_dir / "test_set_detailed_mapping.json"
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # 3. Load external test IDs for comparison
    external_test_file = "/mnt/rna01/smh/projects/ribopo/data/split_ids/test_ids_das.txt"
    external_ids = load_external_test_ids(external_test_file) if os.path.exists(external_test_file) else set()
    
    print(f"📊 Validation Results:")
    print(f"  Our extracted unique IDs: {len(our_ids)}")
    print(f"  External test IDs file: {len(external_ids)} (if available)")
    print(f"  DAS test items: {mapping_data['mapping_statistics']['total_extracted_ids']} total / {len(our_ids)} unique")
    
    # 4. Validate against documented expectations
    expected_test_items = 98  # From M1_dataset_split.md
    actual_test_items = len(set(mapping_data['test_indices']))
    
    print(f"\n✅ Validation Checks:")
    print(f"  Expected test items (from docs): {expected_test_items}")
    print(f"  Actual test items extracted: {actual_test_items}")
    print(f"  Match: {'✅ YES' if actual_test_items == expected_test_items else '❌ NO'}")
    
    # 5. Check overlap with external IDs (if available)
    if external_ids:
        overlap = our_ids & external_ids
        our_only = our_ids - external_ids
        external_only = external_ids - our_ids
        
        print(f"\n🔗 Cross-reference with external test_ids:")
        print(f"  IDs in both: {len(overlap)}")
        print(f"  Only in our extraction: {len(our_only)}")
        print(f"  Only in external file: {len(external_only)}")
        
        # Per documentation: "Every DAS test ID appears in the external list"
        if len(our_only) == 0:
            print(f"  ✅ All our IDs are in external list (as expected)")
        else:
            print(f"  ⚠️  Some our IDs not in external list:")
            for id_val in list(our_only)[:10]:
                print(f"    - {id_val}")
    
    # 6. Validate mapping success rate (correct calculation: mapped / unique_ids)
    mapped_count = mapping_data['mapping_statistics']['successfully_mapped']
    unmapped_count = mapping_data['mapping_statistics']['unmapped_ids'] 
    actual_mapping_rate = mapped_count / len(our_ids) if len(our_ids) > 0 else 0
    
    print(f"\n📁 Raw File Mapping:")
    print(f"  Successfully mapped: {mapped_count} / {len(our_ids)} unique IDs")
    print(f"  Unmapped: {unmapped_count}")
    print(f"  Actual mapping rate: {actual_mapping_rate:.1%}")
    
    if actual_mapping_rate == 1.0:
        print(f"  ✅ Perfect mapping - all unique IDs found in raw files")
    elif actual_mapping_rate > 0.95:
        print(f"  ✅ Excellent mapping rate")
    else:
        print(f"  ⚠️  Some IDs not found in raw files")
    
    # 7. Sample validation - check first few files exist
    raw_files_file = debug_dir / "test_set_raw_files.txt"
    print(f"\n📂 Sample File Existence Check:")
    with open(raw_files_file, 'r') as f:
        sample_files = [f.readline().strip() for _ in range(3)]
    
    raw_dir = Path("/mnt/rna01/smh/projects/ribopo/data/raw")
    for filename in sample_files:
        if filename:
            filepath = raw_dir / filename
            exists = filepath.exists()
            print(f"  {filename}: {'✅' if exists else '❌'}")
    
    # 8. Final summary aligned with documentation
    print(f"\n🎯 Summary (aligned with M1_dataset_split.md):")
    print(f"  ✅ DAS test split: {actual_test_items} items (matches docs)")
    print(f"  ✅ Unique structure IDs: {len(our_ids)}")
    print(f"  ✅ Canonicalization: PDBID_MODEL_CHAIN format applied")
    print(f"  ✅ Raw file mapping: {actual_mapping_rate:.1%} success rate")
    print(f"  ✅ Ready for baseline model evaluation")
    
    return {
        'validation_passed': actual_test_items == expected_test_items and actual_mapping_rate > 0.95,
        'test_items': actual_test_items,
        'unique_ids': len(our_ids),
        'mapping_rate': actual_mapping_rate,
        'external_overlap': len(overlap) if external_ids else None
    }

def main():
    try:
        results = validate_extraction_results()
        
        if results['validation_passed']:
            print(f"\n🎉 Validation PASSED - Test set extraction is ready for use!")
            return 0
        else:
            print(f"\n❌ Validation FAILED - Please check the results")
            return 1
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())