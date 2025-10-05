#!/usr/bin/env python3
"""
Final comprehensive script to investigate the missing structure IDs.
"""

import torch
from collections import Counter

def investigate_missing_ids():
    """Main investigation function."""
    dataset_path = "/mnt/rna01/smh/projects/ribopo/ribopo_v2/data_clean_filtered/processed.pt"
    problematic_ids = ['5HCQ_1_2x', '8AGW_1_x', '2DR2_1_B', '4V99_1_Ac', '7M57_1_h-S']
    
    print("="*70)
    print("COMPREHENSIVE INVESTIGATION OF MISSING STRUCTURE IDs")
    print("="*70)
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    try:
        with open(dataset_path, 'rb') as f:
            data = torch.load(f, map_location='cpu')
        print(f"✅ Successfully loaded dataset with {len(data)} sequence entries")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Extract all IDs from id_list fields
    print("\nExtracting all structure IDs from dataset...")
    all_ids = set()
    id_counter = Counter()
    sequence_to_ids = {}
    
    for sequence, item_data in data.items():
        if 'id_list' in item_data:
            id_list = item_data['id_list']
            sequence_to_ids[sequence] = id_list
            for structure_id in id_list:
                all_ids.add(structure_id)
                id_counter[structure_id] += 1
        else:
            print(f"⚠️ Warning: No id_list found for sequence: {sequence[:50]}...")
    
    print(f"✅ Found {len(all_ids)} unique structure IDs across all sequences")
    
    # Check problematic IDs
    print("\n" + "="*50)
    print("CHECKING PROBLEMATIC IDs:")
    print("="*50)
    
    found_ids = []
    missing_ids = []
    
    for target_id in problematic_ids:
        if target_id in all_ids:
            found_ids.append(target_id)
            count = id_counter[target_id]
            print(f"✅ FOUND: {target_id} (appears {count} times)")
            
            # Find which sequences contain this ID
            containing_sequences = []
            for seq, ids in sequence_to_ids.items():
                if target_id in ids:
                    containing_sequences.append(seq[:50] + "..." if len(seq) > 50 else seq)
            print(f"   Found in {len(containing_sequences)} sequences")
            
        else:
            missing_ids.append(target_id)
            print(f"❌ MISSING: {target_id}")
    
    # Analyze missing IDs patterns
    if missing_ids:
        print("\n" + "="*50)
        print("ANALYZING MISSING IDs:")
        print("="*50)
        
        for missing_id in missing_ids:
            print(f"\nAnalyzing: {missing_id}")
            parts = missing_id.split('_')
            if len(parts) >= 3:
                pdb_code = parts[0]
                model_num = parts[1] 
                chain_desc = parts[2]
                
                print(f"  PDB Code: {pdb_code}")
                print(f"  Model: {model_num}")
                print(f"  Chain/Description: {chain_desc}")
                
                # Look for similar PDB codes
                similar_ids = [id_val for id_val in all_ids if id_val.startswith(pdb_code + '_')]
                if similar_ids:
                    print(f"  ✅ Found similar IDs in dataset: {similar_ids}")
                else:
                    print(f"  ❌ No similar PDB codes found in dataset")
                
                # Look for any IDs with this PDB code (regardless of model/chain)
                any_pdb_ids = [id_val for id_val in all_ids if id_val.startswith(pdb_code)]
                if any_pdb_ids:
                    print(f"  ⚠️ Found any IDs with PDB {pdb_code}: {any_pdb_ids}")
    
    # Generate detailed statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS:")
    print("="*50)
    
    print(f"Total sequences in dataset: {len(data)}")
    print(f"Total unique structure IDs: {len(all_ids)}")
    print(f"Average IDs per sequence: {sum(id_counter.values()) / len(data):.2f}")
    print(f"Most frequent ID: {id_counter.most_common(1)[0][0]} ({id_counter.most_common(1)[0][1]} occurrences)")
    
    # Show ID distribution
    print(f"\nID frequency distribution:")
    freq_dist = Counter(id_counter.values())
    for freq, count in sorted(freq_dist.items()):
        print(f"  {count} IDs appear {freq} time(s)")
    
    # Show some sample IDs to understand naming patterns
    print(f"\nSample structure IDs (first 20):")
    for i, structure_id in enumerate(sorted(all_ids)[:20]):
        print(f"  {i+1:2d}. {structure_id}")
    
    print(f"\nSample structure IDs (last 20):")
    for i, structure_id in enumerate(sorted(all_ids)[-20:]):
        print(f"  {i+1:2d}. {structure_id}")
    
    # Suggest reasons for missing IDs
    print("\n" + "="*50)
    print("POTENTIAL REASONS FOR MISSING IDs:")
    print("="*50)
    
    if missing_ids:
        print("Based on the analysis, the problematic IDs might be missing due to:")
        print("1. 🔍 Data filtering: Structures may have been filtered out during data cleaning")
        print("2. 📝 Chain naming: Chain identifiers may have been standardized (e.g., '2x' → 'A', 'x' → 'X')")
        print("3. 🧹 Quality control: Low-quality structures may have been removed")
        print("4. 📊 Multi-chain handling: Complex multi-chain structures may have been simplified")
        print("5. 🔧 Processing pipeline: IDs may have been modified during processing")
        
        # Check for pattern similarities
        print("\n🔍 Pattern analysis:")
        for missing_id in missing_ids:
            pdb_code = missing_id.split('_')[0]
            chain_part = missing_id.split('_')[2] if len(missing_id.split('_')) > 2 else ""
            
            # Look for potential alternatives
            potential_alts = []
            for existing_id in all_ids:
                if existing_id.startswith(pdb_code + '_'):
                    potential_alts.append(existing_id)
            
            if potential_alts:
                print(f"  {missing_id} → Possible alternatives: {potential_alts}")
            else:
                print(f"  {missing_id} → No alternatives found with same PDB code")
    
    else:
        print("✅ All problematic IDs were found in the dataset!")
    
    # Save detailed report
    output_file = "/mnt/rna01/smh/projects/ribopo/ribopo_v2/missing_ids_investigation_report.txt"
    with open(output_file, 'w') as f:
        f.write("MISSING STRUCTURE IDs INVESTIGATION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Total sequences: {len(data)}\n")
        f.write(f"Total unique IDs: {len(all_ids)}\n\n")
        
        f.write("PROBLEMATIC IDs STATUS:\n")
        f.write("-" * 30 + "\n")
        for target_id in problematic_ids:
            if target_id in all_ids:
                f.write(f"✅ FOUND: {target_id}\n")
            else:
                f.write(f"❌ MISSING: {target_id}\n")
        
        f.write(f"\nALL {len(all_ids)} STRUCTURE IDs IN DATASET:\n")
        f.write("-" * 40 + "\n")
        for structure_id in sorted(all_ids):
            f.write(f"{structure_id}\n")
    
    print(f"\n📝 Detailed report saved to: {output_file}")
    
    return {
        'total_sequences': len(data),
        'total_unique_ids': len(all_ids),
        'found_ids': found_ids,
        'missing_ids': missing_ids,
        'all_ids': all_ids
    }

if __name__ == "__main__":
    results = investigate_missing_ids()
    
    print("\n" + "="*70)
    print("INVESTIGATION COMPLETE")
    print("="*70)
    print(f"Summary: {len(results['found_ids'])}/{len(results['found_ids']) + len(results['missing_ids'])} problematic IDs found")
