#!/usr/bin/env python3
"""
Extract DAS test set raw PDB files and sequences to organized folders.
Based on the extracted test set IDs, copy corresponding files and generate FASTA sequences.
"""

import os
import sys
import shutil
import torch
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

def copy_raw_pdb_files():
    """Copy raw PDB files for DAS test set"""
    print("📁 Copying DAS Test Set Raw PDB Files")
    print("=" * 50)
    
    # Paths
    raw_files_list = "./data/das_split_raw_data/test_set_raw_files.txt"
    source_dir = Path("./data/raw")
    target_dir = Path("./data/das_split_raw_data/das_split_raw_pdb")
    
    # Ensure target directory exists
    target_dir.mkdir(exist_ok=True)
    
    # Read file list
    with open(raw_files_list, 'r') as f:
        raw_files = [line.strip() for line in f if line.strip()]
    
    print(f"📋 Files to copy: {len(raw_files)}")
    
    copied_count = 0
    missing_count = 0
    
    for filename in raw_files:
        source_path = source_dir / filename
        target_path = target_dir / filename
        
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            copied_count += 1
        else:
            print(f"⚠️  Missing file: {filename}")
            missing_count += 1
    
    print(f"✅ Copied: {copied_count} files")
    print(f"❌ Missing: {missing_count} files")
    print(f"📁 Target directory: {target_dir}")
    
    return copied_count, missing_count

def extract_sequences_from_processed():
    """Extract sequences for DAS test set from processed.pt"""
    print("\n🧬 Extracting DAS Test Set Sequences")
    print("=" * 50)
    
    # Load processed data and test indices
    processed_pt = "./data/processed.pt"
    split_pt = "./data/das_split.pt"
    
    print(f"📂 Loading data from: {processed_pt}")
    all_raws = list(torch.load(processed_pt, map_location='cpu').values())
    
    print(f"📂 Loading split from: {split_pt}")
    train_idx, val_idx, test_idx = torch.load(split_pt, map_location='cpu')
    test_indices = list(map(int, test_idx))
    
    # Get test structures
    test_raws = [all_raws[i] for i in test_indices]
    print(f"📊 Test structures: {len(test_raws)}")
    
    # Extract sequences
    target_dir = Path("./data/das_split_raw_data/das_split_raw_seq")
    target_dir.mkdir(exist_ok=True)
    
    # Create sequence records
    sequences = []
    structure_info = []
    
    for idx, raw_item in enumerate(test_raws):
        global_idx = test_indices[idx]
        
        # Get sequence and metadata
        sequence = raw_item.get('sequence', '')
        id_list = raw_item.get('id_list', [])
        
        if not sequence:
            print(f"⚠️  No sequence for structure {idx} (global: {global_idx})")
            continue
        
        # Use the first canonical ID as the primary identifier
        from dpo.common_id import canonicalize_id, canonical_from_id_list
        primary_id = canonical_from_id_list(id_list) if id_list else f"test_structure_{idx}"
        
        # Create sequence record
        seq_record = SeqRecord(
            Seq(sequence),
            id=primary_id,
            description=f"DAS_test_idx={idx} global_idx={global_idx} chains={len(id_list)} length={len(sequence)}"
        )
        sequences.append(seq_record)
        
        # Store structure info for summary
        structure_info.append({
            'test_idx': idx,
            'global_idx': global_idx,
            'primary_id': primary_id,
            'sequence_length': len(sequence),
            'num_chains': len(id_list),
            'chain_ids': [canonicalize_id(str(id_val)) for id_val in id_list]
        })
    
    # Write FASTA file
    fasta_file = target_dir / "das_test_sequences.fasta"
    SeqIO.write(sequences, fasta_file, "fasta")
    print(f"✅ Saved {len(sequences)} sequences to: {fasta_file}")
    
    # Write structure summary
    summary_file = target_dir / "das_test_structure_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("DAS TEST SET STRUCTURE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total structures: {len(structure_info)}\n")
        f.write(f"Total chains: {sum(info['num_chains'] for info in structure_info)}\n\n")
        
        f.write("STRUCTURE DETAILS:\n")
        f.write("-" * 30 + "\n")
        for info in structure_info:
            f.write(f"Test #{info['test_idx']:2d} (Global {info['global_idx']:4d}): {info['primary_id']}\n")
            f.write(f"  Sequence length: {info['sequence_length']:3d}\n")
            f.write(f"  Number of chains: {info['num_chains']}\n")
            if info['num_chains'] > 1:
                f.write(f"  Chain IDs: {', '.join(info['chain_ids'][:5])}")
                if len(info['chain_ids']) > 5:
                    f.write(f" ... (+{len(info['chain_ids'])-5} more)")
                f.write("\n")
            f.write("\n")
    
    print(f"✅ Saved structure summary to: {summary_file}")
    
    return len(sequences), structure_info

def create_usage_readme():
    """Create README file explaining the extracted data"""
    print("\n📄 Creating Usage Documentation")
    print("=" * 30)
    
    readme_path = "./data/das_split_raw_data/README.md"
    
    readme_content = """# DAS Test Set Raw Data

This directory contains the extracted test set data from the DAS split for baseline model evaluation.

## Directory Structure

```
das_split_raw_data/
├── README.md                           # This file
├── test_set_structure_ids.txt          # 235 canonical structure IDs
├── test_set_raw_files.txt             # 235 corresponding raw file names
├── test_set_detailed_mapping.json     # Complete mapping with metadata
├── test_set_extraction_report.txt     # Detailed extraction report
├── das_split_raw_pdb/                 # Raw PDB/CIF files (235 files)
└── das_split_raw_seq/                 # Sequence data
    ├── das_test_sequences.fasta       # FASTA sequences for all 98 structures
    └── das_test_structure_summary.txt # Structure-by-structure summary
```

## Key Statistics

- **98 RNA structures** in DAS test split
- **235 unique chain IDs** (multi-chain complexes)
- **100% mapping success** to raw PDB files
- **No length mismatch issues** - all structures are complete

## Chain Distribution

- 57 structures with 1 chain each (57 IDs)
- 17 structures with 2 chains each (34 IDs)
- 5 structures with 3 chains each (15 IDs)
- 7 structures with 4 chains each (28 IDs)
- Plus larger complexes up to 17 chains

## Usage for Baseline Evaluation

### For structure-based models:
```bash
# Use PDB files in das_split_raw_pdb/
ls das_split_raw_pdb/*.pdb | head -5
```

### For sequence-based models:
```bash
# Use FASTA sequences in das_split_raw_seq/
head das_split_raw_seq/das_test_sequences.fasta
```

### For programmatic access:
```python
import json
with open('test_set_detailed_mapping.json') as f:
    mapping = json.load(f)
# Access mapped_ids, test_indices, etc.
```

## Canonicalization

All IDs follow the `PDBID_MODEL_CHAIN` format:
- `3B58_1_B-C-A` → `3B58_1_B`
- `7M57_1_qq-bb` → `7M57_1_qq`

## Generated by

This data was extracted using proven scripts from `dpo/scripts/` with the same 
canonicalization logic used throughout the RiboPO project.

Command used:
```bash
python -m dpo.scripts.extract_das_test_raw_data
```
"""
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ Created README: {readme_path}")

def main():
    print("🧬 DAS Test Set Raw Data Extraction")
    print("=" * 60)
    
    try:
        # Step 1: Copy raw PDB files
        copied, missing = copy_raw_pdb_files()
        
        # Step 2: Extract sequences
        seq_count, structure_info = extract_sequences_from_processed()
        
        # Step 3: Create documentation
        create_usage_readme()
        
        # Final summary
        print(f"\n🎉 Extraction Complete!")
        print(f"📊 Summary:")
        print(f"  Raw PDB files copied: {copied}")
        print(f"  Missing files: {missing}")
        print(f"  Sequences extracted: {seq_count}")
        print(f"  Total structures: {len(structure_info)}")
        print(f"📁 All data saved to: data/das_split_raw_data/")
        
        return 0
        
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())