#!/usr/bin/env python3
"""
Copy clean PDB files (without gaps) to the new directory structure.
This script reads the gap analysis JSON and copies only the clean PDB files.
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# Set paths
PROJECT_ROOT = Path("/mnt/rna01/smh/projects/ribopo")
SOURCE_PDB_DIR = PROJECT_ROOT / "data/das_split_raw_data/das_split_raw_pdb"
TARGET_PDB_DIR = PROJECT_ROOT / "ribopo_v2/data/native_clean/native_struct_clean"
GAP_ANALYSIS_JSON = Path(__file__).parent / "pdb_gap_analysis.json"
COPY_LOG = Path(__file__).parent / "copy_clean_pdbs.log"

def copy_clean_pdbs():
    """Copy clean PDB files to the new directory structure."""
    
    # Load gap analysis results
    print("Loading gap analysis results...")
    with open(GAP_ANALYSIS_JSON, 'r') as f:
        gap_analysis = json.load(f)
    
    clean_pdbs = gap_analysis['statistics']['clean_pdbs']
    total_clean = len(clean_pdbs)
    
    print(f"Found {total_clean} clean PDB files to copy")
    
    # Create target directory
    TARGET_PDB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize copy log
    copy_summary = {
        'total_clean_pdbs': total_clean,
        'successfully_copied': 0,
        'failed_copies': [],
        'missing_source_files': [],
        'copied_files': []
    }
    
    print(f"Copying clean PDB files to {TARGET_PDB_DIR}...")
    
    for pdb_name in tqdm(clean_pdbs, desc="Copying files"):
        source_file = SOURCE_PDB_DIR / f"{pdb_name}.pdb"
        target_file = TARGET_PDB_DIR / f"{pdb_name}.pdb"
        
        try:
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                copy_summary['successfully_copied'] += 1
                copy_summary['copied_files'].append(pdb_name)
            else:
                print(f"Warning: Source file not found: {source_file}")
                copy_summary['missing_source_files'].append(pdb_name)
                
        except Exception as e:
            print(f"Error copying {pdb_name}: {e}")
            copy_summary['failed_copies'].append({'file': pdb_name, 'error': str(e)})
    
    # Save copy log
    with open(COPY_LOG, 'w') as f:
        json.dump(copy_summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("COPY OPERATION COMPLETE")
    print("="*50)
    print(f"Total clean PDBs identified: {total_clean}")
    print(f"Successfully copied: {copy_summary['successfully_copied']}")
    print(f"Failed copies: {len(copy_summary['failed_copies'])}")
    print(f"Missing source files: {len(copy_summary['missing_source_files'])}")
    print(f"\nClean PDB files copied to: {TARGET_PDB_DIR}")
    print(f"Copy log saved to: {COPY_LOG}")
    
    return copy_summary

def verify_copy():
    """Verify that all copied files exist and have the same size as originals."""
    print("\nVerifying copied files...")
    
    with open(GAP_ANALYSIS_JSON, 'r') as f:
        gap_analysis = json.load(f)
    clean_pdbs = gap_analysis['statistics']['clean_pdbs']
    
    verification_results = {
        'verified_files': 0,
        'size_mismatches': [],
        'missing_copies': []
    }
    
    for pdb_name in tqdm(clean_pdbs[:10], desc="Verifying (sample)"):  # Verify first 10 as sample
        source_file = SOURCE_PDB_DIR / f"{pdb_name}.pdb"
        target_file = TARGET_PDB_DIR / f"{pdb_name}.pdb"
        
        if not target_file.exists():
            verification_results['missing_copies'].append(pdb_name)
            continue
            
        if source_file.exists():
            source_size = source_file.stat().st_size
            target_size = target_file.stat().st_size
            
            if source_size == target_size:
                verification_results['verified_files'] += 1
            else:
                verification_results['size_mismatches'].append({
                    'file': pdb_name,
                    'source_size': source_size,
                    'target_size': target_size
                })
    
    print(f"Verification complete (sample of 10 files):")
    print(f"Verified files: {verification_results['verified_files']}")
    print(f"Size mismatches: {len(verification_results['size_mismatches'])}")
    print(f"Missing copies: {len(verification_results['missing_copies'])}")
    
    return verification_results

if __name__ == "__main__":
    print("=== Copying Clean PDB Files ===")
    copy_summary = copy_clean_pdbs()
    
    if copy_summary['successfully_copied'] > 0:
        verify_copy()