#!/usr/bin/env python3
"""
Check for gaps and data incompleteness in DAS split PDB files
Outputs: 
- JSON file with gap information
- Statistics summary
"""

import os
import json
import torch
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# Set paths
PROJECT_ROOT = Path("/mnt/rna01/smh/projects/ribopo")
PDB_DIR = PROJECT_ROOT / "data/das_split_raw_data/das_split_raw_pdb"
DAS_SPLIT_FILE = PROJECT_ROOT / "data/das_split.pt"
OUTPUT_DIR = PROJECT_ROOT / "ribopo_v2/debug/data_clean"
OUTPUT_JSON = OUTPUT_DIR / "pdb_gap_analysis.json"
OUTPUT_SUMMARY = OUTPUT_DIR / "gap_summary.txt"

def load_das_split():
    """Load the DAS split dataset"""
    das_data = torch.load(DAS_SPLIT_FILE, map_location='cpu')
    return das_data

def check_pdb_gaps(pdb_path):
    """
    Check a PDB file for gaps and missing residues
    Returns:
    - has_gaps: boolean
    - gap_info: dict with gap details
    - sequence_length: int
    - structure_length: int
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('rna', pdb_path)
    
    gap_info = {
        'has_gaps': False,
        'missing_residues': [],
        'gap_regions': [],
        'chain_breaks': [],
        'sequence_length': 0,
        'structure_length': 0,
        'chains': []
    }
    
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            gap_info['chains'].append(chain_id)
            
            residues = list(chain)
            if len(residues) == 0:
                continue
                
            # Get residue numbers 
            res_nums = []
            for res in residues:
                # Skip hetero residues (water, ligands)
                if res.get_id()[0] == ' ':  # Standard residue
                    res_nums.append(res.get_id()[1])
            
            if len(res_nums) == 0:
                continue
                
            res_nums = sorted(res_nums)
            gap_info['structure_length'] = len(res_nums)
            
            # Check for gaps (missing residue numbers)
            expected_range = range(res_nums[0], res_nums[-1] + 1)
            missing = set(expected_range) - set(res_nums)
            
            if missing:
                gap_info['has_gaps'] = True
                gap_info['missing_residues'].extend([(chain_id, num) for num in sorted(missing)])
                
                # Find contiguous gap regions
                missing_sorted = sorted(missing)
                gap_start = missing_sorted[0]
                gap_end = missing_sorted[0]
                
                for i in range(1, len(missing_sorted)):
                    if missing_sorted[i] == missing_sorted[i-1] + 1:
                        gap_end = missing_sorted[i]
                    else:
                        gap_info['gap_regions'].append({
                            'chain': chain_id,
                            'start': gap_start,
                            'end': gap_end,
                            'length': gap_end - gap_start + 1
                        })
                        gap_start = missing_sorted[i]
                        gap_end = missing_sorted[i]
                
                # Add last gap region
                gap_info['gap_regions'].append({
                    'chain': chain_id,
                    'start': gap_start,
                    'end': gap_end,
                    'length': gap_end - gap_start + 1
                })
            
            # Check for chain breaks (large jumps in residue numbering)
            for i in range(1, len(res_nums)):
                if res_nums[i] - res_nums[i-1] > 1:
                    gap_info['chain_breaks'].append({
                        'chain': chain_id,
                        'before': res_nums[i-1],
                        'after': res_nums[i],
                        'gap_size': res_nums[i] - res_nums[i-1] - 1
                    })
    
    return gap_info

def check_sequence_structure_mismatch(pdb_path, expected_length=None):
    """
    Check if sequence length matches structure length
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('rna', pdb_path)
    
    # Count actual residues
    residue_count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':  # Standard residue
                    residue_count += 1
    
    mismatch_info = {
        'structure_residues': residue_count,
        'expected_length': expected_length,
        'has_mismatch': False
    }
    
    if expected_length and residue_count != expected_length:
        mismatch_info['has_mismatch'] = True
        mismatch_info['difference'] = residue_count - expected_length
        
    return mismatch_info

def main():
    """Main analysis function"""
    
    print("Loading DAS split data...")
    das_data = load_das_split()
    
    # Get all PDB files
    pdb_files = sorted(list(PDB_DIR.glob("*.pdb")))
    print(f"Found {len(pdb_files)} PDB files")
    
    # Analysis results
    results = {}
    gap_statistics = {
        'total_pdbs': len(pdb_files),
        'pdbs_with_gaps': 0,
        'pdbs_without_gaps': 0,
        'pdbs_with_chain_breaks': 0,
        'total_missing_residues': 0,
        'gap_size_distribution': defaultdict(int),
        'clean_pdbs': []  # List of PDBs without gaps
    }
    
    print("\nAnalyzing PDB files for gaps and incompleteness...")
    for pdb_path in tqdm(pdb_files):
        pdb_name = pdb_path.stem
        
        # Check for gaps
        gap_info = check_pdb_gaps(pdb_path)
        
        # Check sequence-structure mismatch
        # Note: We'd need the expected sequence length from the dataset
        # For now, just record the structure info
        
        results[pdb_name] = {
            'file': str(pdb_path.name),
            'has_gaps': gap_info['has_gaps'],
            'gap_info': gap_info,
            'clean': not gap_info['has_gaps']  # Mark as clean if no gaps
        }
        
        # Update statistics
        if gap_info['has_gaps']:
            gap_statistics['pdbs_with_gaps'] += 1
            gap_statistics['total_missing_residues'] += len(gap_info['missing_residues'])
            
            # Gap size distribution
            for gap_region in gap_info['gap_regions']:
                gap_statistics['gap_size_distribution'][gap_region['length']] += 1
        else:
            gap_statistics['pdbs_without_gaps'] += 1
            gap_statistics['clean_pdbs'].append(pdb_name)
        
        if gap_info['chain_breaks']:
            gap_statistics['pdbs_with_chain_breaks'] += 1
    
    # Convert defaultdict to regular dict for JSON serialization
    gap_statistics['gap_size_distribution'] = dict(gap_statistics['gap_size_distribution'])
    
    # Save results to JSON
    print(f"\nSaving results to {OUTPUT_JSON}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'analysis_date': str(Path(pdb_files[0]).stat().st_mtime),
        'statistics': gap_statistics,
        'pdb_details': results
    }
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Write summary
    print(f"Writing summary to {OUTPUT_SUMMARY}")
    with open(OUTPUT_SUMMARY, 'w') as f:
        f.write("=== PDB Gap Analysis Summary ===\n\n")
        f.write(f"Total PDB files analyzed: {gap_statistics['total_pdbs']}\n")
        f.write(f"PDBs WITHOUT gaps (clean): {gap_statistics['pdbs_without_gaps']} ({100*gap_statistics['pdbs_without_gaps']/gap_statistics['total_pdbs']:.1f}%)\n")
        f.write(f"PDBs WITH gaps: {gap_statistics['pdbs_with_gaps']} ({100*gap_statistics['pdbs_with_gaps']/gap_statistics['total_pdbs']:.1f}%)\n")
        f.write(f"PDBs with chain breaks: {gap_statistics['pdbs_with_chain_breaks']}\n")
        f.write(f"Total missing residues: {gap_statistics['total_missing_residues']}\n")
        f.write("\n=== Gap Size Distribution ===\n")
        
        if gap_statistics['gap_size_distribution']:
            for size, count in sorted(gap_statistics['gap_size_distribution'].items()):
                f.write(f"  Gap size {size}: {count} occurrences\n")
        
        f.write(f"\n=== Clean PDB Count ===\n")
        f.write(f"Total clean PDBs (no gaps): {len(gap_statistics['clean_pdbs'])}\n")
        f.write("\nClean PDB list saved in JSON file under 'statistics.clean_pdbs'\n")
    
    # Print summary to console
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"Total PDBs: {gap_statistics['total_pdbs']}")
    print(f"Clean PDBs (no gaps): {gap_statistics['pdbs_without_gaps']} ({100*gap_statistics['pdbs_without_gaps']/gap_statistics['total_pdbs']:.1f}%)")
    print(f"PDBs with gaps: {gap_statistics['pdbs_with_gaps']} ({100*gap_statistics['pdbs_with_gaps']/gap_statistics['total_pdbs']:.1f}%)")
    print(f"\nResults saved to:")
    print(f"  - {OUTPUT_JSON}")
    print(f"  - {OUTPUT_SUMMARY}")
    
    return gap_statistics['clean_pdbs']

if __name__ == "__main__":
    clean_pdbs = main()
    print(f"\nIdentified {len(clean_pdbs)} clean PDB files ready for copying")