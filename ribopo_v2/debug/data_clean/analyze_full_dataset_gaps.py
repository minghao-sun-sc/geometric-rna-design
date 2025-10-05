#!/usr/bin/env python3
"""
Comprehensive gap analysis for the full DAS dataset (train/val/test).
This script analyzes all splits for gaps and copies only clean data to RiboPO v2.
"""

import os
import json
import shutil
from pathlib import Path
from Bio.PDB import PDBParser
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# Set paths
PROJECT_ROOT = Path("/mnt/rna01/smh/projects/ribopo")
RAW_PDB_DIR = PROJECT_ROOT / "data/raw"
SPLIT_IDS_DIR = PROJECT_ROOT / "data/split_ids"
RIBOPO_V2_DATA = PROJECT_ROOT / "ribopo_v2/data"
OUTPUT_DIR = PROJECT_ROOT / "ribopo_v2/debug/data_clean"

# Output files
FULL_GAP_ANALYSIS_JSON = OUTPUT_DIR / "full_dataset_gap_analysis.json"
FULL_GAP_SUMMARY = OUTPUT_DIR / "full_dataset_gap_summary.txt"
COPY_LOG = OUTPUT_DIR / "full_dataset_copy_log.json"

class FullDatasetGapAnalyzer:
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)
        self.splits = self._load_split_ids()
        self.analysis_results = {}
        
        # Create output directories
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (RIBOPO_V2_DATA / "train").mkdir(parents=True, exist_ok=True)
        (RIBOPO_V2_DATA / "val").mkdir(parents=True, exist_ok=True)
        (RIBOPO_V2_DATA / "test").mkdir(parents=True, exist_ok=True)
    
    def _load_split_ids(self):
        """Load train/val/test split IDs."""
        splits = {}
        
        # Load train IDs
        with open(SPLIT_IDS_DIR / "train_ids_das.txt", 'r') as f:
            splits['train'] = [line.strip() for line in f if line.strip()]
        
        # Load val IDs
        with open(SPLIT_IDS_DIR / "val_ids_das.txt", 'r') as f:
            splits['val'] = [line.strip() for line in f if line.strip()]
        
        # Load test IDs
        with open(SPLIT_IDS_DIR / "test_ids_das.txt", 'r') as f:
            splits['test'] = [line.strip() for line in f if line.strip()]
        
        return splits
    
    def check_pdb_gaps(self, pdb_path):
        """
        Check a PDB file for gaps and missing residues.
        Returns gap_info dict with detailed gap analysis.
        """
        try:
            structure = self.pdb_parser.get_structure('rna', pdb_path)
        except Exception as e:
            return {
                'has_gaps': True,
                'error': str(e),
                'missing_residues': [],
                'gap_regions': [],
                'chain_breaks': [],
                'sequence_length': 0,
                'structure_length': 0,
                'chains': []
            }
        
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
    
    def analyze_split(self, split_name):
        """Analyze gaps for a specific split (train/val/test)."""
        print(f"\nAnalyzing {split_name} split...")
        
        split_ids = self.splits[split_name]
        split_results = {
            'total_ids': len(split_ids),
            'found_pdbs': 0,
            'missing_pdbs': 0,
            'clean_pdbs': 0,
            'pdbs_with_gaps': 0,
            'clean_pdb_list': [],
            'missing_pdb_list': [],
            'gap_pdb_list': [],
            'gap_statistics': defaultdict(int),
            'detailed_results': {}
        }
        
        for pdb_id in tqdm(split_ids, desc=f"Processing {split_name}"):
            pdb_path = RAW_PDB_DIR / f"{pdb_id}.pdb"
            
            if not pdb_path.exists():
                split_results['missing_pdbs'] += 1
                split_results['missing_pdb_list'].append(pdb_id)
                continue
            
            split_results['found_pdbs'] += 1
            
            # Check for gaps
            gap_info = self.check_pdb_gaps(pdb_path)
            
            split_results['detailed_results'][pdb_id] = {
                'file': f"{pdb_id}.pdb",
                'has_gaps': gap_info['has_gaps'],
                'gap_info': gap_info,
                'clean': not gap_info['has_gaps']
            }
            
            if gap_info['has_gaps']:
                split_results['pdbs_with_gaps'] += 1
                split_results['gap_pdb_list'].append(pdb_id)
                
                # Update gap statistics
                for gap_region in gap_info['gap_regions']:
                    split_results['gap_statistics'][gap_region['length']] += 1
            else:
                split_results['clean_pdbs'] += 1
                split_results['clean_pdb_list'].append(pdb_id)
        
        # Convert defaultdict to regular dict for JSON serialization
        split_results['gap_statistics'] = dict(split_results['gap_statistics'])
        
        self.analysis_results[split_name] = split_results
        
        print(f"{split_name.upper()} Results:")
        print(f"  Total IDs: {split_results['total_ids']}")
        print(f"  Found PDBs: {split_results['found_pdbs']}")
        print(f"  Missing PDBs: {split_results['missing_pdbs']}")
        print(f"  Clean PDBs: {split_results['clean_pdbs']} ({100*split_results['clean_pdbs']/split_results['found_pdbs']:.1f}%)")
        print(f"  PDBs with gaps: {split_results['pdbs_with_gaps']} ({100*split_results['pdbs_with_gaps']/split_results['found_pdbs']:.1f}%)")
        
        return split_results
    
    def copy_clean_pdbs(self, split_name):
        """Copy clean PDB files for a specific split."""
        print(f"\nCopying clean {split_name} PDBs...")
        
        split_results = self.analysis_results[split_name]
        clean_pdbs = split_results['clean_pdb_list']
        target_dir = RIBOPO_V2_DATA / split_name
        
        copy_summary = {
            'split': split_name,
            'total_clean_pdbs': len(clean_pdbs),
            'successfully_copied': 0,
            'failed_copies': [],
            'copied_files': []
        }
        
        for pdb_id in tqdm(clean_pdbs, desc=f"Copying {split_name}"):
            source_file = RAW_PDB_DIR / f"{pdb_id}.pdb"
            target_file = target_dir / f"{pdb_id}.pdb"
            
            try:
                shutil.copy2(source_file, target_file)
                copy_summary['successfully_copied'] += 1
                copy_summary['copied_files'].append(pdb_id)
            except Exception as e:
                print(f"Error copying {pdb_id}: {e}")
                copy_summary['failed_copies'].append({'file': pdb_id, 'error': str(e)})
        
        print(f"  Copied {copy_summary['successfully_copied']}/{len(clean_pdbs)} clean {split_name} PDBs")
        return copy_summary
    
    def save_results(self):
        """Save all analysis results to JSON and generate summary."""
        # Save detailed JSON results
        with open(FULL_GAP_ANALYSIS_JSON, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        # Generate summary report
        with open(FULL_GAP_SUMMARY, 'w') as f:
            f.write("=== Full DAS Dataset Gap Analysis Summary ===\n\n")
            
            total_found = 0
            total_clean = 0
            total_gaps = 0
            
            for split_name in ['train', 'val', 'test']:
                if split_name in self.analysis_results:
                    split_data = self.analysis_results[split_name]
                    found = split_data['found_pdbs']
                    clean = split_data['clean_pdbs']
                    gaps = split_data['pdbs_with_gaps']
                    
                    total_found += found
                    total_clean += clean
                    total_gaps += gaps
                    
                    f.write(f"=== {split_name.upper()} Split ===\n")
                    f.write(f"Total IDs in split file: {split_data['total_ids']}\n")
                    f.write(f"PDB files found: {found}\n")
                    f.write(f"PDB files missing: {split_data['missing_pdbs']}\n")
                    f.write(f"Clean PDBs (no gaps): {clean} ({100*clean/found:.1f}%)\n")
                    f.write(f"PDBs with gaps: {gaps} ({100*gaps/found:.1f}%)\n")
                    
                    if split_data['gap_statistics']:
                        f.write(f"Gap size distribution:\n")
                        for size, count in sorted(split_data['gap_statistics'].items()):
                            f.write(f"  Gap size {size}: {count} occurrences\n")
                    
                    f.write(f"\n")
            
            f.write(f"=== OVERALL SUMMARY ===\n")
            f.write(f"Total PDB files found: {total_found}\n")
            f.write(f"Total clean PDBs: {total_clean} ({100*total_clean/total_found:.1f}%)\n")
            f.write(f"Total PDBs with gaps: {total_gaps} ({100*total_gaps/total_found:.1f}%)\n")
            
            f.write(f"\n=== Clean Data for RiboPO v2 ===\n")
            for split_name in ['train', 'val', 'test']:
                if split_name in self.analysis_results:
                    clean_count = self.analysis_results[split_name]['clean_pdbs']
                    f.write(f"{split_name}: {clean_count} clean PDBs\n")
        
        print(f"\nResults saved to:")
        print(f"  - {FULL_GAP_ANALYSIS_JSON}")
        print(f"  - {FULL_GAP_SUMMARY}")
    
    def run_full_analysis(self):
        """Run complete gap analysis for all splits."""
        print("=== Full DAS Dataset Gap Analysis ===")
        
        print(f"Dataset splits loaded:")
        for split_name, ids in self.splits.items():
            print(f"  {split_name}: {len(ids)} IDs")
        
        copy_logs = {}
        
        # Analyze each split
        for split_name in ['train', 'val', 'test']:
            self.analyze_split(split_name)
            copy_logs[split_name] = self.copy_clean_pdbs(split_name)
        
        # Save results
        self.save_results()
        
        # Save copy logs
        with open(COPY_LOG, 'w') as f:
            json.dump(copy_logs, f, indent=2)
        
        # Print overall summary
        print("\n" + "="*60)
        print("FULL DATASET ANALYSIS COMPLETE")
        print("="*60)
        
        total_found = sum(self.analysis_results[split]['found_pdbs'] for split in self.analysis_results)
        total_clean = sum(self.analysis_results[split]['clean_pdbs'] for split in self.analysis_results)
        total_gaps = sum(self.analysis_results[split]['pdbs_with_gaps'] for split in self.analysis_results)
        
        print(f"Overall Statistics:")
        print(f"  Total PDB files found: {total_found}")
        print(f"  Clean PDBs (no gaps): {total_clean} ({100*total_clean/total_found:.1f}%)")
        print(f"  PDBs with gaps: {total_gaps} ({100*total_gaps/total_found:.1f}%)")
        
        print(f"\nClean data copied to:")
        for split_name in ['train', 'val', 'test']:
            clean_count = self.analysis_results[split_name]['clean_pdbs']
            print(f"  {split_name}: {clean_count} files → {RIBOPO_V2_DATA / split_name}")
        
        return self.analysis_results

def main():
    analyzer = FullDatasetGapAnalyzer()
    results = analyzer.run_full_analysis()
    return results

if __name__ == "__main__":
    main()