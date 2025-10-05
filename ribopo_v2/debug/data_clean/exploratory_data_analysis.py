#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for RiboPO v2 clean dataset.
This script analyzes the distribution of the clean PDB files for DPO training.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils import seq1
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set paths
PROJECT_ROOT = Path("/mnt/rna01/smh/projects/ribopo")
CLEAN_PDB_DIR = PROJECT_ROOT / "ribopo_v2/data/native_clean/native_struct_clean"
DAS_SPLIT_FILE = PROJECT_ROOT / "data/das_split.pt"
GAP_ANALYSIS_JSON = PROJECT_ROOT / "ribopo_v2/debug/data_clean/pdb_gap_analysis.json"
EDA_OUTPUT_DIR = PROJECT_ROOT / "ribopo_v2/debug/data_clean/eda_results"

# RNA nucleotide mapping
RNA_MAPPING = {
    'A': 'A', 'U': 'U', 'G': 'G', 'C': 'C',
    'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C'
}

class CleanDatasetAnalyzer:
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)
        self.clean_pdb_names = self._load_clean_pdbs()
        self.das_data = self._load_das_data()
        self.analysis_results = {}
        
        # Create output directory
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_clean_pdbs(self):
        """Load list of clean PDB names from gap analysis."""
        with open(GAP_ANALYSIS_JSON, 'r') as f:
            gap_analysis = json.load(f)
        return gap_analysis['statistics']['clean_pdbs']
    
    def _load_das_data(self):
        """Load DAS split data."""
        return torch.load(DAS_SPLIT_FILE, map_location='cpu')
    
    def analyze_sequence_lengths(self):
        """Analyze sequence length distribution."""
        print("Analyzing sequence lengths...")
        lengths = []
        
        for pdb_name in tqdm(self.clean_pdb_names):
            pdb_path = CLEAN_PDB_DIR / f"{pdb_name}.pdb"
            try:
                structure = self.pdb_parser.get_structure('rna', pdb_path)
                
                # Count residues in each chain
                total_residues = 0
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if residue.get_id()[0] == ' ':  # Standard residue
                                total_residues += 1
                
                lengths.append(total_residues)
            except Exception as e:
                print(f"Error processing {pdb_name}: {e}")
        
        self.analysis_results['sequence_lengths'] = {
            'lengths': lengths,
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'quartiles': np.percentile(lengths, [25, 50, 75]).tolist()
        }
        
        # Plot distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(lengths, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sequence Lengths')
        plt.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.boxplot(lengths)
        plt.ylabel('Sequence Length')
        plt.title('Sequence Length Box Plot')
        
        plt.subplot(2, 2, 3)
        plt.hist(lengths, bins=30, cumulative=True, alpha=0.7, density=True)
        plt.xlabel('Sequence Length')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution')
        
        plt.subplot(2, 2, 4)
        # Log scale histogram
        plt.hist(lengths, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency (log scale)')
        plt.yscale('log')
        plt.title('Sequence Length Distribution (Log Scale)')
        
        plt.tight_layout()
        plt.savefig(EDA_OUTPUT_DIR / 'sequence_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sequence length statistics:")
        print(f"  Mean: {np.mean(lengths):.1f}")
        print(f"  Median: {np.median(lengths):.1f}")
        print(f"  Std: {np.std(lengths):.1f}")
        print(f"  Range: {np.min(lengths)} - {np.max(lengths)}")
    
    def analyze_nucleotide_composition(self):
        """Analyze nucleotide composition across the dataset."""
        print("Analyzing nucleotide composition...")
        
        nucleotide_counts = {'A': 0, 'U': 0, 'G': 0, 'C': 0, 'T': 0, 'Other': 0}
        per_structure_composition = []
        
        for pdb_name in tqdm(self.clean_pdb_names):
            pdb_path = CLEAN_PDB_DIR / f"{pdb_name}.pdb"
            try:
                structure = self.pdb_parser.get_structure('rna', pdb_path)
                
                structure_counts = {'A': 0, 'U': 0, 'G': 0, 'C': 0, 'T': 0, 'Other': 0}
                total_residues = 0
                
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if residue.get_id()[0] == ' ':
                                res_name = residue.get_resname().strip()
                                if res_name in RNA_MAPPING:
                                    base = RNA_MAPPING[res_name]
                                    nucleotide_counts[base] += 1
                                    structure_counts[base] += 1
                                else:
                                    nucleotide_counts['Other'] += 1
                                    structure_counts['Other'] += 1
                                total_residues += 1
                
                # Calculate percentages for this structure
                if total_residues > 0:
                    structure_composition = {k: v/total_residues for k, v in structure_counts.items()}
                    structure_composition['total_residues'] = total_residues
                    structure_composition['pdb_name'] = pdb_name
                    per_structure_composition.append(structure_composition)
                    
            except Exception as e:
                print(f"Error processing {pdb_name}: {e}")
        
        # Global composition
        total_nucleotides = sum(nucleotide_counts.values())
        global_composition = {k: v/total_nucleotides for k, v in nucleotide_counts.items()}
        
        self.analysis_results['nucleotide_composition'] = {
            'global_counts': nucleotide_counts,
            'global_percentages': global_composition,
            'per_structure': per_structure_composition
        }
        
        # Plot composition
        plt.figure(figsize=(15, 10))
        
        # Global composition pie chart
        plt.subplot(2, 3, 1)
        main_bases = ['A', 'U', 'G', 'C']
        sizes = [global_composition[base] for base in main_bases]
        plt.pie(sizes, labels=main_bases, autopct='%1.1f%%', startangle=90)
        plt.title('Global Nucleotide Composition')
        
        # Global composition bar chart
        plt.subplot(2, 3, 2)
        bases = list(global_composition.keys())
        percentages = list(global_composition.values())
        plt.bar(bases, percentages)
        plt.ylabel('Percentage')
        plt.title('Global Nucleotide Distribution')
        
        # Per-structure composition variability
        plt.subplot(2, 3, 3)
        composition_df = pd.DataFrame(per_structure_composition)
        for base in main_bases:
            plt.hist(composition_df[base], alpha=0.7, label=base, bins=20)
        plt.xlabel('Percentage in Structure')
        plt.ylabel('Number of Structures')
        plt.title('Per-Structure Composition Variability')
        plt.legend()
        
        # GC content distribution
        plt.subplot(2, 3, 4)
        gc_content = composition_df['G'] + composition_df['C']
        plt.hist(gc_content, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('GC Content')
        plt.ylabel('Number of Structures')
        plt.title('GC Content Distribution')
        plt.axvline(gc_content.mean(), color='red', linestyle='--', label=f'Mean: {gc_content.mean():.3f}')
        plt.legend()
        
        # AU content distribution
        plt.subplot(2, 3, 5)
        au_content = composition_df['A'] + composition_df['U']
        plt.hist(au_content, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('AU Content')
        plt.ylabel('Number of Structures')
        plt.title('AU Content Distribution')
        plt.axvline(au_content.mean(), color='red', linestyle='--', label=f'Mean: {au_content.mean():.3f}')
        plt.legend()
        
        # Composition heatmap
        plt.subplot(2, 3, 6)
        composition_matrix = composition_df[main_bases].head(50).T  # Top 50 structures
        sns.heatmap(composition_matrix, cmap='viridis', cbar=True)
        plt.xlabel('Structure Index (first 50)')
        plt.ylabel('Nucleotide')
        plt.title('Composition Heatmap')
        
        plt.tight_layout()
        plt.savefig(EDA_OUTPUT_DIR / 'nucleotide_composition.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Global nucleotide composition:")
        for base, percentage in global_composition.items():
            print(f"  {base}: {percentage:.3f} ({nucleotide_counts[base]} nucleotides)")
    
    def analyze_structural_features(self):
        """Analyze structural features like RMSD, radius of gyration, etc."""
        print("Analyzing structural features...")
        
        structural_features = []
        
        for pdb_name in tqdm(self.clean_pdb_names[:50]):  # Sample first 50 for structural analysis
            pdb_path = CLEAN_PDB_DIR / f"{pdb_name}.pdb"
            try:
                structure = self.pdb_parser.get_structure('rna', pdb_path)
                
                # Extract C4' coordinates (representative RNA backbone atoms)
                c4_coords = []
                all_coords = []
                
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if residue.get_id()[0] == ' ':
                                try:
                                    # C4' atom coordinates
                                    c4_atom = residue["C4'"]
                                    c4_coords.append(c4_atom.get_coord())
                                    
                                    # All heavy atom coordinates
                                    for atom in residue:
                                        if atom.element != 'H':
                                            all_coords.append(atom.get_coord())
                                except KeyError:
                                    continue
                
                if len(c4_coords) > 1:
                    c4_coords = np.array(c4_coords)
                    all_coords = np.array(all_coords)
                    
                    # Calculate radius of gyration
                    centroid = np.mean(c4_coords, axis=0)
                    rog = np.sqrt(np.mean(np.sum((c4_coords - centroid)**2, axis=1)))
                    
                    # Calculate end-to-end distance
                    end_to_end = np.linalg.norm(c4_coords[0] - c4_coords[-1])
                    
                    # Calculate compactness
                    compactness = rog / len(c4_coords)**0.5
                    
                    structural_features.append({
                        'pdb_name': pdb_name,
                        'num_residues': len(c4_coords),
                        'radius_of_gyration': rog,
                        'end_to_end_distance': end_to_end,
                        'compactness': compactness
                    })
                    
            except Exception as e:
                print(f"Error analyzing structure {pdb_name}: {e}")
        
        if structural_features:
            features_df = pd.DataFrame(structural_features)
            
            self.analysis_results['structural_features'] = {
                'features': structural_features,
                'summary_stats': features_df.describe().to_dict()
            }
            
            # Plot structural features
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.scatter(features_df['num_residues'], features_df['radius_of_gyration'])
            plt.xlabel('Number of Residues')
            plt.ylabel('Radius of Gyration (Å)')
            plt.title('Radius of Gyration vs. Size')
            
            plt.subplot(2, 3, 2)
            plt.scatter(features_df['num_residues'], features_df['end_to_end_distance'])
            plt.xlabel('Number of Residues')
            plt.ylabel('End-to-End Distance (Å)')
            plt.title('End-to-End Distance vs. Size')
            
            plt.subplot(2, 3, 3)
            plt.hist(features_df['compactness'], bins=15, alpha=0.7, edgecolor='black')
            plt.xlabel('Compactness')
            plt.ylabel('Frequency')
            plt.title('Compactness Distribution')
            
            plt.subplot(2, 3, 4)
            plt.hist(features_df['radius_of_gyration'], bins=15, alpha=0.7, edgecolor='black')
            plt.xlabel('Radius of Gyration (Å)')
            plt.ylabel('Frequency')
            plt.title('Radius of Gyration Distribution')
            
            plt.subplot(2, 3, 5)
            plt.scatter(features_df['radius_of_gyration'], features_df['end_to_end_distance'])
            plt.xlabel('Radius of Gyration (Å)')
            plt.ylabel('End-to-End Distance (Å)')
            plt.title('RoG vs. End-to-End Distance')
            
            plt.subplot(2, 3, 6)
            plt.hist(features_df['end_to_end_distance'], bins=15, alpha=0.7, edgecolor='black')
            plt.xlabel('End-to-End Distance (Å)')
            plt.ylabel('Frequency')
            plt.title('End-to-End Distance Distribution')
            
            plt.tight_layout()
            plt.savefig(EDA_OUTPUT_DIR / 'structural_features.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Structural features analysis complete (analyzed {len(structural_features)} structures)")
    
    def analyze_split_distribution(self):
        """Analyze how clean PDBs are distributed across train/val/test splits."""
        print("Analyzing split distribution...")
        
        # Map clean PDBs to split assignments
        split_distribution = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
        clean_in_splits = {'train': [], 'val': [], 'test': [], 'unknown': []}
        
        # The DAS data should contain split information
        # This is a placeholder - adapt based on actual DAS data structure
        try:
            for split_name in ['train', 'val', 'test']:
                if split_name in self.das_data:
                    split_items = self.das_data[split_name]
                    for item in split_items:
                        # Extract PDB identifier from item
                        pdb_id = None
                        if isinstance(item, dict) and 'id_list' in item:
                            pdb_id = item['id_list'][0] if item['id_list'] else None
                        elif hasattr(item, 'id_list'):
                            pdb_id = item.id_list[0] if item.id_list else None
                        
                        if pdb_id and pdb_id in self.clean_pdb_names:
                            split_distribution[split_name] += 1
                            clean_in_splits[split_name].append(pdb_id)
            
            # Count unknowns
            known_pdbs = set(clean_in_splits['train'] + clean_in_splits['val'] + clean_in_splits['test'])
            unknown_pdbs = [pdb for pdb in self.clean_pdb_names if pdb not in known_pdbs]
            split_distribution['unknown'] = len(unknown_pdbs)
            clean_in_splits['unknown'] = unknown_pdbs
            
        except Exception as e:
            print(f"Could not analyze split distribution: {e}")
            split_distribution = {'total_clean': len(self.clean_pdb_names)}
        
        self.analysis_results['split_distribution'] = {
            'distribution': split_distribution,
            'clean_in_splits': clean_in_splits
        }
        
        # Plot split distribution
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        splits = list(split_distribution.keys())
        counts = list(split_distribution.values())
        plt.pie(counts, labels=splits, autopct='%1.1f%%', startangle=90)
        plt.title('Clean PDBs Distribution Across Splits')
        
        plt.subplot(1, 2, 2)
        plt.bar(splits, counts)
        plt.ylabel('Number of Clean PDBs')
        plt.title('Clean PDBs per Split')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(EDA_OUTPUT_DIR / 'split_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Split distribution:")
        for split, count in split_distribution.items():
            print(f"  {split}: {count}")
    
    def save_results(self):
        """Save all analysis results to JSON."""
        output_file = EDA_OUTPUT_DIR / 'eda_analysis_results.json'
        
        def convert_to_serializable(obj):
            """Recursively convert numpy types to native Python types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int_)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float_)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(self.analysis_results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nEDA results saved to: {output_file}")
    
    def generate_summary_report(self):
        """Generate a summary report of the EDA."""
        report_file = EDA_OUTPUT_DIR / 'eda_summary_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# RiboPO v2 Clean Dataset - Exploratory Data Analysis Report\n\n")
            f.write(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Clean PDBs**: {len(self.clean_pdb_names)}\n\n")
            
            # Sequence length summary
            if 'sequence_lengths' in self.analysis_results:
                lengths_stats = self.analysis_results['sequence_lengths']
                f.write("## Sequence Length Analysis\n\n")
                f.write(f"- **Mean Length**: {lengths_stats['mean']:.1f} nucleotides\n")
                f.write(f"- **Median Length**: {lengths_stats['median']:.1f} nucleotides\n")
                f.write(f"- **Standard Deviation**: {lengths_stats['std']:.1f}\n")
                f.write(f"- **Range**: {lengths_stats['min']} - {lengths_stats['max']} nucleotides\n")
                f.write(f"- **Quartiles**: {lengths_stats['quartiles']}\n\n")
            
            # Composition summary
            if 'nucleotide_composition' in self.analysis_results:
                comp_stats = self.analysis_results['nucleotide_composition']['global_percentages']
                f.write("## Nucleotide Composition Analysis\n\n")
                for base, percentage in comp_stats.items():
                    if base in ['A', 'U', 'G', 'C']:
                        f.write(f"- **{base}**: {percentage:.3f} ({percentage*100:.1f}%)\n")
                f.write("\n")
            
            # Split distribution summary
            if 'split_distribution' in self.analysis_results:
                split_stats = self.analysis_results['split_distribution']['distribution']
                f.write("## Dataset Split Distribution\n\n")
                for split, count in split_stats.items():
                    f.write(f"- **{split.capitalize()}**: {count} PDBs\n")
                f.write("\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `sequence_length_distribution.png`: Sequence length analysis plots\n")
            f.write("- `nucleotide_composition.png`: Nucleotide composition analysis\n")
            f.write("- `structural_features.png`: Structural feature analysis\n")
            f.write("- `split_distribution.png`: Dataset split distribution\n")
            f.write("- `eda_analysis_results.json`: Complete analysis results\n\n")
            
            f.write("## Key Insights for DPO Training\n\n")
            f.write("1. **Data Quality**: All 182 PDB files are free of gaps and missing residues\n")
            f.write("2. **Size Distribution**: Analyze if sequence length distribution is suitable for batch training\n")
            f.write("3. **Composition Balance**: Check if nucleotide composition is balanced across the dataset\n")
            f.write("4. **Split Balance**: Ensure train/val/test splits maintain representative distributions\n")
        
        print(f"EDA summary report saved to: {report_file}")

def main():
    """Run complete EDA analysis."""
    print("=== RiboPO v2 Clean Dataset EDA ===")
    
    analyzer = CleanDatasetAnalyzer()
    
    # Run all analyses
    analyzer.analyze_sequence_lengths()
    analyzer.analyze_nucleotide_composition()
    analyzer.analyze_structural_features()
    analyzer.analyze_split_distribution()
    
    # Save results and generate report
    analyzer.save_results()
    analyzer.generate_summary_report()
    
    print(f"\n=== EDA Complete ===")
    print(f"Results saved to: {EDA_OUTPUT_DIR}")
    print(f"Total clean PDBs analyzed: {len(analyzer.clean_pdb_names)}")

if __name__ == "__main__":
    main()