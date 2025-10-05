#!/usr/bin/env python3
"""
Evaluate baseline RNA design models (rdesign, rifold, ridiff) using the comprehensive evaluation pipeline.

This script processes different baseline model output formats and evaluates them using
the same 12-metric pipeline as trained models for fair comparison.
"""

import os
import sys
import argparse
import glob
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from multiround.eval_baseline import BaselineEvaluator
from types import SimpleNamespace
import yaml


def load_config(config_path: str):
    """Load evaluation configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to namespace for easier access
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d
    
    return dict_to_namespace(config)


def evaluate_individual_fasta_model(model_name: str, model_dir: str, output_dir: str, config):
    """
    Evaluate models with individual FASTA files per structure (rdesign, rifold).
    
    Args:
        model_name: Name of the model (e.g., 'rdesign', 'rifold')
        model_dir: Directory containing individual FASTA files
        output_dir: Output directory for results
        config: Evaluation configuration
    """
    print(f"\n🧬 Evaluating {model_name} model...")
    print(f"   Model directory: {model_dir}")
    print(f"   Output directory: {output_dir}")
    
    # Find all FASTA files
    fasta_files = glob.glob(os.path.join(model_dir, "*.fasta"))
    print(f"   Found {len(fasta_files)} FASTA files")
    
    if not fasta_files:
        print(f"❌ No FASTA files found in {model_dir}")
        return None
    
    # Create a single consolidated FASTA file for evaluation
    consolidated_fasta = os.path.join(output_dir, f"{model_name}_consolidated_sequences.fasta")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(consolidated_fasta, 'w') as outfile:
        processed_count = 0
        for fasta_file in sorted(fasta_files):
            # Apply limit if specified
            if config.limit and processed_count >= config.limit:
                print(f"   Limiting to first {config.limit} structures for testing")
                break
                
            # Extract structure ID from filename
            structure_id = os.path.basename(fasta_file).replace('.fasta', '')
            
            # Handle RhoDesign's _without2d suffix
            if structure_id.endswith('_without2d'):
                structure_id = structure_id.replace('_without2d', '')
            
            with open(fasta_file, 'r') as infile:
                lines = infile.readlines()
                if len(lines) >= 2:
                    # Write with standardized header
                    outfile.write(f">{structure_id}\n")
                    outfile.write(lines[1])  # sequence line
                    processed_count += 1
    
    print(f"   Created consolidated FASTA: {consolidated_fasta}")
    
    # Run evaluation
    try:
        evaluator = BaselineEvaluator(config)
        results = evaluator.evaluate_baseline_sequences(
            sequences_file=consolidated_fasta,
            model_name=model_name,
            output_dir=output_dir
        )
        
        print(f"✅ {model_name} evaluation completed successfully")
        print(f"   Evaluated {results.get('n_structures', 0)} structures")
        print(f"   Mean TM-score: {results.get('tm_score_mean', 0):.3f}")
        print(f"   Mean sequence recovery: {results.get('sequence_recovery_mean', 0):.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ {model_name} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_ridiff_model(model_dir: str, output_dir: str, config, n_samples_per_structure: int = 3):
    """
    Evaluate ridiff model with single FASTA file containing multiple predictions per structure.
    
    Args:
        model_dir: Directory containing ridiff seq.fasta file
        output_dir: Output directory for results
        config: Evaluation configuration
        n_samples_per_structure: Number of top predictions to use per structure
    """
    model_name = "ridiff"
    print(f"\n🧬 Evaluating {model_name} model...")
    print(f"   Model directory: {model_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Using top {n_samples_per_structure} predictions per structure")
    
    seq_fasta = os.path.join(model_dir, "seq.fasta")
    if not os.path.exists(seq_fasta):
        print(f"❌ seq.fasta not found in {model_dir}")
        return None
    
    # Load test dataset structure IDs to filter ridiff structures  
    # Use the same test set as BaselineEvaluator for consistency
    test_structures = set()
    test_ids_file = "data/das_split_raw_data/test_set_structure_ids.txt"
    if os.path.exists(test_ids_file):
        with open(test_ids_file, 'r') as f:
            for line in f:
                structure_id = line.strip()
                if structure_id:
                    test_structures.add(structure_id)
        print(f"   Found {len(test_structures)} structures in test dataset")
    else:
        print(f"⚠️ Test dataset not found at {test_ids_file}, will process all ridiff structures")
    
    # Parse ridiff format and create consolidated FASTA with top N predictions per structure
    consolidated_fasta = os.path.join(output_dir, f"{model_name}_consolidated_sequences.fasta")
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse the ridiff file to extract top N predictions for each structure
    current_structure = None
    current_predictions = []
    structure_sequences = {}
    
    with open(seq_fasta, 'r') as infile:
        lines = infile.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('>'):
                if line.startswith('>Original_'):
                    # Save previous structure's top predictions if any
                    if current_structure and current_predictions and current_structure in test_structures:
                        # Sort by confidence score (highest first) and take top N
                        top_predictions = sorted(current_predictions, key=lambda x: x[1], reverse=True)[:n_samples_per_structure]
                        structure_sequences[current_structure] = top_predictions
                    
                    # Start new structure
                    current_structure = line[10:]  # Remove '>Original_'
                    current_predictions = []
                    
                    # Skip original sequence (we DON'T want the native sequences)
                    i += 2  # Skip header and sequence line
                    continue
                    
                elif line.startswith('>seq') and current_structure and current_structure in test_structures:
                    # Only collect predictions for structures in test dataset
                    # Parse prediction with confidence score
                    header_parts = line.split('--')
                    if len(header_parts) == 2:
                        try:
                            confidence = float(header_parts[1])
                            if i + 1 < len(lines):
                                sequence = lines[i + 1].strip()
                                current_predictions.append((sequence, confidence))
                        except ValueError:
                            print(f"   ⚠️ Could not parse confidence score: {header_parts[1]}")
                    i += 2  # Skip header and sequence line
                    continue
            
            i += 1
    
    # Handle last structure
    if current_structure and current_predictions and current_structure in test_structures:
        top_predictions = sorted(current_predictions, key=lambda x: x[1], reverse=True)[:n_samples_per_structure]
        structure_sequences[current_structure] = top_predictions
    
    # Write consolidated FASTA with multiple samples per structure
    with open(consolidated_fasta, 'w') as outfile:
        processed_structures = 0
        total_sequences = 0
        
        for structure_id, predictions in sorted(structure_sequences.items()):
            # Apply structure limit if specified
            if config.limit and processed_structures >= config.limit:
                print(f"   Limiting to first {config.limit} structures for testing")
                break
                
            # Write all top predictions for this structure
            # Use the base structure_id for all samples so they match the test dataset
            for i, (sequence, confidence) in enumerate(predictions):
                outfile.write(f">{structure_id} sample_{i+1} confidence={confidence:.4f}\n{sequence}\n")
                total_sequences += 1
            
            processed_structures += 1
    
    print(f"   Created consolidated FASTA: {consolidated_fasta}")
    print(f"   Processed {processed_structures} structures with {total_sequences} total sequences")
    print(f"   Average {total_sequences/processed_structures:.1f} sequences per structure")
    
    # Run evaluation
    try:
        evaluator = BaselineEvaluator(config)
        results = evaluator.evaluate_baseline_sequences(
            sequences_file=consolidated_fasta,
            model_name=model_name,
            output_dir=output_dir
        )
        
        print(f"✅ {model_name} evaluation completed successfully")
        print(f"   Evaluated {results.get('n_structures', 0)} structures")
        print(f"   Mean TM-score: {results.get('tm_score_mean', 0):.3f}")
        print(f"   Mean sequence recovery: {results.get('sequence_recovery_mean', 0):.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ {model_name} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_ribodiffusion_model(model_dir: str, output_dir: str, config):
    """
    Evaluate RiboDiffusion model with individual FASTA files containing multiple predictions per structure.
    Extract only the first designed sequence (seq 1) for evaluation.
    
    Args:
        model_dir: Directory containing RiboDiffusion individual FASTA files
        output_dir: Output directory for results
        config: Evaluation configuration
    """
    model_name = "ribodiffusion"
    print(f"\n🧬 Evaluating {model_name} model...")
    print(f"   Model directory: {model_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Using first designed sequence only")
    
    # Find all FASTA files
    fasta_files = glob.glob(os.path.join(model_dir, "*.fasta"))
    print(f"   Found {len(fasta_files)} FASTA files")
    
    if not fasta_files:
        print(f"❌ No FASTA files found in {model_dir}")
        return None
    
    # Load test dataset structure IDs for filtering
    test_structures = set()
    test_ids_file = "data/das_split_raw_data/test_set_structure_ids.txt"
    if os.path.exists(test_ids_file):
        with open(test_ids_file, 'r') as f:
            for line in f:
                structure_id = line.strip()
                if structure_id:
                    test_structures.add(structure_id)
        print(f"   Found {len(test_structures)} structures in test dataset")
    else:
        print(f"⚠️ Test dataset not found at {test_ids_file}, will process all ribodiffusion structures")
    
    # Create a consolidated FASTA file for evaluation
    consolidated_fasta = os.path.join(output_dir, f"{model_name}_consolidated_sequences.fasta")
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    with open(consolidated_fasta, 'w') as outfile:
        for fasta_file in sorted(fasta_files):
            # Apply limit if specified
            if config.limit and processed_count >= config.limit:
                print(f"   Limiting to first {config.limit} structures for testing")
                break
            
            # Extract structure ID from filename (e.g., 1CSL_1_B-A.fasta -> 1CSL_1_B)
            filename = os.path.basename(fasta_file).replace('.fasta', '')
            # Handle multi-chain structures: 1CSL_1_B-A -> 1CSL_1_B
            structure_id = filename.split('-')[0] if '-' in filename else filename
            
            # Skip if not in test dataset
            if test_structures and structure_id not in test_structures:
                continue
            
            with open(fasta_file, 'r') as infile:
                lines = infile.readlines()
                
                # Find the first designed sequence (seq 1)
                found_seq1 = False
                seq1_sequence = None
                
                for i, line in enumerate(lines):
                    if line.startswith('>') and 'seq 1' in line:
                        # Found seq 1, get the sequence on next line
                        if i + 1 < len(lines):
                            seq1_sequence = lines[i + 1].strip()
                            found_seq1 = True
                            break
                
                if found_seq1 and seq1_sequence:
                    # Write with standardized header
                    outfile.write(f">{structure_id}\n{seq1_sequence}\n")
                    processed_count += 1
                else:
                    print(f"   ⚠️ No 'seq 1' found in {fasta_file}")
    
    print(f"   Created consolidated FASTA: {consolidated_fasta}")
    print(f"   Processed {processed_count} structures")
    
    # Run evaluation
    try:
        evaluator = BaselineEvaluator(config)
        results = evaluator.evaluate_baseline_sequences(
            sequences_file=consolidated_fasta,
            model_name=model_name,
            output_dir=output_dir
        )
        
        print(f"✅ {model_name} evaluation completed successfully")
        print(f"   Evaluated {results.get('n_structures', 0)} structures")
        print(f"   Mean TM-score: {results.get('sc_tm', 0):.3f}")
        print(f"   Mean sequence recovery: {results.get('recovery', 0):.3f}")
        print(f"   Mean RMSD: {results.get('sc_rmsd', 0):.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ {model_name} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_r3design_model(model_dir: str, output_dir: str, config):
    """
    Evaluate r3design model from CSV format with separate chain entries.
    Concatenate chains per structure to match multi-chain native structures.
    
    Args:
        model_dir: Directory containing r3design.csv file
        output_dir: Output directory for results
        config: Evaluation configuration
    """
    model_name = "r3design"
    print(f"\n🧬 Evaluating {model_name} model...")
    print(f"   Model directory: {model_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Concatenating chains per structure for multi-chain evaluation")
    
    csv_file = os.path.join(model_dir, "r3design.csv")
    if not os.path.exists(csv_file):
        print(f"❌ r3design.csv not found in {model_dir}")
        return None
    
    # Load CSV data
    try:
        df = pd.read_csv(csv_file)
        print(f"   Loaded CSV with {len(df)} entries")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return None
    
    # Load test dataset structure IDs for filtering
    test_structures = set()
    test_ids_file = "data/das_split_raw_data/test_set_structure_ids.txt"
    if os.path.exists(test_ids_file):
        with open(test_ids_file, 'r') as f:
            for line in f:
                structure_id = line.strip()
                if structure_id:
                    test_structures.add(structure_id)
        print(f"   Found {len(test_structures)} structures in test dataset")
    else:
        print(f"⚠️ Test dataset not found at {test_ids_file}, will process all r3design structures")
    
    # Group chains by PDB file path, not by PDB ID
    # This correctly handles multi-chain structures vs separate single-chain structures
    structure_sequences = {}
    
    for _, row in df.iterrows():
        pdb_id = row['pdb_id']
        chain = row['chain']
        generated_seq = row['generate_seq']
        pdb_path = row['pdb_path']
        
        # Extract the actual PDB filename from the path
        # e.g., "/content/das_split_raw_data/das_split_raw_pdb/1CSL_1_B-A.pdb" -> "1CSL_1_B-A"
        pdb_filename = os.path.basename(pdb_path).replace('.pdb', '')
        
        # Map PDB filename to test structure ID format
        # e.g., "1CSL_1_B-A" -> "1CSL_1_B" (since 1CSL_1_B is in test set)
        if '-' in pdb_filename:
            structure_id = pdb_filename.split('-')[0]  # Take base part before first hyphen
        else:
            structure_id = pdb_filename
        
        if structure_id not in structure_sequences:
            structure_sequences[structure_id] = {}
        
        structure_sequences[structure_id][chain] = generated_seq
    
    # Create consolidated FASTA file with concatenated sequences
    consolidated_fasta = os.path.join(output_dir, f"{model_name}_consolidated_sequences.fasta")
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    with open(consolidated_fasta, 'w') as outfile:
        for structure_id, chains in sorted(structure_sequences.items()):
            # Apply limit if specified
            if config.limit and processed_count >= config.limit:
                print(f"   Limiting to first {config.limit} structures for testing")
                break
            
            # Check if this structure is in the test dataset
            if test_structures and structure_id not in test_structures:
                continue
            
            # For multi-chain structures (multiple chains for same structure_id), concatenate
            # For single-chain structures, just use the single chain
            if len(chains) > 1:
                # Multi-chain: concatenate in alphabetical order (A, B, C, D, ...)
                sorted_chains = sorted(chains.keys())
                concatenated_sequence = ''.join(chains[chain] for chain in sorted_chains)
                print(f"   Multi-chain {structure_id}: chains {sorted_chains} -> {len(concatenated_sequence)} nt")
            else:
                # Single-chain: use the single sequence
                chain_letter = list(chains.keys())[0]
                concatenated_sequence = chains[chain_letter]
                print(f"   Single-chain {structure_id}: chain {chain_letter} -> {len(concatenated_sequence)} nt")
            
            # Write sequence
            outfile.write(f">{structure_id}\n{concatenated_sequence}\n")
            processed_count += 1
    
    print(f"   Created consolidated FASTA: {consolidated_fasta}")
    print(f"   Processed {processed_count} individual chain sequences")
    
    # Run evaluation
    try:
        evaluator = BaselineEvaluator(config)
        results = evaluator.evaluate_baseline_sequences(
            sequences_file=consolidated_fasta,
            model_name=model_name,
            output_dir=output_dir
        )
        
        print(f"✅ {model_name} evaluation completed successfully")
        print(f"   Evaluated {results.get('n_structures', 0)} structures")
        print(f"   Mean TM-score: {results.get('sc_tm', 0):.3f}")
        print(f"   Mean sequence recovery: {results.get('recovery', 0):.3f}")
        print(f"   Mean RMSD: {results.get('sc_rmsd', 0):.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ {model_name} evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline RNA design models")
    parser.add_argument("--models", nargs="+", default=["rdesign", "rifold", "ridiff", "ribodiffusion", "r3design", "rhodesign"],
                       help="Models to evaluate (default: all)")
    parser.add_argument("--config", default="multiround/config/evaluation/baseline_eval.yaml",
                       help="Evaluation configuration file")
    parser.add_argument("--baselines_dir", default="data/baselines",
                       help="Directory containing baseline model outputs")
    parser.add_argument("--output_dir", default="data/baselines/baseline_eval",
                       help="Output directory for evaluation results")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit evaluation to first N structures for faster testing")
    parser.add_argument("--ridiff_samples", type=int, default=3,
                       help="Number of top predictions per structure for ridiff model")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.join(project_root, args.config)
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Add limit parameter to config
    config.limit = args.limit
    
    print("🚀 Starting baseline model evaluation...")
    print(f"   Models to evaluate: {args.models}")
    print(f"   Configuration: {config_path}")
    print(f"   Baselines directory: {args.baselines_dir}")
    print(f"   Output directory: {args.output_dir}")
    if args.limit:
        print(f"   Limiting to first {args.limit} structures for testing")
    
    results = {}
    
    # Evaluate each model
    for model_name in args.models:
        model_dir = os.path.join(args.baselines_dir, model_name)
        model_output_dir = os.path.join(args.output_dir, model_name)
        
        if not os.path.exists(model_dir):
            print(f"⚠️ Model directory not found: {model_dir}, skipping...")
            continue
        
        if model_name == "ridiff":
            result = evaluate_ridiff_model(model_dir, model_output_dir, config, 
                                          n_samples_per_structure=args.ridiff_samples)
        elif model_name == "ribodiffusion":
            result = evaluate_ribodiffusion_model(model_dir, model_output_dir, config)
        elif model_name == "r3design":
            result = evaluate_r3design_model(model_dir, model_output_dir, config)
        elif model_name in ["rdesign", "rifold", "rhodesign"]:
            result = evaluate_individual_fasta_model(model_name, model_dir, model_output_dir, config)
        else:
            print(f"⚠️ Unknown model type: {model_name}, skipping...")
            continue
        
        if result:
            results[model_name] = result
    
    # Generate comparison summary
    if results:
        print("\n📊 Evaluation Summary:")
        print("=" * 80)
        print(f"{'Model':<10} {'Structures':<10} {'Recovery':<10} {'TM-score':<10} {'RMSD':<10}")
        print("-" * 80)
        
        for model_name, result in results.items():
            n_struct = result.get('n_structures', 0)
            recovery = result.get('recovery', 0)
            tm_score = result.get('sc_tm', 0)
            rmsd = result.get('sc_rmsd', 0)
            
            print(f"{model_name:<10} {n_struct:<10} {recovery:<10.3f} {tm_score:<10.3f} {rmsd:<10.3f}")
        
        # Save comparison summary
        summary_file = os.path.join(args.output_dir, "baseline_comparison_summary.json")
        os.makedirs(args.output_dir, exist_ok=True)
        
        import json
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Comparison summary saved: {summary_file}")
        print("🎉 Baseline evaluation completed!")
    else:
        print("❌ No models were successfully evaluated")
        sys.exit(1)


if __name__ == "__main__":
    main()