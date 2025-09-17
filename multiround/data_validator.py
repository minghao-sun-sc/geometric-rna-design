# multiround/data_validator.py
"""
Enhanced data validation and quality control for multiround training.
Addresses length mismatches, data quality issues, and provides robust fallbacks.
"""

from dpo.env_bootstrap import bootstrap_env; bootstrap_env()

import os
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np

class MultiRoundDataValidator:
    """
    Enhanced data validation for multiround training.
    
    Features:
    - Length mismatch detection and correction
    - Data quality assessment
    - Robust fallback mechanisms
    - Validation caching for performance
    - Detailed reporting and logging
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.validation_cache = {}
        self.stats = defaultdict(int)
        self.problematic_ids = set()
        
        # Validation settings (safe handling of missing validation config)
        validation_cfg = getattr(cfg, 'validation', None)
        self.max_attempts = getattr(validation_cfg, 'max_attempts', 10) if validation_cfg else 10
        self.strict_validation = getattr(validation_cfg, 'strict_validation', False) if validation_cfg else False
        self.cache_validation = getattr(validation_cfg, 'cache_validation', True) if validation_cfg else True
        
        # Output directory for validation reports
        self.validation_dir = os.path.join(
            getattr(cfg.multiround, 'output_root', 'runs/multiround'),
            'validation_reports'
        )
        os.makedirs(self.validation_dir, exist_ok=True)
    
    def validate_pair_file(self, pair_file_path: str, split_name: str = "unknown") -> Dict:
        """
        Validate a preference pair file for quality issues.
        
        Returns:
            dict: Validation report with statistics and recommendations
        """
        print(f"🔍 Validating {split_name} pairs: {pair_file_path}")
        
        if not os.path.exists(pair_file_path):
            return {
                'valid': False,
                'error': 'File not found',
                'total_pairs': 0,
                'valid_pairs': 0,
                'issues': {'file_not_found': 1}
            }
        
        # Check cache first
        file_hash = self._get_file_hash(pair_file_path)
        cache_key = f"{pair_file_path}_{file_hash}"
        
        if self.cache_validation and cache_key in self.validation_cache:
            print(f"✅ Using cached validation for {split_name}")
            return self.validation_cache[cache_key]
        
        # Perform validation
        validation_result = self._validate_pair_file_content(pair_file_path, split_name)
        
        # Cache result
        if self.cache_validation:
            self.validation_cache[cache_key] = validation_result
        
        # Save validation report
        self._save_validation_report(validation_result, split_name)
        
        return validation_result
    
    def _validate_pair_file_content(self, pair_file_path: str, split_name: str) -> Dict:
        """Validate the actual content of a pair file."""
        issues = defaultdict(int)
        valid_pairs = 0
        total_pairs = 0
        length_mismatches = []
        duplicates = set()
        pair_hashes = set()
        
        try:
            with open(pair_file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    total_pairs += 1
                    
                    try:
                        pair_data = json.loads(line.strip())
                        
                        # Check required fields (allow both 'id' and 'pdb_file' as identifier)
                        required_fields = ['winner_seq', 'loser_seq']
                        id_fields = ['id', 'pdb_file']
                        
                        missing_fields = [field for field in required_fields if field not in pair_data]
                        
                        # Check for at least one ID field
                        if not any(field in pair_data for field in id_fields):
                            missing_fields.append('id_or_pdb_file')
                        
                        if missing_fields:
                            issues['missing_fields'] += 1
                            continue
                        
                        # Get identifier (prefer 'id' over 'pdb_file')
                        pair_id = pair_data.get('id', pair_data.get('pdb_file', f'line_{line_num}'))
                        
                        # Check for duplicates
                        pair_content = f"{pair_data['winner_seq']}|{pair_data['loser_seq']}"
                        pair_hash = hashlib.md5(pair_content.encode()).hexdigest()
                        
                        if pair_hash in pair_hashes:
                            issues['duplicates'] += 1
                            duplicates.add(pair_id)
                            continue
                        pair_hashes.add(pair_hash)
                        
                        # Check sequence lengths
                        winner_len = len(pair_data['winner_seq'])
                        loser_len = len(pair_data['loser_seq'])
                        
                        if winner_len != loser_len:
                            issues['length_mismatch_winner_loser'] += 1
                            length_mismatches.append({
                                'id': pair_id,
                                'line': line_num,
                                'winner_len': winner_len,
                                'loser_len': loser_len
                            })
                            continue
                        
                        # Check for valid nucleotides
                        valid_nucleotides = set('ACGU')
                        winner_nucleotides = set(pair_data['winner_seq'].upper())
                        loser_nucleotides = set(pair_data['loser_seq'].upper())
                        
                        if not winner_nucleotides.issubset(valid_nucleotides):
                            issues['invalid_nucleotides_winner'] += 1
                            continue
                        
                        if not loser_nucleotides.issubset(valid_nucleotides):
                            issues['invalid_nucleotides_loser'] += 1
                            continue
                        
                        # Check sequence length reasonableness
                        if winner_len < 10 or winner_len > 1000:  # Reasonable RNA length bounds
                            issues['unreasonable_length'] += 1
                            continue
                        
                        # If we get here, the pair is valid
                        valid_pairs += 1
                        
                    except json.JSONDecodeError:
                        issues['json_decode_error'] += 1
                        continue
                    except KeyError as e:
                        issues['key_error'] += 1
                        continue
                    except Exception as e:
                        issues['other_error'] += 1
                        continue
        
        except Exception as e:
            return {
                'valid': False,
                'error': f'Failed to read file: {str(e)}',
                'total_pairs': 0,
                'valid_pairs': 0,
                'issues': {'file_read_error': 1}
            }
        
        # Compile validation results
        valid_ratio = valid_pairs / max(total_pairs, 1)
        is_valid = valid_ratio >= 0.95  # Require 95% valid pairs
        
        result = {
            'valid': is_valid,
            'total_pairs': total_pairs,
            'valid_pairs': valid_pairs,
            'valid_ratio': valid_ratio,
            'issues': dict(issues),
            'length_mismatches': length_mismatches[:20],  # Sample for debugging
            'duplicates_sample': list(duplicates)[:10],
            'recommendations': self._generate_recommendations(issues, valid_ratio)
        }
        
        return result
    
    def _generate_recommendations(self, issues: Dict, valid_ratio: float) -> List[str]:
        """Generate recommendations based on validation issues."""
        recommendations = []
        
        if valid_ratio < 0.95:
            recommendations.append(f"Data quality low ({valid_ratio:.1%}). Consider re-filtering.")
        
        if issues.get('length_mismatch_winner_loser', 0) > 0:
            recommendations.append("Found winner/loser length mismatches. Check pair generation process.")
        
        if issues.get('duplicates', 0) > 0:
            recommendations.append("Found duplicate pairs. Consider deduplication.")
        
        if issues.get('invalid_nucleotides_winner', 0) + issues.get('invalid_nucleotides_loser', 0) > 0:
            recommendations.append("Found invalid nucleotides. Check sequence cleaning.")
        
        if issues.get('json_decode_error', 0) > 0:
            recommendations.append("Found JSON parsing errors. Check file format.")
        
        if not recommendations:
            recommendations.append("Data quality looks good.")
        
        return recommendations
    
    def validate_pair_config(self, pair_config) -> Dict:
        """Validate a complete pair configuration (train/val/test)."""
        print(f"🔍 Validating pair configuration: {pair_config.description}")
        
        results = {}
        overall_valid = True
        
        for split, path in [('train', pair_config.train_path), 
                           ('val', pair_config.val_path), 
                           ('test', pair_config.test_path)]:
            
            result = self.validate_pair_file(path, split)
            results[split] = result
            
            if not result['valid']:
                overall_valid = False
                self.problematic_ids.update(
                    mismatch['id'] for mismatch in result.get('length_mismatches', [])
                )
        
        summary = {
            'overall_valid': overall_valid,
            'split_results': results,
            'total_pairs': sum(r['total_pairs'] for r in results.values()),
            'total_valid_pairs': sum(r['valid_pairs'] for r in results.values()),
            'problematic_ids': list(self.problematic_ids),
            'recommendations': self._generate_config_recommendations(results)
        }
        
        print(f"📊 Validation summary: {summary['total_valid_pairs']}/{summary['total_pairs']} pairs valid")
        
        return summary
    
    def _generate_config_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations for a complete pair configuration."""
        recommendations = []
        
        total_pairs = sum(r['total_pairs'] for r in results.values())
        total_valid = sum(r['valid_pairs'] for r in results.values())
        
        if total_pairs == 0:
            recommendations.append("No pairs found. Check file paths.")
            return recommendations
        
        overall_ratio = total_valid / total_pairs
        
        if overall_ratio < 0.9:
            recommendations.append(f"Overall data quality low ({overall_ratio:.1%}). Consider re-filtering all splits.")
        
        # Check for inconsistencies between splits
        train_issues = len(results['train'].get('issues', {}))
        val_issues = len(results['val'].get('issues', {}))
        test_issues = len(results['test'].get('issues', {}))
        
        if max(train_issues, val_issues, test_issues) - min(train_issues, val_issues, test_issues) > 2:
            recommendations.append("Quality inconsistency between splits. Check filtering process.")
        
        # Check for reasonable split sizes
        train_pairs = results['train']['total_pairs']
        val_pairs = results['val']['total_pairs']
        test_pairs = results['test']['total_pairs']
        
        if train_pairs < 100:
            recommendations.append("Very small training set. Consider using more data.")
        
        if val_pairs < 10:
            recommendations.append("Very small validation set. May lead to unreliable validation.")
        
        if test_pairs < 10:
            recommendations.append("Very small test set. May lead to unreliable evaluation.")
        
        return recommendations
    
    def get_problematic_ids(self) -> Set[str]:
        """Get set of problematic IDs that should be filtered out."""
        return self.problematic_ids.copy()
    
    def create_length_mismatch_filter_script(self, output_path: str):
        """Create a script to filter out problematic length mismatches."""
        problematic_ids = self.get_problematic_ids()
        
        filter_script = f'''#!/usr/bin/env python3
"""
Auto-generated filter script for length mismatch issues.
Generated by MultiRoundDataValidator.

Usage: python this_script.py input.jsonl output.jsonl
"""

import json
import sys

PROBLEMATIC_IDS = {{
{chr(10).join(f'    "{id_}",' for id_ in sorted(problematic_ids))}
}}

def filter_pairs(input_path, output_path):
    filtered_count = 0
    total_count = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            total_count += 1
            try:
                pair = json.loads(line.strip())
                if pair.get('id') not in PROBLEMATIC_IDS:
                    outfile.write(line)
                else:
                    filtered_count += 1
            except json.JSONDecodeError:
                filtered_count += 1  # Skip malformed lines
    
    print(f"Filtered {{filtered_count}}/{{total_count}} pairs")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_script.py input.jsonl output.jsonl")
        sys.exit(1)
    
    filter_pairs(sys.argv[1], sys.argv[2])
'''
        
        with open(output_path, 'w') as f:
            f.write(filter_script)
        
        os.chmod(output_path, 0o755)  # Make executable
        print(f"📝 Created filter script: {output_path}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for caching."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    
    def _save_validation_report(self, validation_result: Dict, split_name: str):
        """Save detailed validation report to file."""
        report_path = os.path.join(self.validation_dir, f"{split_name}_validation_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(validation_result, f, indent=2)
        
        print(f"📄 Validation report saved: {report_path}")
    
    def generate_summary_report(self) -> Dict:
        """Generate a summary report of all validations performed."""
        return {
            'total_validations': len(self.validation_cache),
            'total_problematic_ids': len(self.problematic_ids),
            'cache_size': len(self.validation_cache),
            'validation_stats': dict(self.stats),
            'problematic_ids_sample': list(self.problematic_ids)[:20]
        }