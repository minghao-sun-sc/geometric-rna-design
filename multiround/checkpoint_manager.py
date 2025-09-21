# multiround/checkpoint_manager.py
"""
Enhanced Checkpoint Management for Multi-Round RiboPO Training

This module provides comprehensive checkpoint management including:
- Intelligent checkpoint selection and cleanup
- Metadata tracking and persistence
- Performance-based retention policies
- Integration with EMA models
- Comprehensive logging and monitoring

Key features:
- Automatic cleanup of old checkpoints based on configurable policies
- Rich metadata storage for reproducibility
- Integration with distribution analysis
- Support for EMA checkpoint management
- Advanced checkpoint selection strategies
"""

from __future__ import annotations

import os
import json
import shutil
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


@dataclass
class CheckpointMetadata:
    """Comprehensive metadata for a checkpoint."""
    
    # Basic identification
    round_num: int
    epoch: int
    step: int
    timestamp: str
    
    # Model information
    model_state_size: int
    optimizer_state_size: int
    has_ema: bool = False
    ema_num_updates: int = 0
    
    # Performance metrics
    eval_metrics: Dict = None
    primary_metric: float = 0.0
    tiebreaker_metric: float = 0.0
    
    # Training context
    temperature: float = 0.5
    n_samples_eval: int = 8
    learning_rate: float = 0.0
    
    # Selection info
    is_best: bool = False
    selection_reason: str = ""
    selection_score: float = 0.0
    
    # File info
    file_path: str = ""
    file_size_mb: float = 0.0
    
    def __post_init__(self):
        if self.eval_metrics is None:
            self.eval_metrics = {}
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class EnhancedCheckpointManager:
    """
    Enhanced checkpoint management with intelligent selection and cleanup.
    
    Features:
    - Performance-based retention policies
    - Automatic cleanup of outdated checkpoints  
    - Rich metadata tracking
    - Integration with EMA models
    - Advanced selection strategies
    """
    
    def __init__(self, cfg, output_root: str):
        self.cfg = cfg
        self.output_root = Path(output_root)
        
        # Configuration
        self.checkpoint_cfg = getattr(cfg, 'checkpoints', None)
        self.save_frequency = getattr(self.checkpoint_cfg, 'save_frequency', 50) if self.checkpoint_cfg else 50
        self.keep_best_n = getattr(self.checkpoint_cfg, 'keep_best_n', 5) if self.checkpoint_cfg else 5
        self.max_checkpoints_per_round = getattr(self.checkpoint_cfg, 'max_per_round', 10) if self.checkpoint_cfg else 10
        
        # Tracking
        self.checkpoint_registry = {}  # round -> list of CheckpointMetadata
        self.best_checkpoints = []  # sorted list of best checkpoints across rounds
        self.metadata_file = self.output_root / "checkpoint_registry.json"
        
        # Load existing registry if available
        self._load_checkpoint_registry()
        
        print(f"🗂️ Enhanced checkpoint manager initialized")
        print(f"   Save frequency: every {self.save_frequency} steps")
        print(f"   Keep best: {self.keep_best_n} checkpoints")
        print(f"   Max per round: {self.max_checkpoints_per_round}")
    
    def save_step_checkpoint(self, round_num: int, epoch: int, step: int, 
                           model, optimizer, scheduler, eval_metrics: Optional[Dict] = None,
                           ema_manager = None) -> CheckpointMetadata:
        """
        Save a step checkpoint with comprehensive metadata.
        
        Args:
            round_num: Current training round
            epoch: Current epoch
            step: Current step
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            eval_metrics: Optional evaluation metrics
            ema_manager: Optional EMA manager
            
        Returns:
            CheckpointMetadata for the saved checkpoint
        """
        # Create checkpoint directory
        round_dir = self.output_root / f"round_{round_num:02d}"
        checkpoints_dir = round_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint filename
        checkpoint_name = f"step_{step}_epoch_{epoch}.pt"
        checkpoint_path = checkpoints_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'step': step,
            'epoch': epoch,
            'round': round_num,
            'config': self.cfg,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add EMA state if available
        ema_num_updates = 0
        if ema_manager and ema_manager.enabled:
            checkpoint_data['ema_state_dict'] = ema_manager.ema.state_dict()
            ema_num_updates = ema_manager.ema.num_updates
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Calculate file size
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
        # Create metadata
        metadata = CheckpointMetadata(
            round_num=round_num,
            epoch=epoch,
            step=step,
            timestamp=datetime.now().isoformat(),
            model_state_size=len(model.state_dict()),
            optimizer_state_size=len(optimizer.state_dict()),
            has_ema=ema_manager.enabled if ema_manager else False,
            ema_num_updates=ema_num_updates,
            eval_metrics=eval_metrics or {},
            learning_rate=self._get_current_lr(optimizer),
            file_path=str(checkpoint_path),
            file_size_mb=file_size_mb
        )
        
        # Register checkpoint
        self._register_checkpoint(metadata)
        
        print(f"💾 Step checkpoint saved: {checkpoint_name} ({file_size_mb:.1f} MB)")
        return metadata
    
    def save_round_checkpoint(self, round_num: int, model, optimizer, scheduler,
                            eval_metrics: Dict, ema_manager = None) -> CheckpointMetadata:
        """
        Save end-of-round checkpoint with best model selection.
        
        Args:
            round_num: Current training round
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            eval_metrics: Evaluation metrics for this round
            ema_manager: Optional EMA manager
            
        Returns:
            CheckpointMetadata for the saved checkpoint
        """
        # Create checkpoint directory
        round_dir = self.output_root / f"round_{round_num:02d}"
        checkpoints_dir = round_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint filename
        checkpoint_name = f"round_{round_num}_final.pt"
        checkpoint_path = checkpoints_dir / checkpoint_name
        
        # Prepare checkpoint data  
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'step': getattr(model, 'global_step', 0),
            'epoch': round_num * 20,  # Approximate
            'round': round_num,
            'eval_metrics': eval_metrics,
            'config': self.cfg,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add EMA state if available
        ema_num_updates = 0
        if ema_manager and ema_manager.enabled:
            checkpoint_data['ema_state_dict'] = ema_manager.ema.state_dict()
            ema_num_updates = ema_manager.ema.num_updates
            
            # Also save EMA model checkpoint
            ema_checkpoint_name = f"round_{round_num}_ema.pt"
            ema_checkpoint_path = checkpoints_dir / ema_checkpoint_name
            ema_checkpoint_data = checkpoint_data.copy()
            ema_checkpoint_data['model_state_dict'] = ema_manager.get_ema_model_copy().state_dict()
            torch.save(ema_checkpoint_data, ema_checkpoint_path)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Calculate file size
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
        # Evaluate checkpoint quality
        primary_metric = eval_metrics.get('passk_tm_0.45_k8', eval_metrics.get('tm_mean', 0.0))
        tiebreaker_metric = eval_metrics.get('mfe_mean', 0.0)
        
        # Create metadata
        metadata = CheckpointMetadata(
            round_num=round_num,
            epoch=round_num * 20,
            step=getattr(model, 'global_step', 0),
            timestamp=datetime.now().isoformat(),
            model_state_size=len(model.state_dict()),
            optimizer_state_size=len(optimizer.state_dict()),
            has_ema=ema_manager.enabled if ema_manager else False,
            ema_num_updates=ema_num_updates,
            eval_metrics=eval_metrics,
            primary_metric=primary_metric,
            tiebreaker_metric=tiebreaker_metric,
            learning_rate=self._get_current_lr(optimizer),
            file_path=str(checkpoint_path),
            file_size_mb=file_size_mb
        )
        
        # Determine if this is a new best checkpoint
        is_best, selection_reason = self._evaluate_checkpoint_quality(metadata)
        metadata.is_best = is_best
        metadata.selection_reason = selection_reason
        metadata.selection_score = self._calculate_selection_score(metadata)
        
        # Register checkpoint
        self._register_checkpoint(metadata)
        
        # Update best checkpoints tracking
        if is_best:
            self._update_best_checkpoints(metadata)
        
        # Cleanup old checkpoints if needed
        self._cleanup_old_checkpoints(round_num)
        
        print(f"💾 Round checkpoint saved: {checkpoint_name} ({file_size_mb:.1f} MB)")
        if is_best:
            print(f"⭐ New best checkpoint! {selection_reason}")
        
        return metadata
    
    def select_best_checkpoint_from_round(self, round_num: int) -> Optional[CheckpointMetadata]:
        """
        Select the best checkpoint from a specific round.
        
        Args:
            round_num: Round number to select from
            
        Returns:
            Best CheckpointMetadata from that round, or None
        """
        if round_num not in self.checkpoint_registry:
            return None
        
        round_checkpoints = self.checkpoint_registry[round_num]
        if not round_checkpoints:
            return None
        
        # Sort by selection score (higher is better)
        best_checkpoint = max(round_checkpoints, key=lambda x: x.selection_score)
        
        print(f"🎯 Best checkpoint from round {round_num}: {best_checkpoint.file_path}")
        print(f"   Selection score: {best_checkpoint.selection_score:.4f}")
        print(f"   Primary metric: {best_checkpoint.primary_metric:.4f}")
        
        return best_checkpoint
    
    def get_best_overall_checkpoint(self) -> Optional[CheckpointMetadata]:
        """
        Get the best checkpoint across all rounds.
        
        Returns:
            Best CheckpointMetadata overall, or None
        """
        if not self.best_checkpoints:
            return None
        
        return self.best_checkpoints[0]  # Already sorted by quality
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer, scheduler = None,
                       ema_manager = None) -> Dict:
        """
        Load a checkpoint with comprehensive restoration.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load into
            optimizer: Optimizer to load into
            scheduler: Optional scheduler to load into
            ema_manager: Optional EMA manager to load into
            
        Returns:
            Dictionary with loaded metadata
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint data
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint_data['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # Load scheduler state if available
        if scheduler and 'scheduler_state_dict' in checkpoint_data and checkpoint_data['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        # Load EMA state if available
        if ema_manager and 'ema_state_dict' in checkpoint_data:
            ema_manager.load_ema_checkpoint(checkpoint_path)
        
        # Extract metadata
        metadata = {
            'step': checkpoint_data.get('step', 0),
            'epoch': checkpoint_data.get('epoch', 0),
            'round': checkpoint_data.get('round', 0),
            'eval_metrics': checkpoint_data.get('eval_metrics', {}),
            'timestamp': checkpoint_data.get('timestamp', ''),
        }
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"   Step: {metadata['step']}, Epoch: {metadata['epoch']}, Round: {metadata['round']}")
        
        return metadata
    
    def cleanup_round_checkpoints(self, round_num: int, keep_best: bool = True):
        """
        Clean up checkpoints from a specific round.
        
        Args:
            round_num: Round to clean up
            keep_best: Whether to keep the best checkpoint from the round
        """
        if round_num not in self.checkpoint_registry:
            return
        
        round_checkpoints = self.checkpoint_registry[round_num]
        best_checkpoint = None
        
        if keep_best and round_checkpoints:
            best_checkpoint = max(round_checkpoints, key=lambda x: x.selection_score)
        
        checkpoints_removed = 0
        for checkpoint in round_checkpoints:
            if keep_best and checkpoint == best_checkpoint:
                continue
            
            # Remove checkpoint file
            if os.path.exists(checkpoint.file_path):
                os.unlink(checkpoint.file_path)
                checkpoints_removed += 1
        
        # Update registry
        if keep_best and best_checkpoint:
            self.checkpoint_registry[round_num] = [best_checkpoint]
        else:
            self.checkpoint_registry[round_num] = []
        
        print(f"🧹 Cleaned up {checkpoints_removed} checkpoints from round {round_num}")
        if keep_best and best_checkpoint:
            print(f"   Kept best checkpoint: {os.path.basename(best_checkpoint.file_path)}")
    
    def get_checkpoint_statistics(self) -> Dict:
        """
        Get comprehensive statistics about checkpoint storage and performance.
        
        Returns:
            Dictionary with checkpoint statistics
        """
        total_checkpoints = sum(len(checkpoints) for checkpoints in self.checkpoint_registry.values())
        total_size_mb = 0
        best_metrics = {}
        
        for round_checkpoints in self.checkpoint_registry.values():
            for checkpoint in round_checkpoints:
                total_size_mb += checkpoint.file_size_mb
                
                # Track best metrics
                if checkpoint.primary_metric > best_metrics.get('best_primary_metric', 0):
                    best_metrics['best_primary_metric'] = checkpoint.primary_metric
                    best_metrics['best_primary_round'] = checkpoint.round_num
        
        stats = {
            'total_checkpoints': total_checkpoints,
            'total_size_mb': total_size_mb,
            'total_size_gb': total_size_mb / 1024,
            'rounds_with_checkpoints': len(self.checkpoint_registry),
            'best_checkpoints_tracked': len(self.best_checkpoints),
            'average_checkpoint_size_mb': total_size_mb / max(total_checkpoints, 1),
            **best_metrics
        }
        
        return stats
    
    def generate_checkpoint_report(self) -> str:
        """
        Generate a comprehensive checkpoint management report.
        
        Returns:
            Path to the generated report
        """
        stats = self.get_checkpoint_statistics()
        
        report_lines = [
            "# Checkpoint Management Report",
            f"",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            "## Summary Statistics",
            f"",
            f"- **Total Checkpoints:** {stats['total_checkpoints']}",
            f"- **Storage Used:** {stats['total_size_gb']:.2f} GB ({stats['total_size_mb']:.1f} MB)",
            f"- **Rounds with Checkpoints:** {stats['rounds_with_checkpoints']}",
            f"- **Best Checkpoints Tracked:** {stats['best_checkpoints_tracked']}",
            f"- **Average Checkpoint Size:** {stats['average_checkpoint_size_mb']:.1f} MB",
            f"",
            "## Best Checkpoints",
            f""
        ]
        
        # Add best checkpoints info
        for i, checkpoint in enumerate(self.best_checkpoints[:10]):  # Top 10
            report_lines.extend([
                f"### Rank {i+1}: Round {checkpoint.round_num}",
                f"",
                f"- **Primary Metric:** {checkpoint.primary_metric:.4f}",
                f"- **Selection Score:** {checkpoint.selection_score:.4f}",
                f"- **File:** {os.path.basename(checkpoint.file_path)}",
                f"- **Size:** {checkpoint.file_size_mb:.1f} MB",
                f"- **EMA:** {'Yes' if checkpoint.has_ema else 'No'}",
                f"- **Selection Reason:** {checkpoint.selection_reason}",
                f""
            ])
        
        # Add round-by-round breakdown
        report_lines.extend([
            "## Round-by-Round Breakdown",
            f""
        ])
        
        for round_num in sorted(self.checkpoint_registry.keys()):
            checkpoints = self.checkpoint_registry[round_num]
            if checkpoints:
                best_in_round = max(checkpoints, key=lambda x: x.selection_score)
                report_lines.extend([
                    f"### Round {round_num}",
                    f"",
                    f"- **Checkpoints:** {len(checkpoints)}",
                    f"- **Best Primary Metric:** {best_in_round.primary_metric:.4f}",
                    f"- **Total Size:** {sum(c.file_size_mb for c in checkpoints):.1f} MB",
                    f""
                ])
        
        # Save report
        report_path = self.output_root / "checkpoint_management_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📝 Checkpoint report generated: {report_path}")
        return str(report_path)
    
    def _register_checkpoint(self, metadata: CheckpointMetadata):
        """Register a checkpoint in the tracking system."""
        if metadata.round_num not in self.checkpoint_registry:
            self.checkpoint_registry[metadata.round_num] = []
        
        self.checkpoint_registry[metadata.round_num].append(metadata)
        self._save_checkpoint_registry()
    
    def _evaluate_checkpoint_quality(self, metadata: CheckpointMetadata) -> Tuple[bool, str]:
        """
        Evaluate if a checkpoint is better than current best.
        
        Returns:
            (is_best, selection_reason)
        """
        if not self.best_checkpoints:
            return True, "First checkpoint"
        
        current_best = self.best_checkpoints[0]
        
        # Primary comparison: pass@k score (higher is better)
        if metadata.primary_metric > current_best.primary_metric:
            return True, f"Higher primary metric ({metadata.primary_metric:.4f} > {current_best.primary_metric:.4f})"
        
        # Secondary comparison: MFE (lower is better) if primary metrics are close
        if abs(metadata.primary_metric - current_best.primary_metric) < 1e-6:
            if metadata.tiebreaker_metric < current_best.tiebreaker_metric:
                return True, f"Equal primary metric, better tiebreaker ({metadata.tiebreaker_metric:.4f} < {current_best.tiebreaker_metric:.4f})"
        
        return False, f"Not better than current best (primary: {metadata.primary_metric:.4f} vs {current_best.primary_metric:.4f})"
    
    def _calculate_selection_score(self, metadata: CheckpointMetadata) -> float:
        """
        Calculate a composite selection score for checkpoint ranking.
        
        Returns:
            Composite score (higher is better)
        """
        # Weighted combination of metrics
        primary_weight = 1.0
        tiebreaker_weight = 0.1  # MFE contribution (normalized)
        
        # Normalize MFE (more negative is better, so invert)
        normalized_mfe = -metadata.tiebreaker_metric / 50.0  # Typical MFE range
        
        score = (primary_weight * metadata.primary_metric + 
                tiebreaker_weight * normalized_mfe)
        
        return score
    
    def _update_best_checkpoints(self, metadata: CheckpointMetadata):
        """Update the best checkpoints tracking list."""
        # Add new checkpoint
        self.best_checkpoints.append(metadata)
        
        # Sort by selection score (descending)
        self.best_checkpoints.sort(key=lambda x: x.selection_score, reverse=True)
        
        # Keep only top N
        if len(self.best_checkpoints) > self.keep_best_n:
            # Remove excess checkpoints from tracking and optionally from disk
            excess_checkpoints = self.best_checkpoints[self.keep_best_n:]
            for checkpoint in excess_checkpoints:
                print(f"🗑️ Removing excess checkpoint: {os.path.basename(checkpoint.file_path)}")
                if os.path.exists(checkpoint.file_path):
                    os.unlink(checkpoint.file_path)
            
            self.best_checkpoints = self.best_checkpoints[:self.keep_best_n]
    
    def _cleanup_old_checkpoints(self, current_round: int):
        """Clean up old step checkpoints from the current round."""
        if current_round not in self.checkpoint_registry:
            return
        
        round_checkpoints = self.checkpoint_registry[current_round]
        step_checkpoints = [cp for cp in round_checkpoints if 'step_' in os.path.basename(cp.file_path)]
        
        if len(step_checkpoints) > self.max_checkpoints_per_round:
            # Sort by step (keep most recent)
            step_checkpoints.sort(key=lambda x: x.step, reverse=True)
            
            # Remove excess step checkpoints
            excess_checkpoints = step_checkpoints[self.max_checkpoints_per_round:]
            for checkpoint in excess_checkpoints:
                if os.path.exists(checkpoint.file_path):
                    os.unlink(checkpoint.file_path)
                round_checkpoints.remove(checkpoint)
            
            print(f"🧹 Cleaned up {len(excess_checkpoints)} old step checkpoints from round {current_round}")
    
    def _get_current_lr(self, optimizer) -> float:
        """Get current learning rate from optimizer."""
        try:
            return optimizer.param_groups[0]['lr']
        except:
            return 0.0
    
    def _save_checkpoint_registry(self):
        """Save checkpoint registry to disk."""
        try:
            registry_data = {}
            for round_num, checkpoints in self.checkpoint_registry.items():
                registry_data[str(round_num)] = [asdict(cp) for cp in checkpoints]
            
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    'registry': registry_data,
                    'best_checkpoints': [asdict(cp) for cp in self.best_checkpoints],
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save checkpoint registry: {e}")
    
    def _load_checkpoint_registry(self):
        """Load checkpoint registry from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                # Load registry
                for round_str, checkpoints_data in data.get('registry', {}).items():
                    round_num = int(round_str)
                    self.checkpoint_registry[round_num] = [
                        CheckpointMetadata(**cp_data) for cp_data in checkpoints_data
                    ]
                
                # Load best checkpoints
                self.best_checkpoints = [
                    CheckpointMetadata(**cp_data) for cp_data in data.get('best_checkpoints', [])
                ]
                
                print(f"📂 Loaded checkpoint registry: {len(self.checkpoint_registry)} rounds, {len(self.best_checkpoints)} best checkpoints")
                
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint registry: {e}")
            self.checkpoint_registry = {}
            self.best_checkpoints = []


def create_enhanced_checkpoint_manager(cfg, output_root: str) -> EnhancedCheckpointManager:
    """
    Factory function to create an EnhancedCheckpointManager.
    
    Args:
        cfg: Configuration object
        output_root: Root directory for outputs
        
    Returns:
        EnhancedCheckpointManager instance
    """
    return EnhancedCheckpointManager(cfg, output_root)