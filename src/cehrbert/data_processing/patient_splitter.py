#!/usr/bin/env python3
"""
Split patient sequences into train/validation/test sets using Polars.
Replaces the PySpark-based patient splitting from cehrbert_data.
"""

import argparse
import polars as pl
from pathlib import Path
import logging
import random
from typing import Tuple, Optional, Dict
import os
import multiprocessing
from tqdm import tqdm
import shutil
try:
    from .ensure_compatibility import ensure_huggingface_compatibility
except ImportError:
    from ensure_compatibility import ensure_huggingface_compatibility

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Polars to use optimal number of threads
num_threads = int(os.environ.get('POLARS_MAX_THREADS', max(1, multiprocessing.cpu_count() - 1)))
os.environ['POLARS_MAX_THREADS'] = str(num_threads)
logger.info(f"Polars configured with {num_threads} threads")


def get_patient_split_assignments(
    sequence_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> dict:
    """
    Get patient ID assignments for train/validation/test splits without loading full data.
    
    Args:
        sequence_path: Path to patient sequences parquet file
        train_ratio: Ratio of patients for training
        val_ratio: Ratio of patients for validation
        test_ratio: Ratio of patients for testing
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping patient IDs to split assignments
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Train, val, and test ratios must sum to 1.0, got {total_ratio}")
    
    # Get unique patient IDs using lazy evaluation
    logger.info("Getting unique patient IDs...")
    patient_ids = (
        pl.scan_parquet(sequence_path)
        .select('person_id')
        .unique()
        .collect()['person_id']
        .to_list()
    )
    n_patients = len(patient_ids)
    
    logger.info(f"Total patients: {n_patients}")
    
    # Shuffle patient IDs
    random.seed(seed)
    random.shuffle(patient_ids)
    
    # Calculate split points
    train_end = int(n_patients * train_ratio)
    val_end = train_end + int(n_patients * val_ratio)
    
    # Create assignment dictionary
    assignments = {}
    
    # Assign train
    for pid in patient_ids[:train_end]:
        assignments[pid] = 'train'
    
    # Assign validation
    for pid in patient_ids[train_end:val_end]:
        assignments[pid] = 'validation'
    
    # Assign test
    for pid in patient_ids[val_end:]:
        assignments[pid] = 'test'
    
    logger.info(f"Train patients: {sum(1 for v in assignments.values() if v == 'train')}")
    logger.info(f"Validation patients: {sum(1 for v in assignments.values() if v == 'validation')}")
    logger.info(f"Test patients: {sum(1 for v in assignments.values() if v == 'test')}")
    
    return assignments


def save_splits_optimized(
    sequence_path: Path,
    assignments: dict,
    output_folder: Path,
    chunk_size: int = 100000
) -> None:
    """Save the train/validation/test splits using optimized streaming."""
    # Create output directories and temp directories
    splits = ['train', 'validation', 'test']
    output_dirs = {}
    temp_dirs = {}
    
    for split_name in splits:
        output_dirs[split_name] = output_folder / split_name
        temp_dirs[split_name] = output_folder / f"{split_name}_temp"
        output_dirs[split_name].mkdir(parents=True, exist_ok=True)
        temp_dirs[split_name].mkdir(parents=True, exist_ok=True)
    
    # Convert assignments to DataFrame for efficient joining
    assignment_df = pl.DataFrame({
        'person_id': list(assignments.keys()),
        'split': list(assignments.values())
    })
    
    # Initialize stats
    stats = {split: {'n_sequences': 0, 'total_concepts': 0, 'total_visits': 0} 
             for split in splits}
    
    # Get total rows for progress tracking
    total_rows = pl.scan_parquet(sequence_path).select(pl.len()).collect().item()
    
    logger.info(f"Processing {total_rows:,} sequences in chunks of {chunk_size:,}...")
    
    # Process data in chunks with progress bar
    chunk_idx = 0
    with tqdm(total=total_rows, desc="Processing sequences") as pbar:
        for offset in range(0, total_rows, chunk_size):
            # Read chunk
            chunk = pl.read_parquet(sequence_path, n_rows=chunk_size, row_index_offset=offset)
            
            # Vectorized join to add split assignments
            chunk = chunk.join(assignment_df, on='person_id', how='left')
            
            # Process all splits in one pass
            for split_name in splits:
                split_chunk = chunk.filter(pl.col('split') == split_name).drop('split')
                
                if len(split_chunk) > 0:
                    # Update stats efficiently
                    stats[split_name]['n_sequences'] += len(split_chunk)
                    stats[split_name]['total_concepts'] += split_chunk['num_of_concepts'].sum()
                    stats[split_name]['total_visits'] += split_chunk['num_of_visits'].sum()
                    
                    # Write to temporary file
                    temp_path = temp_dirs[split_name] / f"chunk_{chunk_idx:06d}.parquet"
                    split_chunk.write_parquet(temp_path)
            
            chunk_idx += 1
            pbar.update(min(chunk_size, total_rows - offset))
    
    # Combine chunks efficiently for each split
    logger.info("Combining chunks into final files...")
    for split_name in splits:
        temp_dir = temp_dirs[split_name]
        chunk_files = sorted(temp_dir.glob("chunk_*.parquet"))
        
        if chunk_files:
            # Use scan_parquet for efficient combining
            combined = pl.scan_parquet(chunk_files).collect()
            
            # Get unique patient count
            n_patients = combined['person_id'].n_unique()
            
            # Ensure HuggingFace compatibility
            combined = ensure_huggingface_compatibility(combined)
            
            # Write final file
            output_path = output_dirs[split_name] / "patient_sequence.parquet"
            combined.write_parquet(output_path)
            
            # Update stats
            n_sequences = stats[split_name]['n_sequences']
            if n_sequences > 0:
                avg_concepts = stats[split_name]['total_concepts'] / n_sequences
                avg_visits = stats[split_name]['total_visits'] / n_sequences
            else:
                avg_concepts = 0
                avg_visits = 0
            
            stats[split_name] = {
                "n_patients": n_patients,
                "n_sequences": n_sequences,
                "avg_sequence_length": float(avg_concepts),
                "avg_visits": float(avg_visits)
            }
            
            logger.info(f"Saved {n_sequences:,} {split_name} sequences from {n_patients:,} patients")
            
            # Clean up temp files
            shutil.rmtree(temp_dir)
    
    # Save split statistics
    import json
    stats_path = output_folder / "split_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved split statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Split patient sequences into train/validation/test sets"
    )
    parser.add_argument(
        "--input_folder", "-i",
        type=str,
        required=True,
        help="Input folder containing patient sequences"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of patients for training (default: 0.8)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of patients for validation (default: 0.1)"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio of patients for testing (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    if args.train_ratio + args.val_ratio + args.test_ratio != 1.0:
        # If only train_ratio is specified, calculate val and test
        if args.val_ratio == 0.1 and args.test_ratio == 0.1:
            remaining = 1.0 - args.train_ratio
            args.val_ratio = remaining / 2
            args.test_ratio = remaining / 2
            logger.info(f"Adjusted ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")
    
    input_folder = Path(args.input_folder)
    
    # Load patient sequences
    sequence_path = input_folder / "patient_sequence.parquet"
    if not sequence_path.exists():
        # Try looking in the current folder
        sequence_path = input_folder / "patient_sequences.parquet"
        if not sequence_path.exists():
            raise ValueError(f"Patient sequence file not found in {input_folder}")
    
    # Get patient assignments without loading full data
    assignments = get_patient_split_assignments(
        sequence_path,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    # Save splits using optimized streaming
    save_splits_optimized(sequence_path, assignments, input_folder)
    
    # Print summary from saved statistics
    import json
    stats_path = input_folder / "split_statistics.json"
    with open(stats_path, "r") as f:
        stats = json.load(f)
    
    print("\nSplit Summary:")
    print(f"Train: {stats['train']['n_sequences']} sequences from {stats['train']['n_patients']} patients")
    print(f"Validation: {stats['validation']['n_sequences']} sequences from {stats['validation']['n_patients']} patients")
    print(f"Test: {stats['test']['n_sequences']} sequences from {stats['test']['n_patients']} patients")


if __name__ == "__main__":
    main()