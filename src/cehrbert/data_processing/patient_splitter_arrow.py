#!/usr/bin/env python3
"""
Split Arrow-based patient sequences into train/validation/test sets.
Processes Arrow shards without loading all data into memory.
"""

import argparse
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from pathlib import Path
import logging
import random
from typing import Dict, List
import os
import json
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_patient_assignments_from_arrow(
    sequences_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[int, str]:
    """
    Get patient ID assignments from Arrow files without loading full data.
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Train, val, and test ratios must sum to 1.0, got {total_ratio}")
    
    # Get all arrow files
    arrow_files = sorted(sequences_dir.glob("*.arrow"))
    if not arrow_files:
        raise ValueError(f"No arrow files found in {sequences_dir}")
    
    logger.info(f"Found {len(arrow_files)} arrow files")
    
    # Collect unique patient IDs from all shards
    all_patient_ids = set()
    
    for arrow_file in tqdm(arrow_files, desc="Reading patient IDs"):
        # Read only the person_id column
        table = pa.ipc.open_file(arrow_file).read_all()
        person_ids = table.column('person_id').to_pylist()
        all_patient_ids.update(person_ids)
    
    # Convert to sorted list for reproducibility
    patient_ids = sorted(list(all_patient_ids))
    n_patients = len(patient_ids)
    logger.info(f"Total unique patients: {n_patients}")
    
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


def process_arrow_shard(
    arrow_file: Path,
    assignments: Dict[int, str],
    output_writers: Dict[str, pa.ipc.RecordBatchFileWriter],
    stats: Dict[str, Dict[str, any]]
) -> None:
    """
    Process a single Arrow shard and write to appropriate split files.
    """
    # Read the shard
    table = pa.ipc.open_file(arrow_file).read_all()
    
    # Get person_ids as numpy array for efficient lookup
    person_ids = table.column('person_id').to_numpy()
    
    # Create split assignments array
    split_assignments = np.array([assignments.get(pid, 'unknown') for pid in person_ids])
    
    # Process each split
    for split_name in ['train', 'validation', 'test']:
        # Create mask for this split
        mask = split_assignments == split_name
        
        if np.any(mask):
            # Filter table using mask
            split_table = table.filter(pa.array(mask))
            
            # Update stats
            n_sequences = len(split_table)
            stats[split_name]['n_sequences'] += n_sequences
            
            # Get split-specific person IDs
            split_person_ids = split_table.column('person_id').to_pylist()
            stats[split_name]['n_patients'].update(split_person_ids)
            
            # Sum concepts and visits
            stats[split_name]['total_concepts'] += pc.sum(split_table.column('num_of_concepts')).as_py()
            stats[split_name]['total_visits'] += pc.sum(split_table.column('num_of_visits')).as_py()
            
            # Write the filtered table directly (maintains schema)
            output_writers[split_name].write_table(split_table)


def split_arrow_sequences(
    input_folder: Path,
    output_folder: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> None:
    """
    Split Arrow-based sequences into train/validation/test sets.
    """
    # Find sequences directory
    sequences_dir = input_folder / "sequences"
    if not sequences_dir.exists():
        sequences_dir = input_folder
    
    # Get patient assignments
    assignments = get_patient_assignments_from_arrow(sequences_dir, train_ratio, val_ratio, test_ratio, seed)
    
    # Create output directories
    splits = ['train', 'validation', 'test']
    output_dirs = {}
    
    for split_name in splits:
        output_dirs[split_name] = output_folder / split_name
        output_dirs[split_name].mkdir(parents=True, exist_ok=True)
    
    # Initialize stats
    stats = {
        split: {
            'n_sequences': 0,
            'n_patients': set(),
            'total_concepts': 0,
            'total_visits': 0
        }
        for split in splits
    }
    
    # Get all arrow files
    arrow_files = sorted(sequences_dir.glob("*.arrow"))
    
    # Process each shard and write to split files
    logger.info("Processing Arrow shards...")
    
    # Open writers for each split
    output_files = {}
    output_writers = {}
    
    # Read schema from first file
    first_table = pa.ipc.open_file(arrow_files[0]).read_all()
    schema = first_table.schema
    
    for split_name in splits:
        output_path = output_dirs[split_name] / f"{split_name}_sequences.arrow"
        output_files[split_name] = pa.OSFile(str(output_path), 'wb')
        output_writers[split_name] = pa.ipc.RecordBatchFileWriter(output_files[split_name], schema)
    
    # Process each shard
    for arrow_file in tqdm(arrow_files, desc="Processing shards"):
        process_arrow_shard(arrow_file, assignments, output_writers, stats)
    
    # Close writers
    for writer in output_writers.values():
        writer.close()
    for file in output_files.values():
        file.close()
    
    # Calculate and save final statistics
    final_stats = {}
    for split_name in splits:
        n_patients = len(stats[split_name]['n_patients'])
        n_sequences = stats[split_name]['n_sequences']
        
        if n_sequences > 0:
            avg_concepts = stats[split_name]['total_concepts'] / n_sequences
            avg_visits = stats[split_name]['total_visits'] / n_sequences
        else:
            avg_concepts = 0
            avg_visits = 0
        
        final_stats[split_name] = {
            "n_patients": n_patients,
            "n_sequences": n_sequences,
            "avg_sequence_length": float(avg_concepts),
            "avg_visits": float(avg_visits)
        }
        
        logger.info(f"{split_name}: {n_sequences} sequences from {n_patients} patients")
    
    # Save statistics
    stats_path = output_folder / "split_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(final_stats, f, indent=2)
    logger.info(f"Saved split statistics to {stats_path}")
    
    # Save metadata for each split
    for split_name in splits:
        metadata = {
            "format": "arrow",
            "split": split_name,
            "n_sequences": final_stats[split_name]["n_sequences"],
            "n_patients": final_stats[split_name]["n_patients"],
            "created_from": str(input_folder)
        }
        
        metadata_path = output_dirs[split_name] / "dataset_info.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Split Arrow-based patient sequences into train/validation/test sets"
    )
    parser.add_argument(
        "--input_folder", "-i",
        type=str,
        required=True,
        help="Input folder containing Arrow sequences"
    )
    parser.add_argument(
        "--output_folder", "-o",
        type=str,
        required=True,
        help="Output folder for split datasets"
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
    output_folder = Path(args.output_folder)
    
    split_arrow_sequences(
        input_folder,
        output_folder,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    print("\nSplit complete!")


if __name__ == "__main__":
    main()