#!/usr/bin/env python3
"""
Filter concepts based on minimum patient count using Polars.
Replaces the PySpark-based concept filtering from cehrbert_data.
"""

import argparse
import polars as pl
from pathlib import Path
from typing import List, Optional
import logging
import os
import multiprocessing
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Polars to use optimal number of threads
num_threads = int(os.environ.get('POLARS_MAX_THREADS', max(1, multiprocessing.cpu_count() - 1)))
os.environ['POLARS_MAX_THREADS'] = str(num_threads)
logger.info(f"Polars configured with {num_threads} threads")


def get_concept_statistics(
    input_folder: Path,
    domain_tables: List[str],
    min_patients: int = 100
) -> pl.DataFrame:
    """
    Calculate concept statistics across domain tables.
    
    Args:
        input_folder: Path to folder containing OMOP parquet files
        domain_tables: List of domain tables to process
        min_patients: Minimum number of patients for concept inclusion
    
    Returns:
        DataFrame with concept statistics
    """
    all_concepts = []
    
    for table_name in tqdm(domain_tables, desc="Processing domain tables"):
        # Check for single file first
        table_path = input_folder / f"{table_name}.parquet"
        
        # If single file doesn't exist, check for directory with partitioned files
        if not table_path.exists():
            table_dir = input_folder / table_name
            if table_dir.exists() and table_dir.is_dir():
                logger.info(f"Processing partitioned {table_name}...")
                # Read all parquet files in the directory
                parquet_files = list(table_dir.glob("*.parquet"))
                if not parquet_files:
                    logger.warning(f"No parquet files found in {table_dir}, skipping...")
                    continue
                try:
                    # Read and concatenate all partitions
                    df = pl.concat([pl.read_parquet(f) for f in parquet_files])
                except Exception as e:
                    logger.warning(f"Error reading partitioned files in {table_dir}: {e}")
                    continue
            else:
                logger.warning(f"Table {table_name} not found, skipping...")
                continue
        else:
            logger.info(f"Processing {table_name}...")
            try:
                # Read the single domain table file
                df = pl.read_parquet(table_path)
            except Exception as e:
                logger.warning(f"Error reading {table_path}: {e}")
                continue
        
        # Get the concept column name (usually *_concept_id)
        concept_col = None
        for col in df.columns:
            if col.endswith('_concept_id') and col != 'visit_concept_id':
                concept_col = col
                break
        
        if not concept_col:
            logger.warning(f"No concept column found in {table_name}")
            continue
        
        # Calculate concept statistics
        concept_stats = (
            df.select([concept_col, 'person_id'])
            .rename({concept_col: 'concept_id'})
            .group_by('concept_id')
            .agg([
                pl.n_unique('person_id').alias('patient_count'),
                pl.len().alias('occurrence_count')
            ])
            .with_columns(pl.lit(table_name).alias('domain'))
        )
        
        all_concepts.append(concept_stats)
    
    # Combine all concept statistics
    if not all_concepts:
        raise ValueError("No concept statistics were generated")
    
    combined_stats = pl.concat(all_concepts)
    
    # Filter by minimum patient count
    filtered_stats = combined_stats.filter(pl.col('patient_count') >= min_patients)
    
    logger.info(f"Total concepts: {len(combined_stats)}")
    logger.info(f"Concepts after filtering (>= {min_patients} patients): {len(filtered_stats)}")
    
    return filtered_stats


def save_included_concepts(
    concept_stats: pl.DataFrame,
    output_folder: Path
) -> None:
    """Save the filtered concept list."""
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Save full statistics
    stats_path = output_folder / "concept_statistics.parquet"
    concept_stats.write_parquet(stats_path)
    logger.info(f"Saved concept statistics to {stats_path}")
    
    # Save just the concept IDs for filtering
    included_concepts = concept_stats.select('concept_id').unique()
    concepts_path = output_folder / "included_concepts.parquet"
    included_concepts.write_parquet(concepts_path)
    logger.info(f"Saved {len(included_concepts)} included concepts to {concepts_path}")
    
    # Also save as CSV for easy inspection
    csv_path = output_folder / "included_concepts.csv"
    included_concepts.write_csv(csv_path)
    logger.info(f"Saved included concepts as CSV to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter OMOP concepts based on minimum patient count"
    )
    parser.add_argument(
        "--input_folder", "-i",
        type=str,
        required=True,
        help="Input folder containing OMOP parquet files"
    )
    parser.add_argument(
        "--output_folder", "-o",
        type=str,
        required=True,
        help="Output folder for filtered concepts"
    )
    parser.add_argument(
        "--min_patients",
        type=int,
        default=100,
        help="Minimum number of patients for concept inclusion (default: 100)"
    )
    parser.add_argument(
        "--domain_tables",
        nargs="+",
        default=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
        help="Domain tables to process"
    )
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    
    if not input_folder.exists():
        raise ValueError(f"Input folder {input_folder} does not exist")
    
    # Get concept statistics
    concept_stats = get_concept_statistics(
        input_folder,
        args.domain_tables,
        args.min_patients
    )
    
    # Save results
    save_included_concepts(concept_stats, output_folder)
    
    # Print summary statistics
    print("\nConcept Statistics Summary:")
    print(concept_stats.group_by('domain').agg([
        pl.len().alias('concept_count'),
        pl.sum('patient_count').alias('total_patients'),
        pl.sum('occurrence_count').alias('total_occurrences')
    ]))


if __name__ == "__main__":
    main()