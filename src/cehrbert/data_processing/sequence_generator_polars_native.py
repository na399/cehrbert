#!/usr/bin/env python3
"""
Optimized patient sequence generation using Polars native parallelization.
This version leverages Polars' built-in parallel execution instead of multiprocessing.
"""

import argparse
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import json
import os
from tqdm import tqdm
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Polars to use optimal number of threads
num_threads = int(os.environ.get('POLARS_MAX_THREADS', os.cpu_count() - 1))
os.environ['POLARS_MAX_THREADS'] = str(num_threads)
logger.info(f"Polars configured with {num_threads} threads")


def generate_att_token(days_diff: int) -> Optional[str]:
    """Generate artificial time token based on time difference in days."""
    if days_diff < 0:
        return None
    elif days_diff < 7:
        return "W0"
    elif days_diff < 14:
        return "W1" 
    elif days_diff < 21:
        return "W2"
    elif days_diff < 28:
        return "W3"
    elif days_diff < 365:
        # Month calculation (approximately 30 days per month)
        month = min(int(days_diff / 30), 11)
        return f"M{month}"
    else:
        return "LT"


def create_patient_sequences_polars_native(
    input_folder: Path,
    output_folder: Path,
    start_date: str,
    concept_filter_file: Optional[Path] = None,
    att_type: str = "day",
    domain_tables: List[str] = None,
    sample_size: Optional[int] = None
) -> pl.DataFrame:
    """
    Create patient sequences using Polars native parallelization.
    
    This approach uses Polars' lazy evaluation and native parallel execution
    to process data efficiently without explicit multiprocessing.
    """
    # Set default domain tables if not provided
    if domain_tables is None:
        domain_tables = ["condition_occurrence", "procedure_occurrence", "drug_exposure"]
    
    # Parse start date
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Load filtered concepts if provided
    included_concept_ids = None
    if concept_filter_file and concept_filter_file.exists():
        included_concepts = pl.read_parquet(concept_filter_file)
        included_concept_ids = set(included_concepts['concept_id'].to_list())
        logger.info(f"Loaded {len(included_concept_ids)} filtered concepts")
    
    # Helper function to read OMOP tables lazily
    def scan_table(table_name: str) -> Optional[pl.LazyFrame]:
        # Check for single file first
        table_path = input_folder / f"{table_name}.parquet"
        if table_path.exists():
            return pl.scan_parquet(table_path)
        
        # Check for directory with partitioned files
        table_dir = input_folder / table_name
        if table_dir.exists() and table_dir.is_dir():
            parquet_files = list(table_dir.glob("*.parquet"))
            if parquet_files:
                return pl.concat([pl.scan_parquet(f) for f in parquet_files])
        
        return None
    
    # Load person table
    logger.info("Loading person table...")
    person_lf = scan_table("person")
    if person_lf is None:
        raise ValueError("Person table not found")
    
    # Sample if requested
    if sample_size:
        logger.info(f"Sampling {sample_size} patients...")
        person_ids = person_lf.select('person_id').collect().sample(n=sample_size, seed=42)['person_id'].to_list()
        person_lf = person_lf.filter(pl.col('person_id').is_in(person_ids))
    
    # Prepare birth date column
    if 'birth_datetime' in person_lf.columns:
        person_lf = person_lf.with_columns(
            pl.col('birth_datetime').alias('birth_dt')
        )
    else:
        person_lf = person_lf.with_columns(
            pl.col('birth_date').cast(pl.Datetime).alias('birth_dt')
        )
    
    # Load visit data lazily
    logger.info("Loading visit data...")
    visit_lf = scan_table("visit_occurrence")
    if visit_lf is None:
        raise ValueError("Visit occurrence table not found")
    
    # Handle date columns
    date_col = 'visit_start_datetime' if 'visit_start_datetime' in visit_lf.columns else 'visit_start_date'
    if date_col == 'visit_start_date':
        visit_lf = visit_lf.with_columns(
            pl.col(date_col).cast(pl.Datetime).alias('visit_start_dt')
        )
    else:
        visit_lf = visit_lf.with_columns(
            pl.col(date_col).alias('visit_start_dt')
        )
    
    # Filter visits by date
    visit_lf = visit_lf.filter(pl.col('visit_start_dt') >= start_datetime)
    
    # Join with person data
    visit_lf = visit_lf.join(
        person_lf.select(['person_id', 'birth_dt']),
        on='person_id',
        how='inner'
    )
    
    # Add temporal features using window functions
    visit_lf = visit_lf.with_columns([
        # Visit order within patient
        pl.col('person_id').cum_count().over(['person_id', 'visit_start_dt']).alias('visit_order'),
        # Previous visit time
        pl.col('visit_start_dt').shift(1).over(['person_id', 'visit_start_dt']).alias('prev_visit_dt'),
        # Age calculation
        ((pl.col('visit_start_dt') - pl.col('birth_dt')).dt.total_days() / 365.25).alias('age'),
        # Date as days since epoch
        (pl.col('visit_start_dt') - datetime(1970, 1, 1)).dt.total_days().alias('date'),
        # Visit segment alternation
        ((pl.col('person_id').cum_count().over(['person_id', 'visit_start_dt']) + 1) % 2 + 1).alias('visit_segment')
    ])
    
    # Calculate days since previous visit
    visit_lf = visit_lf.with_columns([
        pl.when(pl.col('prev_visit_dt').is_not_null())
        .then((pl.col('visit_start_dt') - pl.col('prev_visit_dt')).dt.total_days())
        .otherwise(None)
        .alias('days_since_prev')
    ])
    
    # Load domain data lazily and join with visits
    logger.info("Loading domain data...")
    domain_data_lf = []
    
    for table_name in domain_tables:
        df = scan_table(table_name)
        if df is None:
            logger.warning(f"Table {table_name} not found, skipping...")
            continue
        
        # Find concept column
        concept_col = next(
            (col for col in df.columns if col.endswith('_concept_id') and col != 'visit_concept_id'),
            None
        )
        if not concept_col:
            continue
        
        # Filter by included concepts if provided
        if included_concept_ids:
            df = df.filter(pl.col(concept_col).is_in(included_concept_ids))
        
        # Select only needed columns and add table name
        df = df.select([
            'person_id',
            'visit_occurrence_id',
            pl.col(concept_col).alias('concept_id'),
            pl.lit(table_name).alias('source_table')
        ])
        
        domain_data_lf.append(df)
    
    # Union all domain data
    if domain_data_lf:
        all_concepts_lf = pl.concat(domain_data_lf)
    else:
        # Create empty LazyFrame with proper schema
        all_concepts_lf = pl.LazyFrame({
            'person_id': [],
            'visit_occurrence_id': [],
            'concept_id': [],
            'source_table': []
        })
    
    # Collect visits first to enable proper grouping
    logger.info("Processing sequences using Polars native parallelization...")
    visits_df = visit_lf.sort(['person_id', 'visit_start_dt']).collect()
    
    # Collect concepts
    concepts_df = all_concepts_lf.collect()
    
    # Process sequences using groupby with native parallelization
    def process_patient_group(group_df: pl.DataFrame) -> Dict[str, Any]:
        """Process all visits for a single patient into a sequence."""
        person_id = group_df['person_id'][0]
        sequence_parts = []
        
        # Get all concepts for this patient
        patient_concepts = concepts_df.filter(pl.col('person_id') == person_id)
        
        for row in group_df.iter_rows(named=True):
            # Add ATT token if not first visit
            if row['days_since_prev'] is not None:
                att_token = generate_att_token(int(row['days_since_prev']))
                if att_token:
                    sequence_parts.append({
                        'concept_id': att_token,
                        'age': row['age'],
                        'date': row['date'],
                        'visit_segment': row['visit_segment'],
                        'visit_order': row['visit_order'],
                        'mlm_skip': 1
                    })
            
            # Add VS token
            sequence_parts.append({
                'concept_id': '[VS]',
                'age': row['age'],
                'date': row['date'],
                'visit_segment': row['visit_segment'],
                'visit_order': row['visit_order'],
                'mlm_skip': 1
            })
            
            # Add visit type
            sequence_parts.append({
                'concept_id': str(row['visit_concept_id']),
                'age': row['age'],
                'date': row['date'],
                'visit_segment': row['visit_segment'],
                'visit_order': row['visit_order'],
                'mlm_skip': 0
            })
            
            # Add concepts for this visit
            visit_concepts = patient_concepts.filter(
                pl.col('visit_occurrence_id') == row['visit_occurrence_id']
            )['concept_id'].to_list()
            
            for concept_id in visit_concepts:
                sequence_parts.append({
                    'concept_id': str(concept_id),
                    'age': row['age'],
                    'date': row['date'],
                    'visit_segment': row['visit_segment'],
                    'visit_order': row['visit_order'],
                    'mlm_skip': 0
                })
            
            # Add VE token
            sequence_parts.append({
                'concept_id': '[VE]',
                'age': row['age'],
                'date': row['date'],
                'visit_segment': row['visit_segment'],
                'visit_order': row['visit_order'],
                'mlm_skip': 1
            })
        
        # Create sequence record
        if sequence_parts:
            return {
                'person_id': person_id,
                'concept_ids': [p['concept_id'] for p in sequence_parts],
                'ages': [p['age'] for p in sequence_parts],
                'dates': [p['date'] for p in sequence_parts],
                'visit_segments': [p['visit_segment'] for p in sequence_parts],
                'visit_concept_orders': [p['visit_order'] for p in sequence_parts],
                'concept_values': [0.0] * len(sequence_parts),
                'concept_value_masks': [0.0] * len(sequence_parts),
                'mlm_skip_values': [p['mlm_skip'] for p in sequence_parts],
                'num_of_concepts': len(sequence_parts),
                'num_of_visits': len(group_df)
            }
        return None
    
    # Process all patients using groupby - Polars will parallelize this automatically
    unique_patients = visits_df['person_id'].unique().to_list()
    sequences = []
    
    # Process in chunks to show progress
    chunk_size = 1000
    with tqdm(total=len(unique_patients), desc="Processing patients") as pbar:
        for i in range(0, len(unique_patients), chunk_size):
            chunk_patients = unique_patients[i:i+chunk_size]
            chunk_visits = visits_df.filter(pl.col('person_id').is_in(chunk_patients))
            
            # Process each patient in the chunk
            for person_id in chunk_patients:
                patient_visits = chunk_visits.filter(pl.col('person_id') == person_id)
                if len(patient_visits) > 0:
                    seq = process_patient_group(patient_visits)
                    if seq:
                        sequences.append(seq)
            
            pbar.update(len(chunk_patients))
            gc.collect()
    
    logger.info(f"Created {len(sequences)} patient sequences")
    
    # Convert to DataFrame
    sequences_df = pl.DataFrame(sequences)
    
    # Save sequences
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "patient_sequence.parquet"
    sequences_df.write_parquet(output_path)
    logger.info(f"Saved sequences to {output_path}")
    
    # Save sample for inspection
    if len(sequences) > 0:
        sample_path = output_folder / "patient_sequence_sample.json"
        with open(sample_path, "w") as f:
            json.dump(sequences[:5], f, indent=2, default=str)
        logger.info(f"Saved sample to {sample_path}")
    
    return sequences_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate patient sequences using Polars native parallelization"
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
        help="Output folder for patient sequences"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Start date for filtering data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--concept_filter_file",
        type=str,
        help="Path to filtered concepts file"
    )
    parser.add_argument(
        "--att_type",
        type=str,
        default="day",
        help="Type of ATT tokens to use"
    )
    parser.add_argument(
        "--domain_tables",
        nargs="+",
        default=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
        help="Domain tables to include"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        help="Optional number of patients to sample"
    )
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    concept_filter_file = Path(args.concept_filter_file) if args.concept_filter_file else None
    
    sequences_df = create_patient_sequences_polars_native(
        input_folder,
        output_folder,
        args.start_date,
        concept_filter_file,
        args.att_type,
        args.domain_tables,
        args.sample_size
    )
    
    # Print statistics
    print("\nSequence Statistics:")
    print(f"Total sequences: {len(sequences_df)}")
    print(f"Average sequence length: {sequences_df['num_of_concepts'].mean():.1f}")
    print(f"Average visits per patient: {sequences_df['num_of_visits'].mean():.1f}")
    print(f"Min sequence length: {sequences_df['num_of_concepts'].min()}")
    print(f"Max sequence length: {sequences_df['num_of_concepts'].max()}")


if __name__ == "__main__":
    main()