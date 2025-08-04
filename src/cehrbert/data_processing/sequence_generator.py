#!/usr/bin/env python3
"""
Optimized patient sequence generation from OMOP data using Polars.
Implements batch processing and vectorized operations for better performance.
"""

import argparse
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import json
import os
import multiprocessing
from tqdm import tqdm
import numpy as np
import gc
import psutil
from multiprocessing import Pool, cpu_count
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


def prepare_visit_staging(
    input_folder: Path,
    staging_folder: Path,
    start_date: str,
    person_ids: Optional[set] = None,
    num_partitions: int = 20
) -> Dict[str, Any]:
    """
    Pre-process visit data with temporal information and save to staging.
    Returns metadata about the staging files.
    """
    logger.info("Preparing visit staging data...")
    staging_folder.mkdir(parents=True, exist_ok=True)
    
    # Parse start date
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Helper to read OMOP tables
    def read_table(table_name: str) -> Optional[pl.DataFrame]:
        table_path = input_folder / f"{table_name}.parquet"
        if table_path.exists():
            return pl.scan_parquet(table_path).collect()
        
        table_dir = input_folder / table_name
        if table_dir.exists() and table_dir.is_dir():
            parquet_files = list(table_dir.glob("*.parquet"))
            if parquet_files:
                return pl.concat([pl.scan_parquet(f).collect() for f in parquet_files])
        return None
    
    # Load person table
    person_df = read_table("person")
    if person_df is None:
        raise ValueError("Person table not found")
    
    # Filter by person_ids if provided
    if person_ids:
        person_df = person_df.filter(pl.col('person_id').is_in(person_ids))
    
    # Ensure birth date is datetime
    if 'birth_datetime' in person_df.columns:
        person_df = person_df.with_columns(
            pl.col('birth_datetime').alias('birth_dt')
        )
    else:
        person_df = person_df.with_columns(
            pl.col('birth_date').cast(pl.Datetime).alias('birth_dt')
        )
    
    # Load visit data
    visit_df = read_table("visit_occurrence")
    if visit_df is None:
        raise ValueError("Visit occurrence table not found")
    
    # Handle date columns
    date_col = 'visit_start_datetime' if 'visit_start_datetime' in visit_df.columns else 'visit_start_date'
    if date_col == 'visit_start_date':
        visit_df = visit_df.with_columns(
            pl.col(date_col).cast(pl.Datetime).alias('visit_start_dt')
        )
    else:
        visit_df = visit_df.with_columns(
            pl.col(date_col).alias('visit_start_dt')
        )
    
    # Filter visits
    if person_ids:
        visit_df = visit_df.filter(pl.col('person_id').is_in(person_ids))
    visit_df = visit_df.filter(pl.col('visit_start_dt') >= start_datetime)
    
    # Join with person data
    visit_df = visit_df.join(
        person_df.select(['person_id', 'birth_dt']),
        on='person_id',
        how='left'
    )
    
    # Sort and add temporal features
    visit_df = visit_df.sort(['person_id', 'visit_start_dt'])
    visit_df = visit_df.with_columns([
        pl.col('person_id').cum_count().over('person_id').alias('visit_order'),
        pl.col('visit_start_dt').shift(1).over('person_id').alias('prev_visit_dt'),
        ((pl.col('visit_start_dt') - pl.col('birth_dt')).dt.total_days() / 365.25).alias('age'),
        (pl.col('visit_start_dt') - datetime(1970, 1, 1)).dt.total_days().alias('date'),
        ((pl.col('person_id').cum_count().over('person_id') + 1) % 2 + 1).alias('visit_segment')
    ])
    
    # Calculate days since previous visit
    visit_df = visit_df.with_columns([
        pl.when(pl.col('prev_visit_dt').is_not_null())
        .then((pl.col('visit_start_dt') - pl.col('prev_visit_dt')).dt.total_days())
        .otherwise(None)
        .alias('days_since_prev')
    ])
    
    # Get unique person IDs and partition them
    unique_persons = visit_df['person_id'].unique().sort()
    person_count = len(unique_persons)
    persons_per_partition = (person_count + num_partitions - 1) // num_partitions
    
    # Save visit data partitioned by person_id ranges
    partition_info = []
    for i in range(num_partitions):
        start_idx = i * persons_per_partition
        end_idx = min((i + 1) * persons_per_partition, person_count)
        
        if start_idx >= person_count:
            break
            
        partition_persons = unique_persons[start_idx:end_idx].to_list()
        partition_df = visit_df.filter(pl.col('person_id').is_in(partition_persons))
        
        partition_path = staging_folder / f"visits_partition_{i:04d}.parquet"
        partition_df.write_parquet(partition_path)
        
        partition_info.append({
            'partition_id': i,
            'person_ids': partition_persons,
            'min_person_id': min(partition_persons),
            'max_person_id': max(partition_persons),
            'path': str(partition_path)
        })
        
        logger.info(f"Saved visit partition {i} with {len(partition_persons)} patients")
    
    # Save metadata
    metadata = {
        'num_partitions': len(partition_info),
        'total_patients': person_count,
        'partitions': partition_info
    }
    
    metadata_path = staging_folder / "visit_staging_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Visit staging complete: {len(partition_info)} partitions created")
    return metadata


def prepare_domain_staging(
    input_folder: Path,
    staging_folder: Path,
    domain_tables: List[str],
    included_concept_ids: Optional[set],
    visit_metadata: Dict[str, Any]
) -> None:
    """
    Pre-process domain tables and save to staging partitioned by person_id.
    """
    logger.info("Preparing domain staging data...")
    
    for table_name in domain_tables:
        logger.info(f"Processing {table_name}...")
        table_staging = staging_folder / table_name
        table_staging.mkdir(exist_ok=True)
        
        # Helper to read table
        def read_table(table_name: str) -> Optional[pl.DataFrame]:
            table_path = input_folder / f"{table_name}.parquet"
            if table_path.exists():
                return pl.scan_parquet(table_path).collect()
            
            table_dir = input_folder / table_name
            if table_dir.exists() and table_dir.is_dir():
                parquet_files = list(table_dir.glob("*.parquet"))
                if parquet_files:
                    return pl.concat([pl.scan_parquet(f).collect() for f in parquet_files])
            return None
        
        df = read_table(table_name)
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
        
        # Filter by concepts if provided
        if included_concept_ids:
            df = df.filter(pl.col(concept_col).is_in(included_concept_ids))
        
        # Select only needed columns
        df = df.select([
            'person_id',
            'visit_occurrence_id',
            pl.col(concept_col).alias('concept_id')
        ])
        
        # Save partitioned by person_id using visit metadata
        for partition in visit_metadata['partitions']:
            partition_persons = partition['person_ids']
            partition_df = df.filter(pl.col('person_id').is_in(partition_persons))
            
            if len(partition_df) > 0:
                partition_path = table_staging / f"partition_{partition['partition_id']:04d}.parquet"
                partition_df.write_parquet(partition_path)
            
        logger.info(f"Completed staging for {table_name}")


def process_partition(
    args: tuple
) -> str:
    """
    Process a single partition to generate sequences.
    This function is designed to be run in parallel.
    """
    partition_id, partition_info, staging_folder, domain_tables, output_folder = args
    
    try:
        # Set thread pool size for this worker to avoid oversubscription
        import os
        os.environ['POLARS_MAX_THREADS'] = '1'
        
        # Add small delay to stagger I/O operations
        import time
        time.sleep(partition_id * 0.1)
        
        # Load visit data for this partition
        visit_df = pl.read_parquet(partition_info['path'])
        
        # Load domain data for this partition
        domain_data = {}
        for table_name in domain_tables:
            table_path = staging_folder / table_name / f"partition_{partition_id:04d}.parquet"
            if table_path.exists():
                domain_data[table_name] = pl.read_parquet(table_path)
        
        # Process patients in batches for better CPU utilization
        sequences = []
        unique_patients = visit_df['person_id'].unique().to_list()
        
        # Process in smaller batches within partition
        batch_size = min(100, len(unique_patients))
        
        for i in range(0, len(unique_patients), batch_size):
            batch_patients = unique_patients[i:i+batch_size]
            
            # Process batch of patients
            for person_id in batch_patients:
                patient_visits = visit_df.filter(pl.col('person_id') == person_id)
                
                # Build sequence
                sequence_parts = []
                
                for visit in patient_visits.iter_rows(named=True):
                    visit_parts = []
                    
                    # Add ATT token if not first visit
                    if visit.get('days_since_prev') is not None:
                        att_token = generate_att_token(int(visit['days_since_prev']))
                        if att_token:
                            visit_parts.append({
                                'concept_id': att_token,
                                'age': visit['age'],
                                'date': visit['date'],
                                'visit_segment': visit['visit_segment'],
                                'visit_order': visit['visit_order'],
                                'mlm_skip': 1
                            })
                    
                    # Add VS token
                    visit_parts.append({
                        'concept_id': '[VS]',
                        'age': visit['age'],
                        'date': visit['date'],
                        'visit_segment': visit['visit_segment'],
                        'visit_order': visit['visit_order'],
                        'mlm_skip': 1
                    })
                    
                    # Add visit type
                    visit_parts.append({
                        'concept_id': str(visit['visit_concept_id']),
                        'age': visit['age'],
                        'date': visit['date'],
                        'visit_segment': visit['visit_segment'],
                        'visit_order': visit['visit_order'],
                        'mlm_skip': 0
                    })
                    
                    # Add concepts from domain tables
                    for table_name, df in domain_data.items():
                        visit_concepts = df.filter(
                            (pl.col('person_id') == person_id) &
                            (pl.col('visit_occurrence_id') == visit['visit_occurrence_id'])
                        )['concept_id'].to_list()
                        
                        for concept_id in visit_concepts:
                            visit_parts.append({
                                'concept_id': str(concept_id),
                                'age': visit['age'],
                                'date': visit['date'],
                                'visit_segment': visit['visit_segment'],
                                'visit_order': visit['visit_order'],
                                'mlm_skip': 0
                            })
                    
                    # Add VE token
                    visit_parts.append({
                        'concept_id': '[VE]',
                        'age': visit['age'],
                        'date': visit['date'],
                        'visit_segment': visit['visit_segment'],
                        'visit_order': visit['visit_order'],
                        'mlm_skip': 1
                    })
                    
                    sequence_parts.extend(visit_parts)
                
                # Create final sequence
                if sequence_parts:
                    sequence = {
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
                        'num_of_visits': len(patient_visits)
                    }
                    sequences.append(sequence)
            
            # Garbage collect after each batch
            gc.collect()
        
        # Save partition results
        if sequences:
            partition_df = pl.DataFrame(sequences)
            # Ensure HuggingFace compatibility
            partition_df = ensure_huggingface_compatibility(partition_df)
            output_path = output_folder / f"sequences_partition_{partition_id:04d}.parquet"
            partition_df.write_parquet(output_path)
            return f"Partition {partition_id}: {len(sequences)} sequences"
        else:
            return f"Partition {partition_id}: No sequences"
            
    except Exception as e:
        return f"Partition {partition_id}: Error - {str(e)}"


def generate_sequences_from_staging(
    staging_folder: Path,
    output_folder: Path,
    domain_tables: List[str],
    num_workers: Optional[int] = None
) -> pl.DataFrame:
    """
    Generate sequences from staging files using parallel processing.
    """
    logger.info("Generating sequences from staging...")
    
    # Load metadata
    metadata_path = staging_folder / "visit_staging_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Prepare arguments for parallel processing
    if num_workers is None:
        # Use fewer workers to avoid oversubscription
        num_workers = min(metadata['num_partitions'], max(1, cpu_count() // 2))
    
    partition_args = [
        (p['partition_id'], p, staging_folder, domain_tables, output_folder)
        for p in metadata['partitions']
    ]
    
    # Process partitions in parallel
    logger.info(f"Processing {len(partition_args)} partitions with {num_workers} workers...")
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_partition, partition_args),
            total=len(partition_args),
            desc="Processing partitions"
        ))
    
    # Log results
    for result in results:
        logger.info(result)
    
    # Merge all partition results
    logger.info("Merging partition results...")
    partition_files = sorted(output_folder.glob("sequences_partition_*.parquet"))
    
    if not partition_files:
        raise ValueError("No sequence partitions generated")
    
    # Read and concatenate all partitions
    all_sequences = pl.concat([pl.read_parquet(f) for f in partition_files])
    
    # Clean up partition files
    for f in partition_files:
        f.unlink()
    
    # Save final result
    final_path = output_folder / "patient_sequence.parquet"
    all_sequences.write_parquet(final_path)
    logger.info(f"Saved {len(all_sequences)} sequences to {final_path}")
    
    # Save sample
    if len(all_sequences) > 0:
        sample_sequences = all_sequences.limit(5).to_dicts()
        sample_path = output_folder / "patient_sequence_sample.json"
        with open(sample_path, "w") as f:
            json.dump(sample_sequences, f, indent=2, default=str)
        logger.info(f"Saved sample to {sample_path}")
    
    return all_sequences


def create_patient_sequences(
    input_folder: Path,
    output_folder: Path,
    start_date: str,
    concept_filter_file: Optional[Path] = None,
    att_type: str = "day",
    domain_tables: List[str] = None,
    sample_size: Optional[int] = None,
    batch_size: int = 5000,
    use_staging: bool = False,
    num_workers: Optional[int] = None,
    num_partitions: int = 20
) -> pl.DataFrame:
    """
    Create patient sequences using optimized batch processing.
    
    Major optimizations:
    1. Batch processing instead of patient-by-patient
    2. Vectorized operations where possible
    3. Pre-joined data to minimize lookups
    4. Columnar operations instead of row iterations
    5. Optional staging mode for very large datasets
    """
    # Set default domain tables if not provided
    if domain_tables is None:
        domain_tables = ["condition_occurrence", "procedure_occurrence", "drug_exposure"]
    
    # If using staging mode, use the optimized pipeline
    if use_staging:
        logger.info("Using staging mode for sequence generation")
        staging_folder = output_folder / "staging"
        
        # Check if staging already exists
        if staging_folder.exists() and (staging_folder / "visit_staging_metadata.json").exists():
            logger.info("Using existing staging data")
        else:
            # Prepare staging
            logger.info("Creating staging data...")
            
            # Load person IDs if sampling
            person_ids = None
            if sample_size:
                # Helper to read table
                def read_table(table_name: str) -> Optional[pl.DataFrame]:
                    table_path = input_folder / f"{table_name}.parquet"
                    if table_path.exists():
                        return pl.scan_parquet(table_path).collect()
                    
                    table_dir = input_folder / table_name
                    if table_dir.exists() and table_dir.is_dir():
                        parquet_files = list(table_dir.glob("*.parquet"))
                        if parquet_files:
                            return pl.concat([pl.scan_parquet(f).collect() for f in parquet_files])
                    return None
                
                person_df = read_table("person")
                if person_df is not None:
                    person_df = person_df.sample(n=min(sample_size, len(person_df)), seed=42)
                    person_ids = set(person_df['person_id'].to_list())
            
            # Prepare visit staging
            visit_metadata = prepare_visit_staging(
                input_folder, staging_folder, start_date, person_ids, num_partitions
            )
            
            # Load filtered concepts
            included_concept_ids = None
            if concept_filter_file and concept_filter_file.exists():
                included_concepts = pl.read_parquet(concept_filter_file)
                included_concept_ids = set(included_concepts['concept_id'].to_list())
                logger.info(f"Loaded {len(included_concept_ids)} filtered concepts")
            
            # Prepare domain staging
            prepare_domain_staging(
                input_folder, staging_folder, domain_tables, included_concept_ids, visit_metadata
            )
        
        # Generate sequences from staging
        return generate_sequences_from_staging(
            staging_folder, output_folder, domain_tables, num_workers
        )
    
    # Otherwise, use the existing batch processing approach
    
    # Parse start date
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Load filtered concepts if provided
    included_concept_ids = None
    if concept_filter_file:
        included_concepts = pl.read_parquet(concept_filter_file)
        included_concept_ids = set(included_concepts['concept_id'].to_list())
        logger.info(f"Loaded {len(included_concept_ids)} filtered concepts")
    
    # Helper function to read OMOP tables
    def read_table(table_name: str) -> Optional[pl.DataFrame]:
        # Check for single file first
        table_path = input_folder / f"{table_name}.parquet"
        if table_path.exists():
            return pl.scan_parquet(table_path).collect()
        
        # Check for directory with partitioned files
        table_dir = input_folder / table_name
        if table_dir.exists() and table_dir.is_dir():
            parquet_files = list(table_dir.glob("*.parquet"))
            if parquet_files:
                return pl.concat([pl.scan_parquet(f).collect() for f in parquet_files])
        
        return None
    
    # Load and prepare person table with birth dates
    logger.info("Loading person table...")
    person_df = read_table("person")
    if person_df is None:
        raise ValueError("Person table not found")
    
    if sample_size:
        logger.info(f"Sampling {sample_size} patients...")
        person_df = person_df.sample(n=min(sample_size, len(person_df)), seed=42)
    
    # Ensure birth date is datetime
    if 'birth_datetime' in person_df.columns:
        person_df = person_df.with_columns(
            pl.col('birth_datetime').alias('birth_dt')
        )
    else:
        person_df = person_df.with_columns(
            pl.col('birth_date').cast(pl.Datetime).alias('birth_dt')
        )
    
    person_ids = set(person_df['person_id'].to_list())
    logger.info(f"Processing {len(person_ids)} patients")
    
    # Load and prepare visit data
    logger.info("Loading and preparing visit data...")
    visit_df = read_table("visit_occurrence")
    if visit_df is None:
        raise ValueError("Visit occurrence table not found")
    
    # Handle date columns
    date_col = 'visit_start_datetime' if 'visit_start_datetime' in visit_df.columns else 'visit_start_date'
    if date_col == 'visit_start_date':
        visit_df = visit_df.with_columns(
            pl.col(date_col).cast(pl.Datetime).alias('visit_start_dt')
        )
    else:
        visit_df = visit_df.with_columns(
            pl.col(date_col).alias('visit_start_dt')
        )
    
    # Filter and sort visits
    visit_df = visit_df.filter(
        (pl.col('person_id').is_in(person_ids)) &
        (pl.col('visit_start_dt') >= start_datetime)
    ).sort(['person_id', 'visit_start_dt'])
    
    # Add visit order within each patient
    visit_df = visit_df.with_columns([
        pl.col('person_id').cum_count().over('person_id').alias('visit_order'),
        # Add previous visit time for ATT calculation
        pl.col('visit_start_dt').shift(1).over('person_id').alias('prev_visit_dt')
    ])
    
    # Calculate days difference for ATT tokens
    visit_df = visit_df.with_columns([
        pl.when(pl.col('prev_visit_dt').is_not_null())
        .then((pl.col('visit_start_dt') - pl.col('prev_visit_dt')).dt.total_days())
        .otherwise(None)
        .alias('days_since_prev')
    ])
    
    # Join with person data to get birth dates
    visit_df = visit_df.join(
        person_df.select(['person_id', 'birth_dt']),
        on='person_id',
        how='left'
    )
    
    # Calculate age and date fields
    visit_df = visit_df.with_columns([
        ((pl.col('visit_start_dt') - pl.col('birth_dt')).dt.total_days() / 365.25).alias('age'),
        (pl.col('visit_start_dt') - datetime(1970, 1, 1)).dt.total_days().alias('date'),
        # Alternate visit segments
        ((pl.col('visit_order') + 1) % 2 + 1).alias('visit_segment')
    ])
    
    # For large datasets, we'll process domain data in chunks
    logger.info("Preparing domain table information...")
    domain_info = {}
    
    for table_name in domain_tables:
        logger.info(f"Checking {table_name}...")
        # Just check if table exists and get concept column
        # Read a small sample to check if table exists
        table_path = input_folder / f"{table_name}.parquet"
        if table_path.exists():
            df_sample = pl.scan_parquet(table_path).head(1).collect()
        else:
            table_dir = input_folder / table_name
            if table_dir.exists() and table_dir.is_dir():
                parquet_files = list(table_dir.glob("*.parquet"))
                if parquet_files:
                    df_sample = pl.scan_parquet(parquet_files[0]).head(1).collect()
                else:
                    df_sample = None
            else:
                df_sample = None
        if df_sample is None:
            logger.warning(f"Table {table_name} not found, skipping...")
            continue
        
        # Find concept column
        concept_col = next(
            (col for col in df_sample.columns if col.endswith('_concept_id') and col != 'visit_concept_id'),
            None
        )
        if concept_col:
            domain_info[table_name] = concept_col
        del df_sample  # Free memory
    
    # Process in batches
    logger.info("Creating patient sequences in batches...")
    unique_patients = visit_df['person_id'].unique().to_list()
    n_batches = (len(unique_patients) + batch_size - 1) // batch_size
    
    all_sequences = []
    
    for i in tqdm(range(n_batches), desc="Processing patient batches"):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(unique_patients))
        batch_patients = unique_patients[batch_start:batch_end]
        
        # Get visits for this batch
        batch_visits = visit_df.filter(pl.col('person_id').is_in(batch_patients))
        
        # Load domain data only for this batch of patients
        batch_domain_data = []
        for table_name, concept_col in domain_info.items():
            df = read_table(table_name)
            if df is None:
                continue
            
            # Filter by batch patients first to reduce memory usage
            df = df.filter(pl.col('person_id').is_in(batch_patients))
            
            if included_concept_ids:
                df = df.filter(pl.col(concept_col).is_in(included_concept_ids))
            
            # Select only needed columns
            df = df.select([
                'person_id',
                'visit_occurrence_id',
                pl.col(concept_col).alias('concept_id')
            ])
            
            batch_domain_data.append(df)
        
        # Combine domain data for this batch
        if batch_domain_data:
            batch_concepts = pl.concat(batch_domain_data)
        else:
            batch_concepts = pl.DataFrame({
                'person_id': [],
                'visit_occurrence_id': [],
                'concept_id': []
            })
        
        # Clear individual dataframes to free memory
        del batch_domain_data
        gc.collect()
        
        # Process each patient in the batch
        for patient_id in batch_patients:
            patient_visits = batch_visits.filter(pl.col('person_id') == patient_id)
            patient_concepts = batch_concepts.filter(pl.col('person_id') == patient_id)
            
            if len(patient_visits) == 0:
                continue
            
            # Build sequence efficiently
            sequence_parts = []
            
            for visit in patient_visits.iter_rows(named=True):
                visit_parts = []
                
                # Add ATT token if not first visit
                if visit.get('days_since_prev') is not None:
                    att_token = generate_att_token(int(visit['days_since_prev']))
                    if att_token:
                        visit_parts.append({
                            'concept_id': att_token,
                        'age': visit['age'],
                        'date': visit['date'],
                        'visit_segment': visit['visit_segment'],
                        'visit_order': visit['visit_order'],
                        'mlm_skip': 1
                    })
                
                # Add VS token
                visit_parts.append({
                    'concept_id': '[VS]',
                    'age': visit['age'],
                    'date': visit['date'],
                    'visit_segment': visit['visit_segment'],
                    'visit_order': visit['visit_order'],
                    'mlm_skip': 1
                })
                
                # Add visit type
                visit_parts.append({
                    'concept_id': str(visit['visit_concept_id']),
                    'age': visit['age'],
                    'date': visit['date'],
                    'visit_segment': visit['visit_segment'],
                    'visit_order': visit['visit_order'],
                    'mlm_skip': 0
                })
                
                # Add concepts for this visit
                visit_concepts = patient_concepts.filter(
                    pl.col('visit_occurrence_id') == visit['visit_occurrence_id']
                )
                
                for concept in visit_concepts.iter_rows(named=True):
                    visit_parts.append({
                        'concept_id': str(concept['concept_id']),
                        'age': visit['age'],
                        'date': visit['date'],
                        'visit_segment': visit['visit_segment'],
                        'visit_order': visit['visit_order'],
                        'mlm_skip': 0
                    })
                
                # Add VE token
                visit_parts.append({
                    'concept_id': '[VE]',
                    'age': visit['age'],
                    'date': visit['date'],
                    'visit_segment': visit['visit_segment'],
                    'visit_order': visit['visit_order'],
                    'mlm_skip': 1
                })
                
                sequence_parts.extend(visit_parts)
            
            # Create final sequence
            if sequence_parts:
                sequence = {
                    'person_id': patient_id,
                    'concept_ids': [p['concept_id'] for p in sequence_parts],
                    'ages': [p['age'] for p in sequence_parts],
                    'dates': [p['date'] for p in sequence_parts],
                    'visit_segments': [p['visit_segment'] for p in sequence_parts],
                    'visit_concept_orders': [p['visit_order'] for p in sequence_parts],
                    'concept_values': [0.0] * len(sequence_parts),
                    'concept_value_masks': [0.0] * len(sequence_parts),
                    'mlm_skip_values': [p['mlm_skip'] for p in sequence_parts],
                    'num_of_concepts': len(sequence_parts),
                    'num_of_visits': len(patient_visits)
                }
                all_sequences.append(sequence)
        
        # Clear batch data to free memory
        del batch_visits, batch_concepts
        gc.collect()
        
        # Log memory usage
        if i % 10 == 0:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
            logger.info(f"Memory usage: {memory_usage:.2f} GB")
    
    logger.info(f"Created {len(all_sequences)} patient sequences")
    
    # Convert to DataFrame
    sequences_df = pl.DataFrame(all_sequences)
    
    # Ensure HuggingFace compatibility
    sequences_df = ensure_huggingface_compatibility(sequences_df)
    
    # Save sequences
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "patient_sequence.parquet"
    sequences_df.write_parquet(output_path)
    logger.info(f"Saved sequences to {output_path}")
    
    # Save sample for inspection
    if len(all_sequences) > 0:
        sample_path = output_folder / "patient_sequence_sample.json"
        with open(sample_path, "w") as f:
            json.dump(all_sequences[:5], f, indent=2, default=str)
        logger.info(f"Saved sample to {sample_path}")
    
    return sequences_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate patient sequences from OMOP data"
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5000,
        help="Batch size for processing (default: 5000)"
    )
    parser.add_argument(
        "--use-staging",
        action="store_true",
        help="Use staging mode for very large datasets"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of parallel workers for staging mode"
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=20,
        help="Number of partitions for staging (default: 20)"
    )
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    concept_filter_file = Path(args.concept_filter_file) if args.concept_filter_file else None
    
    sequences_df = create_patient_sequences(
        input_folder,
        output_folder,
        args.start_date,
        concept_filter_file,
        args.att_type,
        args.domain_tables,
        args.sample_size,
        args.batch_size,
        args.use_staging,
        args.num_workers,
        args.num_partitions
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