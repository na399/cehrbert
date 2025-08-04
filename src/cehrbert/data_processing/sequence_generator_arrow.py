#!/usr/bin/env python3
"""
Arrow-based patient sequence generation from OMOP data using Polars.
Saves sequences as Arrow shards on the fly to avoid memory issues.
"""

import argparse
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import logging
import json
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Polars to use optimal number of threads
num_threads = int(os.environ.get('POLARS_MAX_THREADS', os.cpu_count() - 1))
pl.Config.set_streaming_chunk_size(10000)
logger.info(f"Polars configured with {num_threads} threads")


def vectorized_att_tokens(days_col: pl.Expr) -> pl.Expr:
    """Vectorized generation of artificial time tokens."""
    return (
        pl.when(days_col.is_null()).then(None)
        .when(days_col < 0).then(None)
        .when(days_col < 7).then(pl.lit("W0"))
        .when(days_col < 14).then(pl.lit("W1"))
        .when(days_col < 21).then(pl.lit("W2"))
        .when(days_col < 28).then(pl.lit("W3"))
        .when(days_col < 365).then(
            pl.concat_str([
                pl.lit("M"),
                (days_col / 30).cast(pl.Int32).clip(0, 11).cast(pl.Utf8)
            ])
        )
        .otherwise(pl.lit("LT"))
    )


def process_patient_chunk(visits_df: pl.DataFrame, concepts_df: pl.DataFrame) -> pl.DataFrame:
    """Process a chunk of patients using vectorized operations."""
    # 1. Create ATT tokens (for visits with previous visits)
    att_tokens = (
        visits_df
        .filter(pl.col('days_since_prev').is_not_null())
        .with_columns([
            vectorized_att_tokens(pl.col('days_since_prev')).alias('concept_id'),
            pl.lit(1).alias('mlm_skip'),
            pl.lit(0).alias('token_order')  # ATT comes first
        ])
        .filter(pl.col('concept_id').is_not_null())
        .select(['person_id', 'visit_occurrence_id', 'visit_order', 'concept_id', 
                'age', 'date', 'visit_segment', 'mlm_skip', 'token_order'])
    )
    
    # 2. Create VS tokens (start of visit)
    vs_tokens = (
        visits_df
        .with_columns([
            pl.lit('[VS]').alias('concept_id'),
            pl.lit(1).alias('mlm_skip'),
            pl.lit(1).alias('token_order')  # VS comes after ATT
        ])
        .select(['person_id', 'visit_occurrence_id', 'visit_order', 'concept_id',
                'age', 'date', 'visit_segment', 'mlm_skip', 'token_order'])
    )
    
    # 3. Create visit type tokens
    visit_type_tokens = (
        visits_df
        .with_columns([
            pl.col('visit_concept_id').cast(pl.Utf8).alias('concept_id'),
            pl.lit(0).alias('mlm_skip'),
            pl.lit(2).alias('token_order')  # Visit type after VS
        ])
        .select(['person_id', 'visit_occurrence_id', 'visit_order', 'concept_id',
                'age', 'date', 'visit_segment', 'mlm_skip', 'token_order'])
    )
    
    # 4. Process domain concepts with visit metadata
    if len(concepts_df) > 0:
        # Join concepts with visit metadata
        concept_tokens = (
            concepts_df
            .join(
                visits_df.select(['person_id', 'visit_occurrence_id', 'visit_order',
                                'age', 'date', 'visit_segment']),
                on=['person_id', 'visit_occurrence_id'],
                how='inner'
            )
            .with_columns([
                pl.lit(0).alias('mlm_skip'),
                pl.lit(3).alias('token_order')  # Domain concepts after visit type
            ])
            .select(['person_id', 'visit_occurrence_id', 'visit_order', 'concept_id',
                    'age', 'date', 'visit_segment', 'mlm_skip', 'token_order'])
        )
    else:
        # Create empty dataframe with same schema
        concept_tokens = pl.DataFrame({
            'person_id': [],
            'visit_occurrence_id': [],
            'visit_order': [],
            'concept_id': [],
            'age': [],
            'date': [],
            'visit_segment': [],
            'mlm_skip': [],
            'token_order': []
        })
    
    # 5. Create VE tokens (end of visit)
    ve_tokens = (
        visits_df
        .with_columns([
            pl.lit('[VE]').alias('concept_id'),
            pl.lit(1).alias('mlm_skip'),
            pl.lit(4).alias('token_order')  # VE comes last
        ])
        .select(['person_id', 'visit_occurrence_id', 'visit_order', 'concept_id',
                'age', 'date', 'visit_segment', 'mlm_skip', 'token_order'])
    )
    
    # Combine all tokens
    all_tokens = pl.concat([
        att_tokens,
        vs_tokens,
        visit_type_tokens,
        concept_tokens,
        ve_tokens
    ])
    
    # Sort tokens by person, visit order, and token order
    all_tokens = all_tokens.sort(['person_id', 'visit_order', 'token_order'])
    
    # Create final sequences using group_by and aggregation
    sequences_df = (
        all_tokens
        .group_by('person_id', maintain_order=True)
        .agg([
            pl.col('concept_id').alias('concept_ids'),
            pl.col('age').alias('ages'),
            pl.col('date').alias('dates'),
            pl.col('visit_segment').alias('visit_segments'),
            pl.col('visit_order').alias('visit_concept_orders'),
            pl.col('mlm_skip').alias('mlm_skip_values'),
            pl.len().alias('num_of_concepts')
        ])
    )
    
    # Add visit counts
    visit_counts = (
        visits_df
        .group_by('person_id')
        .agg(pl.len().alias('num_of_visits'))
    )
    
    sequences_df = sequences_df.join(visit_counts, on='person_id', how='left')
    
    # Add concept values (placeholder)
    sequences_df = sequences_df.with_columns([
        pl.col('concept_ids').list.eval(pl.lit(0.0)).alias('concept_values'),
        pl.col('concept_ids').list.eval(pl.lit(0.0)).alias('concept_value_masks')
    ])
    
    return sequences_df


def save_as_arrow_shard(df: pl.DataFrame, output_path: Path):
    """Save DataFrame as Arrow file using PyArrow directly."""
    # Convert to PyArrow table
    arrow_table = df.to_arrow()
    
    # Write as Arrow file
    with pa.OSFile(str(output_path), 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, arrow_table.schema) as writer:
            writer.write_table(arrow_table)
    
    logger.info(f"Saved {len(df)} sequences to {output_path}")


def create_patient_sequences_arrow(
    input_folder: Path,
    output_folder: Path,
    start_date: str,
    concept_filter_file: Optional[Path] = None,
    domain_tables: List[str] = None,
    sample_size: Optional[int] = None,
    chunk_size: int = 5000
) -> None:
    """
    Create patient sequences and save as Arrow shards.
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
    
    # Helper function to scan OMOP tables
    def scan_table(table_name: str) -> Optional[pl.LazyFrame]:
        table_path = input_folder / f"{table_name}.parquet"
        if table_path.exists():
            return pl.scan_parquet(table_path)
        
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
    person_columns = person_lf.collect_schema().names()
    if 'birth_datetime' in person_columns:
        person_lf = person_lf.with_columns(pl.col('birth_datetime').alias('birth_dt'))
    else:
        person_lf = person_lf.with_columns(pl.col('birth_date').cast(pl.Datetime).alias('birth_dt'))
    
    # Load visit data
    logger.info("Loading visit data...")
    visit_lf = scan_table("visit_occurrence")
    if visit_lf is None:
        raise ValueError("Visit occurrence table not found")
    
    # Handle date columns
    visit_columns = visit_lf.collect_schema().names()
    date_col = 'visit_start_datetime' if 'visit_start_datetime' in visit_columns else 'visit_start_date'
    if date_col == 'visit_start_date':
        visit_lf = visit_lf.with_columns(pl.col(date_col).cast(pl.Datetime).alias('visit_start_dt'))
    else:
        visit_lf = visit_lf.with_columns(pl.col(date_col).alias('visit_start_dt'))
    
    # Filter visits and join with person data
    visit_lf = (
        visit_lf
        .filter(pl.col('visit_start_dt') >= start_datetime)
        .join(person_lf.select(['person_id', 'birth_dt']), on='person_id', how='inner')
        .sort(['person_id', 'visit_start_dt'])
    )
    
    # Add temporal features using window functions
    visit_lf = visit_lf.with_columns([
        # Visit order within patient
        pl.col('person_id').cum_count().over('person_id').alias('visit_order'),
        # Previous visit time
        pl.col('visit_start_dt').shift(1).over('person_id').alias('prev_visit_dt'),
        # Age calculation
        ((pl.col('visit_start_dt') - pl.col('birth_dt')).dt.total_days() / 365.25).alias('age'),
        # Date as days since epoch
        (pl.col('visit_start_dt') - datetime(1970, 1, 1)).dt.total_days().alias('date'),
        # Visit segment alternation
        ((pl.col('person_id').cum_count().over('person_id') + 1) % 2 + 1).alias('visit_segment')
    ])
    
    # Calculate days since previous visit
    visit_lf = visit_lf.with_columns([
        pl.when(pl.col('prev_visit_dt').is_not_null())
        .then((pl.col('visit_start_dt') - pl.col('prev_visit_dt')).dt.total_days())
        .otherwise(None)
        .alias('days_since_prev')
    ])
    
    # Get unique person IDs
    all_person_ids = visit_lf.select('person_id').unique().collect()['person_id'].to_list()
    total_patients = len(all_person_ids)
    logger.info(f"Processing {total_patients} patients...")
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    sequences_dir = output_folder / "sequences"
    sequences_dir.mkdir(exist_ok=True)
    
    # Calculate total number of shards
    total_shards = (total_patients + chunk_size - 1) // chunk_size
    
    # Process patients in chunks
    shard_idx = 0
    pbar = tqdm(total=total_patients, desc="Processing patients")
    
    for chunk_start in range(0, total_patients, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_patients)
        chunk_person_ids = all_person_ids[chunk_start:chunk_end]
        
        # Update progress
        pbar.update(len(chunk_person_ids))
        
        # Load visit data for this chunk
        chunk_visits_df = visit_lf.filter(pl.col('person_id').is_in(chunk_person_ids)).collect()
        
        # Load domain concepts for this chunk
        all_concepts = []
        
        for table_name in domain_tables:
            df = scan_table(table_name)
            if df is None:
                continue
            
            # Find concept column
            df_columns = df.collect_schema().names()
            concept_col = next(
                (col for col in df_columns if col.endswith('_concept_id') and col != 'visit_concept_id'),
                None
            )
            if not concept_col:
                continue
            
            # Filter by chunk person IDs first, then by concepts
            df = df.filter(pl.col('person_id').is_in(chunk_person_ids))
            
            if included_concept_ids:
                df = df.filter(pl.col(concept_col).is_in(included_concept_ids))
            
            # Select and standardize columns
            df = df.select([
                'person_id',
                'visit_occurrence_id',
                pl.col(concept_col).cast(pl.Utf8).alias('concept_id')
            ])
            
            all_concepts.append(df.collect())
        
        # Combine concept data for this chunk
        if all_concepts:
            chunk_concepts_df = pl.concat(all_concepts)
        else:
            chunk_concepts_df = pl.DataFrame({
                'person_id': [],
                'visit_occurrence_id': [],
                'concept_id': []
            })
        
        # Process this chunk
        chunk_sequences = process_patient_chunk(chunk_visits_df, chunk_concepts_df)
        
        # Save as Arrow shard
        shard_path = sequences_dir / f"data-{shard_idx:05d}-of-{total_shards:05d}.arrow"
        save_as_arrow_shard(chunk_sequences, shard_path)
        
        shard_idx += 1
        
        # Free memory
        del chunk_visits_df, chunk_concepts_df, all_concepts, chunk_sequences
        import gc
        gc.collect()
    
    # Close progress bar
    pbar.close()
    
    logger.info(f"Created {shard_idx} Arrow shards with {total_patients} patient sequences")
    
    # Save metadata
    metadata = {
        "format": "arrow",
        "num_shards": shard_idx,
        "total_sequences": total_patients,
        "chunk_size": chunk_size,
        "domain_tables": domain_tables,
        "start_date": start_date,
        "created_at": datetime.now().isoformat()
    }
    
    metadata_path = output_folder / "dataset_info.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate patient sequences from OMOP data and save as Arrow shards"
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
        help="Output folder for Arrow sequences"
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
        "--chunk_size",
        type=int,
        default=5000,
        help="Number of patients per Arrow shard (default: 5000)"
    )
    
    args = parser.parse_args()
    
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    concept_filter_file = Path(args.concept_filter_file) if args.concept_filter_file else None
    
    create_patient_sequences_arrow(
        input_folder,
        output_folder,
        args.start_date,
        concept_filter_file,
        args.domain_tables,
        args.sample_size,
        args.chunk_size
    )


if __name__ == "__main__":
    main()