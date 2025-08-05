#!/usr/bin/env python3
"""
Consolidate partitioned OMOP parquet files into single files for each table.
"""

import polars as pl
from pathlib import Path
import sys

def consolidate_omop_tables(input_dir: Path, output_dir: Path):
    """Consolidate partitioned OMOP tables into single parquet files."""
    
    # Tables to consolidate
    tables = [
        "person",
        "visit_occurrence", 
        "condition_occurrence",
        "procedure_occurrence",
        "drug_exposure",
        "measurement",
        "observation_period",
        "concept",
        "concept_relationship",
        "concept_ancestor"
    ]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for table in tables:
        table_dir = input_dir / table
        if not table_dir.exists():
            print(f"Skipping {table} - directory not found")
            continue
            
        print(f"Consolidating {table}...")
        
        # Find all parquet files
        parquet_files = list(table_dir.glob("*.parquet"))
        if not parquet_files:
            print(f"  No parquet files found in {table_dir}")
            continue
        
        # Read and concatenate all partitions
        df = pl.concat([pl.read_parquet(f) for f in parquet_files])
        
        # Save as single file
        output_file = output_dir / f"{table}.parquet"
        df.write_parquet(output_file)
        print(f"  Saved {len(df)} rows to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/omop_consolidated")
    else:
        input_dir = Path("data/omop")
        output_dir = Path("data/omop_consolidated")
    
    consolidate_omop_tables(input_dir, output_dir)