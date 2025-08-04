#!/usr/bin/env python3
"""
Benchmark different sequence generation implementations.
Compares performance of original, staging, and vectorized approaches.
"""

import time
import argparse
from pathlib import Path
import logging
import json
import polars as pl
import os
import shutil
from datetime import datetime
import numpy as np
from datetime import timedelta

from cehrbert.data_processing.sequence_generator import create_patient_sequences
from cehrbert.data_processing.sequence_generator_vectorized import create_patient_sequences_vectorized

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_test_data(output_dir: Path, n_patients: int, avg_visits: int = 20):
    """Generate synthetic OMOP data for benchmarking."""
    logger.info(f"Generating test data for {n_patients} patients...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate person table
    person_data = {
        'person_id': list(range(1, n_patients + 1)),
        'birth_datetime': [
            datetime(1950, 1, 1) + timedelta(days=int(np.random.randint(0, 365*70)))
            for _ in range(n_patients)
        ],
        'gender_concept_id': np.random.choice([8507, 8532], n_patients).tolist()
    }
    person_df = pl.DataFrame(person_data)
    person_df.write_parquet(output_dir / "person.parquet")
    
    # Generate visits
    visit_data = []
    visit_id = 1
    for person_id in range(1, n_patients + 1):
        n_visits = np.random.poisson(avg_visits)
        start_date = datetime(2020, 1, 1)
        
        for _ in range(n_visits):
            visit_date = start_date + timedelta(days=int(np.random.randint(0, 365*3)))
            visit_data.append({
                'visit_occurrence_id': visit_id,
                'person_id': person_id,
                'visit_concept_id': np.random.choice([9201, 9202, 9203]),
                'visit_start_datetime': visit_date
            })
            visit_id += 1
    
    visit_df = pl.DataFrame(visit_data)
    visit_df.write_parquet(output_dir / "visit_occurrence.parquet")
    
    # Generate conditions
    condition_data = []
    for visit in visit_data:
        n_conditions = np.random.poisson(3)
        for _ in range(n_conditions):
            condition_data.append({
                'person_id': visit['person_id'],
                'visit_occurrence_id': visit['visit_occurrence_id'],
                'condition_concept_id': np.random.randint(100000, 100500)
            })
    
    if condition_data:
        condition_df = pl.DataFrame(condition_data)
        condition_df.write_parquet(output_dir / "condition_occurrence.parquet")
    
    # Generate drugs
    drug_data = []
    for visit in visit_data:
        if np.random.random() < 0.5:
            n_drugs = np.random.poisson(2)
            for _ in range(n_drugs):
                drug_data.append({
                    'person_id': visit['person_id'],
                    'visit_occurrence_id': visit['visit_occurrence_id'],
                    'drug_concept_id': np.random.randint(200000, 200300)
                })
    
    if drug_data:
        drug_df = pl.DataFrame(drug_data)
        drug_df.write_parquet(output_dir / "drug_exposure.parquet")
    
    logger.info(f"Generated {len(visit_data)} visits")
    return len(visit_data)


def benchmark_implementation(
    name: str,
    func: callable,
    input_dir: Path,
    output_dir: Path,
    **kwargs
) -> dict:
    """Benchmark a single implementation."""
    logger.info(f"\nBenchmarking {name}...")
    
    # Clear output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    start_time = time.time()
    try:
        result = func(
            input_folder=input_dir,
            output_folder=output_dir,
            start_date="2020-01-01",
            domain_tables=["condition_occurrence", "drug_exposure"],
            **kwargs
        )
        elapsed_time = time.time() - start_time
        
        # Get statistics
        stats = {
            'name': name,
            'success': True,
            'elapsed_time': elapsed_time,
            'num_sequences': len(result),
            'avg_sequence_length': float(result['num_of_concepts'].mean()),
            'max_sequence_length': int(result['num_of_concepts'].max()),
            'min_sequence_length': int(result['num_of_concepts'].min())
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        stats = {
            'name': name,
            'success': False,
            'elapsed_time': elapsed_time,
            'error': str(e)
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark different sequence generation implementations"
    )
    parser.add_argument(
        "--patients",
        type=int,
        default=1000,
        help="Number of patients for benchmarking"
    )
    parser.add_argument(
        "--avg-visits",
        type=int,
        default=20,
        help="Average visits per patient"
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=["original", "staging", "vectorized"],
        default=["original", "vectorized"],
        help="Implementations to benchmark"
    )
    
    args = parser.parse_args()
    
    # Create temporary directories
    temp_dir = Path("benchmark_temp")
    input_dir = temp_dir / "input"
    
    # Generate test data
    try:
        total_visits = generate_test_data(input_dir, args.patients, args.avg_visits)
        
        print(f"\n{'='*60}")
        print(f"Benchmarking Sequence Generation Implementations")
        print(f"{'='*60}")
        print(f"Test data: {args.patients} patients, ~{total_visits} visits")
        print(f"{'='*60}\n")
        
        results = []
        
        # Benchmark each implementation
        if "original" in args.implementations:
            result = benchmark_implementation(
                "Original (batch processing)",
                create_patient_sequences,
                input_dir,
                temp_dir / "output_original",
                batch_size=5000
            )
            results.append(result)
        
        if "staging" in args.implementations:
            result = benchmark_implementation(
                "Staging (multiprocessing)",
                create_patient_sequences,
                input_dir,
                temp_dir / "output_staging",
                use_staging=True,
                num_workers=4,
                num_partitions=10
            )
            results.append(result)
        
        if "vectorized" in args.implementations:
            result = benchmark_implementation(
                "Vectorized (columnar ops)",
                create_patient_sequences_vectorized,
                input_dir,
                temp_dir / "output_vectorized"
            )
            results.append(result)
        
        # Print results
        print("\nResults:")
        print(f"{'Implementation':<30} {'Time (s)':<10} {'Sequences':<10} {'Avg Length':<10}")
        print("-" * 60)
        
        for result in results:
            if result['success']:
                print(f"{result['name']:<30} {result['elapsed_time']:<10.2f} "
                      f"{result['num_sequences']:<10} {result['avg_sequence_length']:<10.1f}")
            else:
                print(f"{result['name']:<30} {'FAILED':<10} {result.get('error', '')}")
        
        # Calculate speedups
        if len(results) > 1 and results[0]['success']:
            baseline_time = results[0]['elapsed_time']
            print(f"\nSpeedups (vs {results[0]['name']}):")
            for result in results[1:]:
                if result['success']:
                    speedup = baseline_time / result['elapsed_time']
                    print(f"  {result['name']}: {speedup:.2f}x faster")
        
        # Save detailed results
        with open("benchmark_results.json", "w") as f:
            json.dump({
                'config': {
                    'patients': args.patients,
                    'avg_visits': args.avg_visits,
                    'total_visits': total_visits
                },
                'results': results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to benchmark_results.json")
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()