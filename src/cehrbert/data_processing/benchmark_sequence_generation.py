#!/usr/bin/env python3
"""
Benchmark script for testing sequence generation performance.
Helps estimate processing time for large datasets on specific hardware.
"""

import argparse
import time
import polars as pl
import numpy as np
from pathlib import Path
import shutil
import logging
from datetime import datetime, timedelta
import json
import multiprocessing
import os

from sequence_generator import create_patient_sequences

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_omop_data(output_dir: Path, n_patients: int, avg_visits_per_patient: int = 50):
    """Generate synthetic OMOP data for benchmarking."""
    logger.info(f"Generating synthetic data for {n_patients} patients...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate person table
    person_data = []
    base_date = datetime(1950, 1, 1)
    for i in range(n_patients):
        birth_date = base_date + timedelta(days=np.random.randint(0, 365*70))
        person_data.append({
            'person_id': i + 1,
            'birth_datetime': birth_date,
            'gender_concept_id': np.random.choice([8507, 8532])  # Male/Female
        })
    
    person_df = pl.DataFrame(person_data)
    person_df.write_parquet(output_dir / "person.parquet")
    
    # Generate visit occurrence table
    visit_data = []
    visit_id = 1
    for person_id in range(1, n_patients + 1):
        n_visits = np.random.poisson(avg_visits_per_patient)
        start_date = datetime(2018, 1, 1)
        
        for v in range(n_visits):
            visit_date = start_date + timedelta(days=np.random.randint(0, 365*5))
            visit_data.append({
                'visit_occurrence_id': visit_id,
                'person_id': person_id,
                'visit_concept_id': np.random.choice([9201, 9202, 9203]),  # IP/OP/ER
                'visit_start_datetime': visit_date,
                'visit_end_datetime': visit_date + timedelta(days=np.random.randint(1, 7))
            })
            visit_id += 1
    
    visit_df = pl.DataFrame(visit_data)
    visit_df.write_parquet(output_dir / "visit_occurrence.parquet")
    
    # Generate condition occurrence table
    condition_data = []
    condition_concepts = list(range(100000, 100500))  # 500 different conditions
    
    for visit in visit_data:
        n_conditions = np.random.poisson(3)
        for _ in range(n_conditions):
            condition_data.append({
                'person_id': visit['person_id'],
                'visit_occurrence_id': visit['visit_occurrence_id'],
                'condition_concept_id': np.random.choice(condition_concepts),
                'condition_start_datetime': visit['visit_start_datetime']
            })
    
    if condition_data:
        condition_df = pl.DataFrame(condition_data)
        condition_df.write_parquet(output_dir / "condition_occurrence.parquet")
    
    # Generate drug exposure table
    drug_data = []
    drug_concepts = list(range(200000, 200300))  # 300 different drugs
    
    for visit in visit_data:
        if np.random.random() < 0.7:  # 70% of visits have drugs
            n_drugs = np.random.poisson(2)
            for _ in range(n_drugs):
                drug_data.append({
                    'person_id': visit['person_id'],
                    'visit_occurrence_id': visit['visit_occurrence_id'],
                    'drug_concept_id': np.random.choice(drug_concepts),
                    'drug_exposure_start_datetime': visit['visit_start_datetime']
                })
    
    if drug_data:
        drug_df = pl.DataFrame(drug_data)
        drug_df.write_parquet(output_dir / "drug_exposure.parquet")
    
    # Generate procedure occurrence table
    procedure_data = []
    procedure_concepts = list(range(300000, 300200))  # 200 different procedures
    
    for visit in visit_data:
        if np.random.random() < 0.3:  # 30% of visits have procedures
            n_procedures = np.random.poisson(1)
            for _ in range(n_procedures):
                procedure_data.append({
                    'person_id': visit['person_id'],
                    'visit_occurrence_id': visit['visit_occurrence_id'],
                    'procedure_concept_id': np.random.choice(procedure_concepts),
                    'procedure_datetime': visit['visit_start_datetime']
                })
    
    if procedure_data:
        procedure_df = pl.DataFrame(procedure_data)
        procedure_df.write_parquet(output_dir / "procedure_occurrence.parquet")
    
    logger.info(f"Generated {len(visit_data)} visits for {n_patients} patients")
    return len(visit_data)


def run_benchmark(
    n_patients: int,
    avg_visits_per_patient: int,
    num_workers: int,
    num_partitions: int,
    use_staging: bool = True
):
    """Run a benchmark test."""
    # Create temporary directories
    temp_dir = Path("benchmark_temp")
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    
    try:
        # Generate synthetic data
        start_time = time.time()
        total_visits = generate_synthetic_omop_data(input_dir, n_patients, avg_visits_per_patient)
        data_gen_time = time.time() - start_time
        
        # Run sequence generation
        logger.info(f"Running sequence generation with {num_workers} workers...")
        start_time = time.time()
        
        sequences_df = create_patient_sequences(
            input_folder=input_dir,
            output_folder=output_dir,
            start_date="2018-01-01",
            domain_tables=["condition_occurrence", "drug_exposure", "procedure_occurrence"],
            use_staging=use_staging,
            num_workers=num_workers,
            num_partitions=num_partitions
        )
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        patients_per_second = n_patients / processing_time
        visits_per_second = total_visits / processing_time
        avg_sequence_length = sequences_df['num_of_concepts'].mean()
        
        # Estimate for larger datasets
        estimates = {}
        for target_size in [10_000, 100_000, 1_000_000, 10_000_000]:
            est_time = target_size / patients_per_second
            estimates[f"{target_size:,}"] = {
                "seconds": int(est_time),
                "hours": round(est_time / 3600, 1)
            }
        
        results = {
            "configuration": {
                "n_patients": n_patients,
                "avg_visits_per_patient": avg_visits_per_patient,
                "total_visits": total_visits,
                "num_workers": num_workers,
                "num_partitions": num_partitions,
                "use_staging": use_staging
            },
            "performance": {
                "data_generation_time": round(data_gen_time, 2),
                "processing_time": round(processing_time, 2),
                "patients_per_second": round(patients_per_second, 2),
                "visits_per_second": round(visits_per_second, 2),
                "avg_sequence_length": round(avg_sequence_length, 1)
            },
            "estimates_for_larger_datasets": estimates,
            "hardware": {
                "cpu_count": multiprocessing.cpu_count(),
                "polars_threads": int(os.environ.get('POLARS_MAX_THREADS', multiprocessing.cpu_count()))
            }
        }
        
        return results
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sequence generation performance"
    )
    parser.add_argument(
        "--patients",
        type=int,
        default=1000,
        help="Number of patients to generate for benchmark"
    )
    parser.add_argument(
        "--avg-visits",
        type=int,
        default=50,
        help="Average visits per patient"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=10,
        help="Number of partitions for staging"
    )
    parser.add_argument(
        "--no-staging",
        action="store_true",
        help="Disable staging mode"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"CEHR-BERT Sequence Generation Benchmark")
    print(f"{'='*60}\n")
    
    results = run_benchmark(
        n_patients=args.patients,
        avg_visits_per_patient=args.avg_visits,
        num_workers=args.workers,
        num_partitions=args.partitions,
        use_staging=not args.no_staging
    )
    
    # Print results
    print(f"Configuration:")
    print(f"  Patients: {results['configuration']['n_patients']:,}")
    print(f"  Total visits: {results['configuration']['total_visits']:,}")
    print(f"  Workers: {results['configuration']['num_workers']}")
    print(f"  Partitions: {results['configuration']['num_partitions']}")
    print(f"  Staging: {'Enabled' if results['configuration']['use_staging'] else 'Disabled'}")
    print()
    
    print(f"Performance:")
    print(f"  Data generation: {results['performance']['data_generation_time']:.1f}s")
    print(f"  Processing time: {results['performance']['processing_time']:.1f}s")
    print(f"  Rate: {results['performance']['patients_per_second']:.1f} patients/sec")
    print(f"  Rate: {results['performance']['visits_per_second']:.1f} visits/sec")
    print()
    
    print(f"Estimates for larger datasets (with {args.workers} workers):")
    for size, estimate in results['estimates_for_larger_datasets'].items():
        print(f"  {size} patients: {estimate['hours']:.1f} hours")
    print()
    
    print(f"Hardware:")
    print(f"  CPU cores: {results['hardware']['cpu_count']}")
    print(f"  Polars threads: {results['hardware']['polars_threads']}")
    print()
    
    # Save detailed results
    results_file = "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()