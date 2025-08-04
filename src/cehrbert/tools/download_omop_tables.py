"""
Download OMOP CDM tables using Polars for efficient data processing.

This tool supports downloading OMOP tables from:
1. Direct database connections (PostgreSQL, MySQL, SQL Server, etc.)
2. Existing parquet files (for copying/filtering)

The tool uses Polars for efficient memory usage and supports parallel processing.
"""

import argparse
import configparser
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import polars as pl
import connectorx as cx


OMOP_TABLE_DICT = {
    "person": "person_id",
    "condition_occurrence": "condition_occurrence_id",
    "measurement": "measurement_id",
    "drug_exposure": "drug_exposure_id",
    "procedure_occurrence": "procedure_occurrence_id",
    "observation": "observation_id",
    "visit_occurrence": "visit_occurrence_id",
}


class OMOPTableDownloader:
    """Downloads OMOP CDM tables using Polars and ConnectorX."""
    
    def __init__(self, db_properties: Optional[Dict[str, str]] = None):
        """
        Initialize the downloader.
        
        Args:
            db_properties: Database connection properties including:
                - driver: Database driver (postgresql, mysql, mssql, etc.)
                - host: Database host
                - port: Database port
                - database: Database name
                - user: Username
                - password: Password
        """
        self.db_properties = db_properties
        self._connection_string = None
        
        if db_properties:
            self._build_connection_string()
    
    def _build_connection_string(self) -> None:
        """Build database connection string from properties."""
        props = self.db_properties
        driver = props.get("driver", "postgresql")
        
        # Map common driver names to connectorx format
        driver_map = {
            "postgresql": "postgresql",
            "postgres": "postgresql",
            "mysql": "mysql",
            "mssql": "mssql",
            "sqlserver": "mssql",
            "oracle": "oracle",
        }
        
        cx_driver = driver_map.get(driver.lower(), driver.lower())
        
        # Build connection string based on driver type
        if cx_driver == "postgresql":
            self._connection_string = (
                f"postgresql://{props['user']}:{props['password']}@"
                f"{props['host']}:{props.get('port', 5432)}/{props['database']}"
            )
        elif cx_driver == "mysql":
            self._connection_string = (
                f"mysql://{props['user']}:{props['password']}@"
                f"{props['host']}:{props.get('port', 3306)}/{props['database']}"
            )
        elif cx_driver == "mssql":
            self._connection_string = (
                f"mssql://{props['user']}:{props['password']}@"
                f"{props['host']}:{props.get('port', 1433)}/{props['database']}"
            )
        else:
            # Fallback to base_url if provided
            self._connection_string = props.get("base_url", "")
    
    def find_num_of_records(self, table_name: str, column_name: str) -> int:
        """
        Find the maximum ID value in a table for partitioning.
        
        Args:
            table_name: Name of the OMOP table
            column_name: Name of the ID column
            
        Returns:
            Maximum ID value in the table
        """
        query = f"SELECT MAX({column_name}) as max_id FROM {table_name}"
        
        # Use connectorx for efficient single value retrieval
        df = cx.read_sql(self._connection_string, query, return_type="polars")
        max_id = df["max_id"][0]
        
        return max_id if max_id is not None else 0
    
    def download_table_from_db(
        self,
        table_name: str,
        output_folder: Path,
        partition_column: Optional[str] = None,
        num_partitions: int = 16,
    ) -> None:
        """
        Download a table from database to parquet files.
        
        Args:
            table_name: Name of the OMOP table
            output_folder: Output directory path
            partition_column: Column to use for partitioning (optional)
            num_partitions: Number of partitions for parallel download
        """
        output_path = output_folder / table_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        if partition_column and num_partitions > 1:
            # Get max ID for partitioning
            max_id = self.find_num_of_records(table_name, partition_column)
            
            if max_id > 0:
                # Use connectorx's built-in partitioning
                query = f"SELECT * FROM {table_name}"
                
                # ConnectorX automatically handles partitioned reads
                df = cx.read_sql(
                    self._connection_string,
                    query,
                    return_type="polars",
                    partition_on=partition_column,
                    partition_num=num_partitions,
                )
            else:
                # Empty table or no valid partition column
                df = cx.read_sql(
                    f"SELECT * FROM {table_name}",
                    self._connection_string,
                    return_type="polars",
                )
        else:
            # Simple query without partitioning
            df = cx.read_sql(
                f"SELECT * FROM {table_name}",
                self._connection_string,
                return_type="polars",
            )
        
        # Write to parquet
        df.write_parquet(output_path / f"{table_name}.parquet")
        print(f"Table '{table_name}' downloaded successfully ({len(df)} rows)")
    
    def download_tables(
        self,
        table_list: List[str],
        output_folder: Union[str, Path],
        num_partitions: int = 16,
    ) -> List[str]:
        """
        Download multiple OMOP tables.
        
        Args:
            table_list: List of table names to download
            output_folder: Output directory path
            num_partitions: Number of partitions for parallel download
            
        Returns:
            List of successfully downloaded tables
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        downloaded_tables = []
        
        for table_name in table_list:
            try:
                # Check if table has a known partition column
                partition_column = OMOP_TABLE_DICT.get(table_name)
                
                self.download_table_from_db(
                    table_name=table_name,
                    output_folder=output_folder,
                    partition_column=partition_column,
                    num_partitions=num_partitions,
                )
                
                downloaded_tables.append(table_name)
                
            except Exception as e:
                print(f"Error downloading table '{table_name}': {e}")
        
        return downloaded_tables
    
    def create_patient_splits(self, data_folder: Union[str, Path], seed: int = 42) -> None:
        """
        Create train/test patient splits from person table.
        
        Args:
            data_folder: Folder containing the downloaded OMOP tables
            seed: Random seed for reproducibility
        """
        data_folder = Path(data_folder)
        patient_splits_folder = data_folder / "patient_splits"
        
        if patient_splits_folder.exists():
            print("Patient splits already exist, skipping...")
            return
        
        # Read person table
        person_path = data_folder / "person" / "person.parquet"
        if not person_path.exists():
            print("Person table not found, cannot create patient splits")
            return
        
        # Load person IDs
        person_df = pl.read_parquet(person_path).select("person_id")
        
        # Create random splits (80/20)
        n_total = len(person_df)
        n_train = int(0.8 * n_total)
        
        # Shuffle and split
        shuffled = person_df.sample(fraction=1.0, shuffle=True, seed=seed)
        
        train_split = shuffled.head(n_train).with_columns(
            pl.lit("train").alias("split")
        )
        test_split = shuffled.tail(n_total - n_train).with_columns(
            pl.lit("test").alias("split")
        )
        
        # Combine and save
        patient_splits = pl.concat([train_split, test_split])
        
        patient_splits_folder.mkdir(parents=True, exist_ok=True)
        patient_splits.write_parquet(patient_splits_folder / "patient_splits.parquet")
        
        print(f"Created patient splits: {n_train} train, {n_total - n_train} test")


def load_parquet_tables(
    source_folder: Union[str, Path],
    table_list: List[str],
    output_folder: Union[str, Path],
) -> List[str]:
    """
    Copy parquet tables from source to output folder.
    
    This is useful when working with pre-downloaded parquet files.
    
    Args:
        source_folder: Source directory containing parquet files
        table_list: List of tables to copy
        output_folder: Output directory
        
    Returns:
        List of successfully copied tables
    """
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    copied_tables = []
    
    for table_name in table_list:
        try:
            # Look for parquet files in various possible locations
            possible_paths = [
                source_folder / table_name / f"{table_name}.parquet",
                source_folder / f"{table_name}.parquet",
                source_folder / table_name,  # Directory with multiple parquet files
            ]
            
            source_path = None
            for path in possible_paths:
                if path.exists():
                    source_path = path
                    break
            
            if source_path is None:
                print(f"Table '{table_name}' not found in source folder")
                continue
            
            # Create output directory
            output_path = output_folder / table_name
            output_path.mkdir(parents=True, exist_ok=True)
            
            if source_path.is_file():
                # Single parquet file
                df = pl.read_parquet(source_path)
                df.write_parquet(output_path / f"{table_name}.parquet")
            else:
                # Directory with multiple parquet files
                df = pl.read_parquet(source_path / "*.parquet")
                df.write_parquet(output_path / f"{table_name}.parquet")
            
            copied_tables.append(table_name)
            print(f"Table '{table_name}' copied successfully")
            
        except Exception as e:
            print(f"Error copying table '{table_name}': {e}")
    
    return copied_tables


def main():
    """Main entry point for the OMOP table download tool."""
    parser = argparse.ArgumentParser(
        description="Download OMOP CDM tables using Polars",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from database
  python download_omop_tables.py -c db_config.ini -tc person visit_occurrence -o /data/omop/
  
  # Copy from existing parquet files
  python download_omop_tables.py -s /source/omop/ -tc person visit_occurrence -o /data/omop/
        """,
    )
    
    # Create mutually exclusive group for source
    source_group = parser.add_mutually_exclusive_group(required=True)
    
    source_group.add_argument(
        "-c",
        "--credential_path",
        dest="credential_path",
        help="Path to database credentials file (INI format)",
    )
    
    source_group.add_argument(
        "-s",
        "--source_folder",
        dest="source_folder",
        help="Source folder containing parquet files",
    )
    
    parser.add_argument(
        "-tc",
        "--domain_table_list",
        dest="domain_table_list",
        nargs="+",
        action="store",
        help="List of OMOP tables to download",
        required=True,
    )
    
    parser.add_argument(
        "-o",
        "--output_folder",
        dest="output_folder",
        action="store",
        help="Output folder for downloaded tables",
        required=True,
    )
    
    parser.add_argument(
        "-p",
        "--partitions",
        dest="num_partitions",
        type=int,
        default=16,
        help="Number of partitions for parallel download (default: 16)",
    )
    
    parser.add_argument(
        "--create-splits",
        dest="create_splits",
        action="store_true",
        help="Create train/test patient splits after download",
    )
    
    args = parser.parse_args()
    
    if args.credential_path:
        # Database download mode
        config = configparser.ConfigParser()
        config.read(args.credential_path)
        db_properties = dict(config.defaults())
        
        downloader = OMOPTableDownloader(db_properties)
        downloaded_tables = downloader.download_tables(
            table_list=args.domain_table_list,
            output_folder=args.output_folder,
            num_partitions=args.num_partitions,
        )
        
        print(f"\nSuccessfully downloaded tables: {downloaded_tables}")
        
        if args.create_splits and "person" in downloaded_tables:
            downloader.create_patient_splits(args.output_folder)
    
    else:
        # Parquet copy mode
        copied_tables = load_parquet_tables(
            source_folder=args.source_folder,
            table_list=args.domain_table_list,
            output_folder=args.output_folder,
        )
        
        print(f"\nSuccessfully copied tables: {copied_tables}")
        
        if args.create_splits and "person" in copied_tables:
            downloader = OMOPTableDownloader()
            downloader.create_patient_splits(args.output_folder)


if __name__ == "__main__":
    main()