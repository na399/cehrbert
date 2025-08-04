"""
OMOP CDM data loader with support for both database connections and parquet files.

This module provides a unified interface for loading OMOP tables from various sources
and includes schema validation for OMOP CDM v5.4.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import polars as pl
import connectorx as cx


class OMOPDataLoader:
    """
    Unified data loader for OMOP CDM tables.
    
    Supports loading from:
    - Direct database connections
    - Parquet files
    - Mixed sources (some tables from DB, others from parquet)
    """
    
    def __init__(self, source: Union[str, Path, Dict[str, str]]):
        """
        Initialize the OMOP data loader.
        
        Args:
            source: Can be:
                - str: Database connection string
                - Path: Directory containing parquet files
                - Dict: Mapping of data source configuration
                    {
                        "type": "database" or "parquet" or "mixed",
                        "connection_string": "...",  # for database
                        "parquet_dir": "/path/to/parquet",  # for parquet
                        "table_sources": {  # for mixed mode
                            "person": "database",
                            "visit_occurrence": "parquet",
                            ...
                        }
                    }
        """
        self.source = source
        self._connection_string = None
        self._parquet_dir = None
        self._table_sources = {}
        
        self._parse_source()
    
    def _parse_source(self) -> None:
        """Parse and validate the data source configuration."""
        if isinstance(self.source, str):
            # Assume it's a connection string
            self._connection_string = self.source
            self._source_type = "database"
            
        elif isinstance(self.source, Path) or (isinstance(self.source, str) and "/" in self.source):
            # It's a file path
            self._parquet_dir = Path(self.source)
            if not self._parquet_dir.exists():
                raise ValueError(f"Parquet directory does not exist: {self._parquet_dir}")
            self._source_type = "parquet"
            
        elif isinstance(self.source, dict):
            # Complex configuration
            source_type = self.source.get("type", "").lower()
            
            if source_type == "database":
                self._connection_string = self.source.get("connection_string")
                if not self._connection_string:
                    raise ValueError("Database source requires 'connection_string'")
                self._source_type = "database"
                
            elif source_type == "parquet":
                self._parquet_dir = Path(self.source.get("parquet_dir", ""))
                if not self._parquet_dir.exists():
                    raise ValueError(f"Parquet directory does not exist: {self._parquet_dir}")
                self._source_type = "parquet"
                
            elif source_type == "mixed":
                self._connection_string = self.source.get("connection_string")
                self._parquet_dir = Path(self.source.get("parquet_dir", ""))
                self._table_sources = self.source.get("table_sources", {})
                self._source_type = "mixed"
                
            else:
                raise ValueError(f"Unknown source type: {source_type}")
        else:
            raise ValueError("Invalid source configuration")
    
    def load_table(self, table_name: str, lazy: bool = True) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Load an OMOP table.
        
        Args:
            table_name: Name of the OMOP table
            lazy: Whether to return a LazyFrame (True) or DataFrame (False)
            
        Returns:
            Polars DataFrame or LazyFrame containing the table data
        """
        # Determine source for this table
        if self._source_type == "database":
            return self._load_from_database(table_name, lazy)
        elif self._source_type == "parquet":
            return self._load_from_parquet(table_name, lazy)
        elif self._source_type == "mixed":
            table_source = self._table_sources.get(table_name, "parquet")
            if table_source == "database":
                return self._load_from_database(table_name, lazy)
            else:
                return self._load_from_parquet(table_name, lazy)
    
    def _load_from_database(self, table_name: str, lazy: bool) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Load table from database."""
        query = f"SELECT * FROM {table_name}"
        
        # ConnectorX always returns eager DataFrame
        df = cx.read_sql(self._connection_string, query, return_type="polars")
        
        if lazy:
            return df.lazy()
        return df
    
    def _load_from_parquet(self, table_name: str, lazy: bool) -> Union[pl.DataFrame, pl.LazyFrame]:
        """Load table from parquet files."""
        # Try different possible locations
        possible_paths = [
            self._parquet_dir / table_name / f"{table_name}.parquet",
            self._parquet_dir / f"{table_name}.parquet",
            self._parquet_dir / table_name,  # Directory with multiple files
        ]
        
        for path in possible_paths:
            if path.exists():
                if path.is_file():
                    if lazy:
                        return pl.scan_parquet(path)
                    else:
                        return pl.read_parquet(path)
                else:
                    # Directory with multiple parquet files
                    if lazy:
                        return pl.scan_parquet(path / "*.parquet")
                    else:
                        return pl.read_parquet(path / "*.parquet")
        
        raise FileNotFoundError(f"Table '{table_name}' not found in {self._parquet_dir}")
    
    def load_tables(
        self, 
        table_names: List[str], 
        lazy: bool = True
    ) -> Dict[str, Union[pl.DataFrame, pl.LazyFrame]]:
        """
        Load multiple OMOP tables.
        
        Args:
            table_names: List of table names to load
            lazy: Whether to return LazyFrames (True) or DataFrames (False)
            
        Returns:
            Dictionary mapping table names to their data
        """
        tables = {}
        
        for table_name in table_names:
            try:
                tables[table_name] = self.load_table(table_name, lazy)
                print(f"Loaded table: {table_name}")
            except Exception as e:
                print(f"Failed to load table '{table_name}': {e}")
        
        return tables
    
    def validate_schema(self, table_name: str, schema: Dict[str, pl.DataType]) -> bool:
        """
        Validate table schema against expected OMOP CDM schema.
        
        Args:
            table_name: Name of the table to validate
            schema: Expected schema as dict of column_name: polars_dtype
            
        Returns:
            True if schema matches, False otherwise
        """
        try:
            # Load just the schema (no data)
            if self._source_type == "parquet" or (
                self._source_type == "mixed" and 
                self._table_sources.get(table_name, "parquet") == "parquet"
            ):
                # For parquet, we can get schema without loading data
                df = self.load_table(table_name, lazy=True)
                actual_schema = df.schema
            else:
                # For database, we need to load a small sample
                query = f"SELECT * FROM {table_name} LIMIT 0"
                df = cx.read_sql(self._connection_string, query, return_type="polars")
                actual_schema = df.schema
            
            # Check if all expected columns exist with correct types
            for col_name, expected_type in schema.items():
                if col_name not in actual_schema:
                    print(f"Missing column '{col_name}' in table '{table_name}'")
                    return False
                
                actual_type = actual_schema[col_name]
                if actual_type != expected_type:
                    print(
                        f"Type mismatch for column '{col_name}' in table '{table_name}': "
                        f"expected {expected_type}, got {actual_type}"
                    )
                    # Some type mismatches are acceptable (e.g., Int32 vs Int64)
                    if not self._types_compatible(expected_type, actual_type):
                        return False
            
            return True
            
        except Exception as e:
            print(f"Error validating schema for table '{table_name}': {e}")
            return False
    
    def _types_compatible(self, expected: pl.DataType, actual: pl.DataType) -> bool:
        """Check if two data types are compatible."""
        # Handle integer type variations
        int_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
        if expected in int_types and actual in int_types:
            return True
        
        # Handle float variations
        float_types = {pl.Float32, pl.Float64}
        if expected in float_types and actual in float_types:
            return True
        
        # Handle string variations
        if expected == pl.Utf8 and actual == pl.Utf8:
            return True
        
        return False
    
    def get_row_count(self, table_name: str) -> int:
        """Get the number of rows in a table."""
        df = self.load_table(table_name, lazy=True)
        return df.select(pl.count()).collect().item()
    
    def get_table_info(self, table_name: str) -> Dict[str, any]:
        """
        Get information about a table.
        
        Returns:
            Dictionary with:
                - row_count: Number of rows
                - schema: Column names and types
                - memory_usage: Estimated memory usage (for parquet)
        """
        df = self.load_table(table_name, lazy=True)
        
        info = {
            "row_count": self.get_row_count(table_name),
            "schema": df.schema,
            "columns": list(df.schema.keys()),
        }
        
        # For parquet files, we can get file size
        if self._source_type in ["parquet", "mixed"]:
            try:
                path = None
                possible_paths = [
                    self._parquet_dir / table_name / f"{table_name}.parquet",
                    self._parquet_dir / f"{table_name}.parquet",
                ]
                
                for p in possible_paths:
                    if p.exists() and p.is_file():
                        path = p
                        break
                
                if path:
                    file_size_mb = path.stat().st_size / (1024 * 1024)
                    info["file_size_mb"] = round(file_size_mb, 2)
                    
            except Exception:
                pass
        
        return info