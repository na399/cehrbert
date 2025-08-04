"""
OMOP CDM v5.4 schema validation utilities.

This module provides functions to validate OMOP tables against the standard schema
and check data quality.
"""

from typing import Dict, List, Optional, Tuple

import polars as pl

from cehrbert.data_processing.constants import (
    OMOP_SCHEMAS,
    OMOP_PRIMARY_KEYS,
    CONCEPT_ID_RANGES,
    get_domain_from_concept_id,
)


class OMOPSchemaValidator:
    """Validates OMOP CDM tables against v5.4 schema."""
    
    def __init__(self, strict: bool = False):
        """
        Initialize the validator.
        
        Args:
            strict: If True, requires all columns to be present.
                   If False, only checks that present columns have correct types.
        """
        self.strict = strict
        self.validation_results = {}
    
    def validate_table(
        self, 
        df: pl.DataFrame, 
        table_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate a single OMOP table.
        
        Args:
            df: DataFrame to validate
            table_name: Name of the OMOP table
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if table_name not in OMOP_SCHEMAS:
            issues.append(f"Unknown OMOP table: {table_name}")
            return False, issues
        
        expected_schema = OMOP_SCHEMAS[table_name]
        actual_schema = df.schema
        
        # Check for required columns
        if self.strict:
            missing_cols = set(expected_schema.keys()) - set(actual_schema.keys())
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")
        
        # Check column types
        for col_name, expected_type in expected_schema.items():
            if col_name in actual_schema:
                actual_type = actual_schema[col_name]
                if not self._types_compatible(expected_type, actual_type):
                    issues.append(
                        f"Column '{col_name}' has incorrect type: "
                        f"expected {expected_type}, got {actual_type}"
                    )
        
        # Check for extra columns (warning only)
        extra_cols = set(actual_schema.keys()) - set(expected_schema.keys())
        if extra_cols and self.strict:
            issues.append(f"Extra columns found (warning): {extra_cols}")
        
        # Validate primary key
        if table_name in OMOP_PRIMARY_KEYS:
            pk_col = OMOP_PRIMARY_KEYS[table_name]
            if pk_col in df.columns:
                # Check for nulls in primary key
                null_count = df[pk_col].null_count()
                if null_count > 0:
                    issues.append(f"Primary key '{pk_col}' contains {null_count} null values")
                
                # Check for duplicates
                duplicate_count = len(df) - df[pk_col].n_unique()
                if duplicate_count > 0:
                    issues.append(f"Primary key '{pk_col}' contains {duplicate_count} duplicate values")
        
        is_valid = len(issues) == 0
        self.validation_results[table_name] = {
            "valid": is_valid,
            "issues": issues,
            "row_count": len(df),
            "column_count": len(df.columns),
        }
        
        return is_valid, issues
    
    def validate_relationships(
        self,
        tables: Dict[str, pl.DataFrame]
    ) -> Tuple[bool, List[str]]:
        """
        Validate foreign key relationships between tables.
        
        Args:
            tables: Dictionary of table_name -> DataFrame
            
        Returns:
            Tuple of (all_valid, list_of_issues)
        """
        issues = []
        
        # Check person_id references
        if "person" in tables:
            person_ids = set(tables["person"]["person_id"].unique().to_list())
            
            for table_name in ["visit_occurrence", "condition_occurrence", 
                             "drug_exposure", "procedure_occurrence", 
                             "measurement", "observation"]:
                if table_name in tables and "person_id" in tables[table_name].columns:
                    table_person_ids = set(tables[table_name]["person_id"].unique().to_list())
                    orphaned = table_person_ids - person_ids
                    if orphaned:
                        issues.append(
                            f"Table '{table_name}' contains {len(orphaned)} "
                            f"person_ids not in person table"
                        )
        
        # Check visit_occurrence_id references
        if "visit_occurrence" in tables:
            visit_ids = set(tables["visit_occurrence"]["visit_occurrence_id"].unique().to_list())
            
            for table_name in ["condition_occurrence", "drug_exposure", 
                             "procedure_occurrence", "measurement", "observation"]:
                if table_name in tables and "visit_occurrence_id" in tables[table_name].columns:
                    # Filter out nulls (visits are optional)
                    table_visit_ids = tables[table_name].filter(
                        pl.col("visit_occurrence_id").is_not_null()
                    )["visit_occurrence_id"].unique().to_list()
                    
                    orphaned = set(table_visit_ids) - visit_ids
                    if orphaned:
                        issues.append(
                            f"Table '{table_name}' contains {len(orphaned)} "
                            f"visit_occurrence_ids not in visit_occurrence table"
                        )
        
        return len(issues) == 0, issues
    
    def validate_concept_ids(
        self,
        df: pl.DataFrame,
        table_name: str,
        concept_column: str = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate concept IDs are in expected ranges.
        
        Args:
            df: DataFrame to validate
            table_name: Name of the table
            concept_column: Specific concept column to check (default: main concept column)
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Determine concept column
        if concept_column is None:
            concept_column_map = {
                "condition_occurrence": "condition_concept_id",
                "drug_exposure": "drug_concept_id",
                "procedure_occurrence": "procedure_concept_id",
                "measurement": "measurement_concept_id",
                "observation": "observation_concept_id",
            }
            concept_column = concept_column_map.get(table_name)
        
        if concept_column and concept_column in df.columns:
            # Get non-null concept IDs
            concept_ids = df.filter(
                pl.col(concept_column).is_not_null()
            )[concept_column]
            
            if len(concept_ids) > 0:
                # Check if concept IDs are positive
                negative_count = (concept_ids < 0).sum()
                if negative_count > 0:
                    issues.append(
                        f"Column '{concept_column}' contains {negative_count} negative concept IDs"
                    )
                
                # For domain-specific tables, check if concepts are in expected range
                expected_domain = table_name.replace("_occurrence", "").replace("_exposure", "")
                if expected_domain in CONCEPT_ID_RANGES:
                    min_id, max_id = CONCEPT_ID_RANGES[expected_domain]
                    out_of_range = ((concept_ids < min_id) | (concept_ids > max_id)).sum()
                    if out_of_range > 0:
                        issues.append(
                            f"Column '{concept_column}' contains {out_of_range} "
                            f"concept IDs outside expected range [{min_id}, {max_id}]"
                        )
        
        return len(issues) == 0, issues
    
    def validate_dates(
        self,
        df: pl.DataFrame,
        table_name: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate date columns for reasonable values.
        
        Args:
            df: DataFrame to validate
            table_name: Name of the table
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Date columns to check
        date_columns = []
        for col, dtype in df.schema.items():
            if dtype in [pl.Date, pl.Datetime]:
                date_columns.append(col)
        
        for col in date_columns:
            # Check for future dates
            if df[col].dtype == pl.Date:
                future_count = (df[col] > pl.date(2025, 12, 31)).sum()
            else:
                future_count = (df[col] > pl.datetime(2025, 12, 31)).sum()
                
            if future_count > 0:
                issues.append(f"Column '{col}' contains {future_count} future dates (after 2025)")
            
            # Check for very old dates
            if df[col].dtype == pl.Date:
                old_count = (df[col] < pl.date(1900, 1, 1)).sum()
            else:
                old_count = (df[col] < pl.datetime(1900, 1, 1)).sum()
                
            if old_count > 0:
                issues.append(f"Column '{col}' contains {old_count} dates before 1900")
        
        # Check date order (start should be before end)
        date_pairs = [
            ("visit_start_date", "visit_end_date"),
            ("condition_start_date", "condition_end_date"),
            ("drug_exposure_start_date", "drug_exposure_end_date"),
            ("procedure_date", "procedure_end_date"),
        ]
        
        for start_col, end_col in date_pairs:
            if start_col in df.columns and end_col in df.columns:
                # Filter to non-null pairs
                valid_pairs = df.filter(
                    pl.col(start_col).is_not_null() & pl.col(end_col).is_not_null()
                )
                
                if len(valid_pairs) > 0:
                    invalid_count = (valid_pairs[start_col] > valid_pairs[end_col]).sum()
                    if invalid_count > 0:
                        issues.append(
                            f"Found {invalid_count} records where {start_col} > {end_col}"
                        )
        
        return len(issues) == 0, issues
    
    def _types_compatible(self, expected: pl.DataType, actual: pl.DataType) -> bool:
        """Check if two data types are compatible."""
        # Exact match
        if expected == actual:
            return True
        
        # Integer compatibility
        int_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
        if expected in int_types and actual in int_types:
            return True
        
        # Float compatibility
        float_types = {pl.Float32, pl.Float64}
        if expected in float_types and actual in float_types:
            return True
        
        # Date/datetime compatibility (datetime can be used for date columns)
        if expected == pl.Date and actual == pl.Datetime:
            return True
        
        return False
    
    def generate_report(self) -> str:
        """Generate a validation report for all validated tables."""
        report = ["OMOP CDM Validation Report", "=" * 50, ""]
        
        for table_name, results in self.validation_results.items():
            report.append(f"\nTable: {table_name}")
            report.append(f"  Rows: {results['row_count']:,}")
            report.append(f"  Columns: {results['column_count']}")
            report.append(f"  Valid: {'✓' if results['valid'] else '✗'}")
            
            if results['issues']:
                report.append("  Issues:")
                for issue in results['issues']:
                    report.append(f"    - {issue}")
        
        return "\n".join(report)


def quick_validate(
    df: pl.DataFrame,
    table_name: str,
    check_concepts: bool = True,
    check_dates: bool = True
) -> Dict[str, any]:
    """
    Quick validation of an OMOP table.
    
    Args:
        df: DataFrame to validate
        table_name: Name of the OMOP table
        check_concepts: Whether to validate concept IDs
        check_dates: Whether to validate dates
        
    Returns:
        Dictionary with validation results
    """
    validator = OMOPSchemaValidator(strict=False)
    
    results = {
        "table_name": table_name,
        "row_count": len(df),
        "column_count": len(df.columns),
    }
    
    # Schema validation
    schema_valid, schema_issues = validator.validate_table(df, table_name)
    results["schema_valid"] = schema_valid
    results["schema_issues"] = schema_issues
    
    # Concept validation
    if check_concepts:
        concept_valid, concept_issues = validator.validate_concept_ids(df, table_name)
        results["concept_valid"] = concept_valid
        results["concept_issues"] = concept_issues
    
    # Date validation
    if check_dates:
        date_valid, date_issues = validator.validate_dates(df, table_name)
        results["date_valid"] = date_valid
        results["date_issues"] = date_issues
    
    # Overall validity
    results["valid"] = all(
        results.get(f"{check}_valid", True) 
        for check in ["schema", "concept", "date"]
    )
    
    return results