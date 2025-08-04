#!/usr/bin/env python3
"""
Utility functions to ensure data type compatibility with HuggingFace datasets
"""

import polars as pl
from typing import Optional

def ensure_huggingface_compatibility(df: pl.DataFrame, include_extra_columns: bool = False) -> pl.DataFrame:
    """
    Ensure DataFrame has compatible data types for HuggingFace datasets.
    
    Args:
        df: Input DataFrame
        include_extra_columns: Whether to add extra columns for compatibility
    
    Returns:
        DataFrame with compatible types
    """
    # Fix string_view in lists (already handled by explicit cast)
    if 'concept_ids' in df.columns:
        df = df.with_columns(
            pl.col('concept_ids').list.eval(pl.element().cast(pl.Utf8)).alias('concept_ids')
        )
    
    # Convert UInt32 to Int32
    uint32_columns = [col for col, dtype in zip(df.columns, df.dtypes) if str(dtype) == 'UInt32']
    if uint32_columns:
        df = df.with_columns([
            pl.col(col).cast(pl.Int32).alias(col) for col in uint32_columns
        ])
    
    # Convert list of UInt32 to list of Int32
    list_uint32_columns = [col for col, dtype in zip(df.columns, df.dtypes) if str(dtype) == 'List(UInt32)']
    if list_uint32_columns:
        df = df.with_columns([
            pl.col(col).list.eval(pl.element().cast(pl.Int32)).alias(col) for col in list_uint32_columns
        ])
    
    # Ensure ages are Int64 (matching sample data)
    if 'ages' in df.columns:
        df = df.with_columns(
            pl.col('ages').list.eval(pl.element().cast(pl.Int64)).alias('ages')
        )
    
    # Ensure dates are Int32 (matching sample data)
    if 'dates' in df.columns:
        df = df.with_columns(
            pl.col('dates').list.eval(pl.element().cast(pl.Int32)).alias('dates')
        )
    
    # Ensure concept_values and concept_value_masks are Int32 (matching sample data)
    if 'concept_values' in df.columns:
        df = df.with_columns(
            pl.col('concept_values').list.eval(pl.element().cast(pl.Int32)).alias('concept_values')
        )
    
    if 'concept_value_masks' in df.columns:
        df = df.with_columns(
            pl.col('concept_value_masks').list.eval(pl.element().cast(pl.Int32)).alias('concept_value_masks')
        )
    
    # Add extra columns if needed for compatibility
    if include_extra_columns:
        if 'cohort_member_id' not in df.columns:
            df = df.with_columns(pl.col('person_id').alias('cohort_member_id'))
        
        if 'orders' not in df.columns and 'visit_concept_orders' in df.columns:
            df = df.with_columns(pl.col('visit_concept_orders').alias('orders'))
        
        if 'visit_concept_ids' not in df.columns and 'visit_segments' in df.columns:
            # Create dummy visit_concept_ids (same length as other lists)
            df = df.with_columns(
                pl.col('visit_segments').list.eval(pl.lit(0)).alias('visit_concept_ids')
            )
        
        if '__index_level_0__' not in df.columns:
            df = df.with_columns(pl.arange(0, len(df)).cast(pl.Int64).alias('__index_level_0__'))
    
    # Ensure column order matches sample if needed
    expected_columns = [
        'cohort_member_id', 'person_id', 'concept_ids', 'visit_segments', 
        'orders', 'dates', 'ages', 'visit_concept_orders', 'num_of_visits',
        'num_of_concepts', 'concept_value_masks', 'concept_values', 
        'mlm_skip_values', 'visit_concept_ids', '__index_level_0__'
    ]
    
    if include_extra_columns:
        # Reorder columns to match expected order
        existing_columns = [col for col in expected_columns if col in df.columns]
        extra_columns = [col for col in df.columns if col not in expected_columns]
        df = df.select(existing_columns + extra_columns)
    
    return df