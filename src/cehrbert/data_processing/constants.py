"""
Constants and schemas for OMOP CDM v5.4.

This module contains table schemas, concept ID ranges, and other constants
used throughout the data processing pipeline.
"""

import polars as pl


# OMOP CDM v5.4 Core Table Schemas
OMOP_SCHEMAS = {
    "person": {
        "person_id": pl.Int64,
        "gender_concept_id": pl.Int64,
        "year_of_birth": pl.Int32,
        "month_of_birth": pl.Int32,
        "day_of_birth": pl.Int32,
        "birth_datetime": pl.Datetime,
        "race_concept_id": pl.Int64,
        "ethnicity_concept_id": pl.Int64,
        "location_id": pl.Int64,
        "provider_id": pl.Int64,
        "care_site_id": pl.Int64,
        "person_source_value": pl.Utf8,
        "gender_source_value": pl.Utf8,
        "gender_source_concept_id": pl.Int64,
        "race_source_value": pl.Utf8,
        "race_source_concept_id": pl.Int64,
        "ethnicity_source_value": pl.Utf8,
        "ethnicity_source_concept_id": pl.Int64,
    },
    "visit_occurrence": {
        "visit_occurrence_id": pl.Int64,
        "person_id": pl.Int64,
        "visit_concept_id": pl.Int64,
        "visit_start_date": pl.Date,
        "visit_start_datetime": pl.Datetime,
        "visit_end_date": pl.Date,
        "visit_end_datetime": pl.Datetime,
        "visit_type_concept_id": pl.Int64,
        "provider_id": pl.Int64,
        "care_site_id": pl.Int64,
        "visit_source_value": pl.Utf8,
        "visit_source_concept_id": pl.Int64,
        "admitted_from_concept_id": pl.Int64,
        "admitted_from_source_value": pl.Utf8,
        "discharged_to_concept_id": pl.Int64,
        "discharged_to_source_value": pl.Utf8,
        "preceding_visit_occurrence_id": pl.Int64,
    },
    "condition_occurrence": {
        "condition_occurrence_id": pl.Int64,
        "person_id": pl.Int64,
        "condition_concept_id": pl.Int64,
        "condition_start_date": pl.Date,
        "condition_start_datetime": pl.Datetime,
        "condition_end_date": pl.Date,
        "condition_end_datetime": pl.Datetime,
        "condition_type_concept_id": pl.Int64,
        "condition_status_concept_id": pl.Int64,
        "stop_reason": pl.Utf8,
        "provider_id": pl.Int64,
        "visit_occurrence_id": pl.Int64,
        "visit_detail_id": pl.Int64,
        "condition_source_value": pl.Utf8,
        "condition_source_concept_id": pl.Int64,
        "condition_status_source_value": pl.Utf8,
    },
    "drug_exposure": {
        "drug_exposure_id": pl.Int64,
        "person_id": pl.Int64,
        "drug_concept_id": pl.Int64,
        "drug_exposure_start_date": pl.Date,
        "drug_exposure_start_datetime": pl.Datetime,
        "drug_exposure_end_date": pl.Date,
        "drug_exposure_end_datetime": pl.Datetime,
        "verbatim_end_date": pl.Date,
        "drug_type_concept_id": pl.Int64,
        "stop_reason": pl.Utf8,
        "refills": pl.Int32,
        "quantity": pl.Float64,
        "days_supply": pl.Int32,
        "sig": pl.Utf8,
        "route_concept_id": pl.Int64,
        "lot_number": pl.Utf8,
        "provider_id": pl.Int64,
        "visit_occurrence_id": pl.Int64,
        "visit_detail_id": pl.Int64,
        "drug_source_value": pl.Utf8,
        "drug_source_concept_id": pl.Int64,
        "route_source_value": pl.Utf8,
        "dose_unit_source_value": pl.Utf8,
    },
    "procedure_occurrence": {
        "procedure_occurrence_id": pl.Int64,
        "person_id": pl.Int64,
        "procedure_concept_id": pl.Int64,
        "procedure_date": pl.Date,
        "procedure_datetime": pl.Datetime,
        "procedure_end_date": pl.Date,
        "procedure_end_datetime": pl.Datetime,
        "procedure_type_concept_id": pl.Int64,
        "modifier_concept_id": pl.Int64,
        "quantity": pl.Int32,
        "provider_id": pl.Int64,
        "visit_occurrence_id": pl.Int64,
        "visit_detail_id": pl.Int64,
        "procedure_source_value": pl.Utf8,
        "procedure_source_concept_id": pl.Int64,
        "modifier_source_value": pl.Utf8,
    },
    "measurement": {
        "measurement_id": pl.Int64,
        "person_id": pl.Int64,
        "measurement_concept_id": pl.Int64,
        "measurement_date": pl.Date,
        "measurement_datetime": pl.Datetime,
        "measurement_time": pl.Utf8,
        "measurement_type_concept_id": pl.Int64,
        "operator_concept_id": pl.Int64,
        "value_as_number": pl.Float64,
        "value_as_concept_id": pl.Int64,
        "unit_concept_id": pl.Int64,
        "range_low": pl.Float64,
        "range_high": pl.Float64,
        "provider_id": pl.Int64,
        "visit_occurrence_id": pl.Int64,
        "visit_detail_id": pl.Int64,
        "measurement_source_value": pl.Utf8,
        "measurement_source_concept_id": pl.Int64,
        "unit_source_value": pl.Utf8,
        "unit_source_concept_id": pl.Int64,
        "value_source_value": pl.Utf8,
        "measurement_event_id": pl.Int64,
        "meas_event_field_concept_id": pl.Int64,
    },
    "observation": {
        "observation_id": pl.Int64,
        "person_id": pl.Int64,
        "observation_concept_id": pl.Int64,
        "observation_date": pl.Date,
        "observation_datetime": pl.Datetime,
        "observation_type_concept_id": pl.Int64,
        "value_as_number": pl.Float64,
        "value_as_string": pl.Utf8,
        "value_as_concept_id": pl.Int64,
        "qualifier_concept_id": pl.Int64,
        "unit_concept_id": pl.Int64,
        "provider_id": pl.Int64,
        "visit_occurrence_id": pl.Int64,
        "visit_detail_id": pl.Int64,
        "observation_source_value": pl.Utf8,
        "observation_source_concept_id": pl.Int64,
        "unit_source_value": pl.Utf8,
        "qualifier_source_value": pl.Utf8,
        "value_source_value": pl.Utf8,
        "observation_event_id": pl.Int64,
        "obs_event_field_concept_id": pl.Int64,
    },
}


# Tables with their primary key columns
OMOP_PRIMARY_KEYS = {
    "person": "person_id",
    "visit_occurrence": "visit_occurrence_id",
    "condition_occurrence": "condition_occurrence_id",
    "drug_exposure": "drug_exposure_id",
    "procedure_occurrence": "procedure_occurrence_id",
    "measurement": "measurement_id",
    "observation": "observation_id",
    "death": "person_id",
    "note": "note_id",
    "note_nlp": "note_nlp_id",
    "specimen": "specimen_id",
    "device_exposure": "device_exposure_id",
}


# Domain tables used in CEHR-BERT
CEHRBERT_DOMAIN_TABLES = [
    "condition_occurrence",
    "drug_exposure",
    "procedure_occurrence",
    "measurement",
    "observation",
]


# Concept ID ranges for different domains (OMOP convention)
CONCEPT_ID_RANGES = {
    "condition": (1, 999999999),  # Conditions
    "drug": (1000000000, 1999999999),  # Drugs
    "procedure": (2000000000, 2999999999),  # Procedures
    "observation": (3000000000, 3999999999),  # Observations
    "measurement": (3000000000, 3999999999),  # Measurements (shared with observation)
    "visit": (9201, 9203),  # Standard visit concepts
}


# Standard concept IDs for visits
VISIT_CONCEPTS = {
    "inpatient": 9201,
    "outpatient": 9202,
    "emergency": 9203,
}


# Gender concept IDs
GENDER_CONCEPTS = {
    "male": 8507,
    "female": 8532,
}


# Special tokens used in CEHR-BERT
CEHRBERT_SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "[VS]": 5,  # Visit Start
    "[VE]": 6,  # Visit End
    "[ATT]": 7,  # Artificial Time Token
}


# Date format used in OMOP
OMOP_DATE_FORMAT = "%Y-%m-%d"
OMOP_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# Default values
DEFAULT_MIN_PATIENT_EVENTS = 5
DEFAULT_MIN_CONCEPT_COUNT = 100
DEFAULT_SEQUENCE_LENGTH = 512
DEFAULT_TRAIN_TEST_SPLIT = 0.8


# Temporal token types
TEMPORAL_TOKEN_TYPES = ["VS", "VE", "ATT"]


# ATT (Artificial Time Token) granularity options
ATT_TYPES = {
    "day": 1,
    "week": 7,
    "month": 30,
    "cehr_bert": "variable",  # Original CEHR-BERT logic
}


def get_domain_from_concept_id(concept_id: int) -> str:
    """
    Determine the domain of a concept based on its ID range.
    
    Args:
        concept_id: OMOP concept ID
        
    Returns:
        Domain name (condition, drug, procedure, etc.)
    """
    for domain, (min_id, max_id) in CONCEPT_ID_RANGES.items():
        if min_id <= concept_id <= max_id:
            return domain
    return "unknown"


def is_valid_omop_date(date_str: str) -> bool:
    """
    Check if a date string is in valid OMOP format.
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        from datetime import datetime
        datetime.strptime(date_str, OMOP_DATE_FORMAT)
        return True
    except ValueError:
        return False