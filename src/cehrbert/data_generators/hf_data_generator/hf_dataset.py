from typing import Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import (
    DatasetMapping,
    HFFineTuningMapping,
    HFTokenizationMapping,
    SortPatientSequenceMapping,
)
from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments

CEHRBERT_COLUMNS = [
    "person_id",
    "concept_ids",
    "ages",
    "dates",
    "visit_segments",
    "visit_concept_orders",
    "concept_values",
    "concept_value_masks",
    "mlm_skip_values",
    "num_of_concepts",
    "num_of_visits",
    "number_as_values",
    "concept_as_values",
]

TRANSFORMER_COLUMNS = ["input_ids", "labels"]

FINETUNING_COLUMNS = ["age_at_index", "classifier_label", "index_date", "person_id"]


def create_cehrbert_pretraining_dataset(
    dataset: Union[Dataset, DatasetDict],
    concept_tokenizer: CehrBertTokenizer,
    data_args: DataTrainingArguments,
    cache_file_collector: Optional[CacheFileCollector] = None,
) -> Dataset:
    required_columns = TRANSFORMER_COLUMNS + CEHRBERT_COLUMNS

    # Remove patients without any records
    dataset = filter_dataset(dataset, data_args)
    # Apply tokenization mapping
    mapping_functions = [
        # SortPatientSequenceMapping(),
        HFTokenizationMapping(concept_tokenizer, True),
    ]

    for mapping_function in mapping_functions:
        dataset = apply_cehrbert_dataset_mapping(
            dataset,
            mapping_function,
            num_proc=data_args.preprocessing_num_workers,
            batch_size=data_args.preprocessing_batch_size,
            streaming=data_args.streaming,
            cache_file_collector=cache_file_collector,
        )

    if not data_args.streaming:
        if isinstance(dataset, DatasetDict):
            all_columns = dataset["train"].column_names
        else:
            all_columns = dataset.column_names
        columns_to_remove = [_ for _ in all_columns if _ not in required_columns]
        dataset = dataset.remove_columns(columns_to_remove)

    return dataset


def create_cehrbert_finetuning_dataset(
    dataset: Union[Dataset, DatasetDict],
    concept_tokenizer: CehrBertTokenizer,
    data_args: DataTrainingArguments,
    cache_file_collector: Optional[CacheFileCollector] = None,
) -> Dataset:
    required_columns = TRANSFORMER_COLUMNS + CEHRBERT_COLUMNS + FINETUNING_COLUMNS

    # Remove patients without any records
    dataset = filter_dataset(dataset, data_args)
    # Apply fine-tuning and tokenization mappings
    mapping_functions = [
        HFFineTuningMapping(),
        # SortPatientSequenceMapping(),
        HFTokenizationMapping(concept_tokenizer, False),
    ]

    for mapping_function in mapping_functions:
        dataset = apply_cehrbert_dataset_mapping(
            dataset,
            mapping_function,
            num_proc=data_args.preprocessing_num_workers,
            batch_size=data_args.preprocessing_batch_size,
            streaming=data_args.streaming,
            cache_file_collector=cache_file_collector,
        )

    if not data_args.streaming:
        if isinstance(dataset, DatasetDict):
            all_columns = dataset["train"].column_names
        else:
            all_columns = dataset.column_names
        columns_to_remove = [_ for _ in all_columns if _ not in required_columns]
        dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def filter_dataset(dataset: Union[Dataset, DatasetDict], data_args: DataTrainingArguments):
    # Remove patients without any records and fix data issues
    def filter_and_fix_batch(batch):
        valid_indices = []
        for i in range(len(batch["num_of_concepts"])):
            # Skip if no concepts
            if batch["num_of_concepts"][i] <= 0:
                continue
                
            # Check concept_values and concept_value_masks dimensions
            if "concept_values" in batch and "concept_value_masks" in batch:
                concept_ids_len = len(batch["concept_ids"][i])
                concept_values_len = len(batch["concept_values"][i])
                concept_value_masks_len = len(batch["concept_value_masks"][i])
                
                # Fix single-element arrays by expanding them
                if concept_values_len == 1 and concept_ids_len > 1:
                    batch["concept_values"][i] = [batch["concept_values"][i][0]] * concept_ids_len
                    batch["concept_value_masks"][i] = [batch["concept_value_masks"][i][0]] * concept_ids_len
                elif concept_values_len != concept_ids_len or concept_value_masks_len != concept_ids_len:
                    # Skip samples with unfixable dimension mismatches
                    continue
            
            valid_indices.append(i)
        
        # Return only valid samples
        return {key: [values[i] for i in valid_indices] for key, values in batch.items()}
    
    # check if DatatsetDict or IterableDatasetDict, if so, filter each dataset
    if isinstance(dataset, DatasetDict) and data_args.streaming:
        for key in dataset.keys():
            dataset[key] = dataset[key].map(
                filter_and_fix_batch,
                batched=True,
                batch_size=data_args.preprocessing_batch_size,
                remove_columns=[],
            )
    else:
        dataset = dataset.map(
            filter_and_fix_batch,
            num_proc=data_args.preprocessing_num_workers if not data_args.streaming else None,
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            remove_columns=[],
        )
    return dataset


def apply_cehrbert_dataset_mapping(
    dataset: Union[DatasetDict, Dataset, IterableDataset, IterableDatasetDict],
    mapping_function: DatasetMapping,
    batch_size: int = 128,
    num_proc: int = 1,
    streaming: bool = False,
    cache_file_collector: Optional[CacheFileCollector] = None,
):
    if streaming:
        if isinstance(dataset, DatasetDict):
            for dataset_name in dataset.keys():
                dataset[dataset_name] = dataset[dataset_name].map(
                    mapping_function.batch_transform,
                    batched=True,
                    batch_size=batch_size,
                )
                if mapping_function.remove_columns():
                    dataset[dataset_name] = dataset[dataset_name].remove_columns(mapping_function.remove_columns())
        else:
            dataset = dataset.map(mapping_function.batch_transform, batched=True, batch_size=batch_size)
            if mapping_function.remove_columns():
                dataset = dataset.remove_columns(mapping_function.remove_columns())
    else:
        dataset = dataset.map(
            mapping_function.batch_transform,
            num_proc=num_proc,
            batched=True,
            batch_size=batch_size,
        )
        if mapping_function.remove_columns():
            dataset = dataset.remove_columns(mapping_function.remove_columns())

        if cache_file_collector:
            cache_file_collector.add_cache_files(dataset)

    return dataset
