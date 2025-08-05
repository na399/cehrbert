# CEHR-BERT

[![PyPI - Version](https://img.shields.io/pypi/v/cehrbert)](https://pypi.org/project/cehrbert/)
![Python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)
[![tests](https://github.com/cumc-dbmi/cehrbert/actions/workflows/tests.yml/badge.svg)](https://github.com/cumc-dbmi/cehrbert/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/cumc-dbmi/cehrbert/blob/main/LICENSE)
[![contributors](https://img.shields.io/github/contributors/cumc-dbmi/cehrbert.svg)](https://github.com/cumc-dbmi/cehrbert/graphs/contributors)


CEHR-BERT is a large language model developed for the structured EHR data, the work has been published
at https://proceedings.mlr.press/v158/pang21a.html. CEHR-BERT focuses exclusively on the structured EHR data in the
OMOP CDM v5.4 format, which is a common data model used to support observational studies and managed by the Observational Health
Data Science and Informatics (OHDSI) open-science community.
There are three major components in CEHR-BERT, data generation, model pre-training, and model evaluation with
fine-tuning, those components work in conjunction to provide an end-to-end model evaluation framework. The CEHR-BERT
framework is designed to be extensible, users could write their
own [pretraining models](trainers/README.md), [evaluation procedures](evaluations/README.md),
and [downstream prediction tasks](spark_apps/README.md) by extending the abstract classes, see click on the links for
more details. For a quick start, navigate to the [Get Started](#getting-started) section.

## Patient Representation

For each patient, all medical codes were aggregated and constructed into a sequence chronologically.
In order to incorporate temporal information, we inserted an artificial time token (ATT) between two neighboring visits
based on their time interval.
The following logic was used for creating ATTs based on the following time intervals between visits, if less than 28
days, ATTs take on the form of $W_n$ where n represents the week number ranging from 0-3 (e.g. $W_1$); 2) if between 28
days and 365 days, ATTs are in the form of **$M_n$** where n represents the month number ranging from 1-11 e.g $M_{11}$;

3) beyond 365 days then a **LT** (Long Term) token is inserted. In addition, we added two more special tokens — **VS**
   and **VE** to represent the start and the end of a visit to explicitly define the visit segment, where all the
   concepts
   associated with the visit are subsumed by **VS** and **VE**.

!["patient_representation"](https://raw.githubusercontent.com/cumc-dbmi/cehr-bert/main/images/tokenization_att_generation.png)

## Model Architecture

Overview of our BERT architecture on structured EHR data. To distinguish visit boundaries, visit segment embeddings are
added to concept embeddings. Next, both visit embeddings and concept embeddings go through a temporal transformation,
where concept, age and time embeddings are concatenated together. The concatenated embeddings are then fed into a fully
connected layer. This temporal concept embedding becomes the input to BERT. We used the BERT learning objective Masked
Language Model as the primary learning objective and introduced an EHR specific secondary learning objective visit type
prediction.

!["cehr-bert architecture diagram"](https://raw.githubusercontent.com/cumc-dbmi/cehr-bert/main/images/cehr_bert_architecture.png)

## Pre-requisite

The project is built in python 3.10, and project dependency needs to be installed

Create a new Python virtual environment using uv

```bash
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Build the project

```bash
uv pip install -e .[dev]
```

For GPU support with CUDA 12.8:
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Key Updates in This Version

### Data Processing Modernization
CEHR-BERT now uses modern data processing libraries:
- **Polars**: Replaces PySpark for efficient data processing with native Rust performance
- **ConnectorX**: High-performance database connectivity for direct OMOP table access
- **PyTorch/HuggingFace**: All models are now based on the HuggingFace transformers library

### New Features
- Direct database connections to OMOP CDM v5.4 databases (PostgreSQL, MySQL, SQL Server, etc.)
- Improved data processing performance with Polars
- Streaming data processing for large datasets
- Simplified installation without Spark dependencies
- GPU support with PyTorch CUDA 12.8

### Simplified Architecture
- Removed TensorFlow/Keras dependencies - all models now use PyTorch
- Integrated core data processing functionality (previously in cehrbert_data)
- Focus on OMOP CDM v5.4 format exclusively
- Streamlined configuration with YAML files

## Instructions for Use with OMOP CDM v5.4

### Step 1. Download OMOP tables as parquet files
CEHR-BERT now uses Polars and ConnectorX for efficient data processing. You can either:
- Connect directly to your OMOP database
- Use pre-exported parquet files

To download OMOP tables from your database:
```bash
python -u -m cehrbert.tools.download_omop_tables -c db_properties.ini \
   -tc person visit_occurrence condition_occurrence procedure_occurrence \
   drug_exposure measurement observation_period \
   concept concept_relationship concept_ancestor \
   -o data/omop_test/
```

Update `db_properties.ini` with your database connection details:
```ini
[database]
host = your_host
port = your_port
database = your_database
username = your_username
password = your_password
driver = postgresql  # or mysql, mssql, etc.
```

We have prepared a synthea dataset with 1M patients for you to test, you could download it
at [omop_synthea.tar.gz](https://drive.google.com/file/d/1k7-cZACaDNw8A1JRI37mfMAhEErxKaQJ/view?usp=share_link)

```bash
tar -xvf omop_synthea.tar data/omop
```

#### Step 1.5. Consolidate Partitioned OMOP Data (if needed)
If your OMOP data is in partitioned format (multiple parquet files per table), consolidate them first:

```bash
python scripts/consolidate_omop_data.py data/omop data/omop_consolidated
```

This script will:
- Combine all partitioned parquet files for each table into single files
- Handle large datasets efficiently using Polars
- Output consolidated files named as `{table_name}.parquet`

### Step 2. Generate Patient Sequences from OMOP Data
Convert your OMOP tables into patient sequences for CEHR-BERT pre-training.

#### Recommended: Arrow Mode (Best Compatibility)
Arrow mode provides the best balance of performance and compatibility:
```bash
bash src/cehrbert/scripts/create_cehrbert_pretraining_data.sh \
   --input_folder data/omop \
   --output_folder data/patient_sequences \
   --start_date 2020-01-01 \
   --min_patients 100 \
   --use_arrow \
   --chunk_size 5000
```

**Why Arrow mode?**
- ✅ Memory efficient - processes data in chunks
- ✅ No string_view compatibility issues
- ✅ Best for HuggingFace integration
- ✅ Works reliably with all dataset sizes

#### Alternative Processing Modes

**Vectorized mode** (fastest, but requires more memory):
```bash
bash src/cehrbert/scripts/create_cehrbert_pretraining_data.sh \
   --input_folder data/omop \
   --output_folder data/patient_sequences \
   --start_date 2020-01-01 \
   --min_patients 100 \
   --use_vectorized \
   --chunk_size 2500
```

**Staging mode** (for distributed processing):
```bash
bash src/cehrbert/scripts/create_cehrbert_pretraining_data.sh \
   data/omop \
   data/patient_sequences \
   2020-01-01 \
   100 \
   --use_staging \
   --num_workers 16 \
   --num_partitions 50
```

#### What the Pipeline Does
1. Filter concepts based on minimum patient count
2. Generate patient sequences with temporal tokens (ATT)
3. Create train/validation/test splits

#### Performance Comparison
| Mode | Speed | Memory Usage | Best For |
|------|-------|--------------|----------|
| Arrow | Fast | Low (~8GB) | **Most users** - reliable and compatible |
| Vectorized | Fastest | High (~20GB) | Small-medium datasets with ample RAM |
| Staging | Moderate | Low | Very large datasets (>10M patients) |

#### Chunk Size Recommendations
| Dataset Size | Arrow Mode | Vectorized Mode | Memory Usage |
|--------------|------------|-----------------|--------------|
| < 100K | 10,000 | 10,000 | ~4GB |
| 100K - 500K | 5,000 | 5,000 | ~8GB |
| 500K - 1M | 5,000 | 2,000 | ~15GB |
| > 1M | 5,000 | 1,000 | ~20GB |

💡 **Tip**: If you encounter out-of-memory errors, reduce chunk_size by half.

#### Manual Processing (Advanced)
```bash
# Step 1: Filter concepts by minimum patient count
python -m cehrbert.data_processing.concept_filter \
   --input_folder data/omop_test/ \
   --output_folder data/patient_sequences/ \
   --min_patients 100

# Step 2: Generate patient sequences
python -m cehrbert.data_processing.sequence_generator \
   --input_folder data/omop_test/ \
   --output_folder data/patient_sequences/ \
   --start_date 1985-01-01 \
   --concept_filter_file data/patient_sequences/included_concepts.parquet

# Step 3: Split into train/val/test (now memory-efficient for large datasets)
python -m cehrbert.data_processing.patient_splitter \
   --input_folder data/patient_sequences/ \
   --train_ratio 0.8
```

### Step 3. Pre-train CEHR-BERT
If you don't have your own OMOP instance, we have provided a sample of patient sequence data generated using Synthea
at `sample_data/pretrain/patient_sequence.parquet` in the repo.

```bash
mkdir test_dataset_prepared;
mkdir test_results;
python -m cehrbert.runners.hf_cehrbert_pretrain_runner \
   sample_configs/hf_cehrbert_pretrain_runner_config.yaml
```

Note: Update the `data_folder` path in the config file to point to your patient sequences folder.

### Step 4. Fine-tune CEHR-BERT
```bash
mkdir test_finetune_results;
python -m cehrbert.runners.hf_cehrbert_finetune_runner \
   sample_configs/hf_cehrbert_finetuning_runner_config.yaml
```

### Step 5. Evaluate CEHR-BERT
After fine-tuning, run comprehensive evaluation on your test set:

```bash
python -m cehrbert.evaluations.evaluate_model \
    --model_path test_finetune_results \
    --test_data sample_data/finetune/test \
    --output_dir evaluation_results
```

This will generate:
- **Evaluation metrics**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Visualizations**: Confusion matrix, ROC curve, class distributions, metrics summary
- **Output files**: `evaluation_metrics.json`, `predictions.npz`, and PNG plots

For additional examples, see the demo notebooks:
- `cehrbert_training_demo.ipynb` - Complete training workflow
- `cehrbert_pipeline_demo.ipynb` - End-to-end pipeline example

## Contact us

If you have any questions, feel free to contact us at CEHR-BERT@lists.cumc.columbia.edu

## Deprecation Notice

The following features have been deprecated in this version:
- **MEDS format support**: CEHR-BERT now focuses exclusively on OMOP CDM v5.4 format
- **TensorFlow/Keras models**: All models are now PyTorch/HuggingFace based
- **PySpark dependency**: Data processing now uses Polars for better performance
- **Dask dependency**: Removed in favor of built-in PyTorch DataLoader and HuggingFace datasets
- **cehrbert_data dependency**: Core functionality has been integrated into the main package

## Citation

Please acknowledge the following work in papers

Chao Pang, Xinzhuo Jiang, Krishna S. Kalluri, Matthew Spotnitz, RuiJun Chen, Adler
Perotte, and Karthik Natarajan. "Cehr-bert: Incorporating temporal information from
structured ehr data to improve prediction tasks." In Proceedings of Machine Learning for
Health, volume 158 of Proceedings of Machine Learning Research, pages 239–260. PMLR,
04 Dec 2021.
