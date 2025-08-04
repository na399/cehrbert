#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --input_folder INPUT_FOLDER --output_folder OUTPUT_FOLDER --start_date START_DATE"
    echo ""
    echo "Required Arguments:"
    echo "  --input_folder PATH      Input folder path containing OMOP parquet files"
    echo "  --output_folder PATH     Output folder path for processed data"
    echo "  --start_date DATE        Start date for filtering data (YYYY-MM-DD)"
    echo ""
    echo "Optional Arguments:"
    echo "  --min_patients NUM       Minimum number of patients for concept inclusion (default: 100)"
    echo "  --domain_tables TABLES   Comma-separated list of domain tables (default: condition_occurrence,procedure_occurrence,drug_exposure)"
    echo "  --sample_size NUM        Optional number of patients to sample for testing"
    echo "  --batch_size NUM         Batch size for processing (default: 5000)"
    echo "  --use_staging            Use staging mode for very large datasets"
    echo "  --num_workers NUM        Number of parallel workers for staging mode"
    echo "  --num_partitions NUM     Number of partitions for staging (default: 20)"
    echo "  --use_vectorized         Use fully vectorized implementation (fastest)"
    echo "  --chunk_size NUM         Chunk size for vectorized mode (default: 5000)"
    echo "  --use_arrow              Use Arrow format (avoids string_view issues, memory efficient)"
    echo ""
    echo "Example:"
    echo "  $0 --input_folder /path/to/omop --output_folder /path/to/output --start_date 1985-01-01"
    exit 1
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    usage
fi

# Initialize variables
INPUT_FOLDER=""
OUTPUT_FOLDER=""
START_DATE=""
MIN_PATIENTS=100
DOMAIN_TABLES="condition_occurrence,procedure_occurrence,drug_exposure"
SAMPLE_SIZE=""
BATCH_SIZE=5000
USE_STAGING=false
NUM_WORKERS=""
NUM_PARTITIONS=20
USE_VECTORIZED=false
CHUNK_SIZE=5000
USE_ARROW=false

# Parse command line arguments
ARGS=$(getopt -o "" --long input_folder:,output_folder:,start_date:,min_patients:,domain_tables:,sample_size:,batch_size:,use_staging,num_workers:,num_partitions:,use_vectorized,chunk_size:,use_arrow,help -n "$0" -- "$@")

if [ $? -ne 0 ]; then
    usage
fi

eval set -- "$ARGS"

while true; do
    case "$1" in
        --input_folder)
            INPUT_FOLDER="$2"
            shift 2
            ;;
        --output_folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        --start_date)
            START_DATE="$2"
            shift 2
            ;;
        --min_patients)
            MIN_PATIENTS="$2"
            shift 2
            ;;
        --domain_tables)
            DOMAIN_TABLES="$2"
            shift 2
            ;;
        --sample_size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --use_staging)
            USE_STAGING=true
            shift
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --num_partitions)
            NUM_PARTITIONS="$2"
            shift 2
            ;;
        --use_vectorized)
            USE_VECTORIZED=true
            shift
            ;;
        --chunk_size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --use_arrow)
            USE_ARROW=true
            shift
            ;;
        --help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_FOLDER" ] || [ -z "$OUTPUT_FOLDER" ] || [ -z "$START_DATE" ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Set Polars to use CPU cores - 1 for parallel processing
# Get number of CPU cores
if command -v nproc &> /dev/null; then
    # Linux
    CPU_CORES=$(nproc)
elif command -v sysctl &> /dev/null; then
    # macOS
    CPU_CORES=$(sysctl -n hw.ncpu)
else
    # Default to 4 if we can't detect
    CPU_CORES=4
fi

# Set to CPU cores - 1, with minimum of 1
NUM_WORKERS=$((CPU_CORES - 1))
if [ $NUM_WORKERS -lt 1 ]; then
    NUM_WORKERS=1
fi

export POLARS_MAX_THREADS=$NUM_WORKERS

echo "CEHR-BERT Data Preprocessing"
echo "============================"
echo "Input folder: $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Start date: $START_DATE"
echo "Min patients: $MIN_PATIENTS"
echo "Domain tables: $DOMAIN_TABLES"
if [ -n "$SAMPLE_SIZE" ]; then
    echo "Sample size: $SAMPLE_SIZE"
fi
echo "Batch size: $BATCH_SIZE"
if [ "$USE_ARROW" = true ]; then
    echo "Using Arrow mode (memory efficient, no string_view issues)"
    echo "Chunk size: $CHUNK_SIZE"
elif [ "$USE_VECTORIZED" = true ]; then
    echo "Using vectorized mode (fastest)"
    echo "Chunk size: $CHUNK_SIZE"
elif [ "$USE_STAGING" = true ]; then
    echo "Using staging mode"
    if [ -n "$NUM_WORKERS" ]; then
        echo "Parallel workers: $NUM_WORKERS"
    else
        echo "Parallel workers: auto"
    fi
    echo "Number of partitions: $NUM_PARTITIONS"
fi
echo "CPU cores detected: $CPU_CORES"
echo "Polars threads: $POLARS_MAX_THREADS"
echo ""

# Convert comma-separated list to array
IFS=',' read -ra DOMAIN_TABLE_ARRAY <<< "$DOMAIN_TABLES"

# Step 1: Generate concept statistics and filter concepts
echo "Step 1: Generating concept statistics..."
python -u -m cehrbert.data_processing.concept_filter \
    --input_folder "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_FOLDER" \
    --min_patients $MIN_PATIENTS \
    --domain_tables "${DOMAIN_TABLE_ARRAY[@]}"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate concept statistics"
    exit 1
fi

# Step 2: Generate patient sequences for training
echo "Step 2: Generating patient sequences..."

if [ "$USE_ARROW" = true ]; then
    # Use Arrow implementation
    SEQ_GEN_CMD="python -m cehrbert.data_processing.sequence_generator_arrow \
        --input_folder '$INPUT_FOLDER' \
        --output_folder '$OUTPUT_FOLDER' \
        --start_date '$START_DATE' \
        --concept_filter_file '$OUTPUT_FOLDER/included_concepts.parquet' \
        --chunk_size $CHUNK_SIZE \
        --domain_tables ${DOMAIN_TABLE_ARRAY[@]}"
elif [ "$USE_VECTORIZED" = true ]; then
    # Use vectorized implementation
    SEQ_GEN_CMD="python -m cehrbert.data_processing.sequence_generator_vectorized \
        --input_folder '$INPUT_FOLDER' \
        --output_folder '$OUTPUT_FOLDER' \
        --start_date '$START_DATE' \
        --concept_filter_file '$OUTPUT_FOLDER/included_concepts.parquet' \
        --chunk_size $CHUNK_SIZE \
        --domain_tables ${DOMAIN_TABLE_ARRAY[@]}"
else
    # Use original or staging implementation
    SEQ_GEN_CMD="python -m cehrbert.data_processing.sequence_generator \
        --input_folder '$INPUT_FOLDER' \
        --output_folder '$OUTPUT_FOLDER' \
        --start_date '$START_DATE' \
        --concept_filter_file '$OUTPUT_FOLDER/included_concepts.parquet' \
        --att_type 'day' \
        --batch_size $BATCH_SIZE \
        --domain_tables ${DOMAIN_TABLE_ARRAY[@]}"
    
    if [ "$USE_STAGING" = true ]; then
        SEQ_GEN_CMD="$SEQ_GEN_CMD --use-staging --num-partitions $NUM_PARTITIONS"
        if [ -n "$NUM_WORKERS" ]; then
            SEQ_GEN_CMD="$SEQ_GEN_CMD --num-workers $NUM_WORKERS"
        fi
    fi
fi

if [ -n "$SAMPLE_SIZE" ]; then
    SEQ_GEN_CMD="$SEQ_GEN_CMD --sample_size $SAMPLE_SIZE"
fi

eval $SEQ_GEN_CMD

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate patient sequences"
    exit 1
fi

# Step 3: Create train/test splits
echo "Step 3: Creating train/test splits..."
if [ "$USE_ARROW" = true ]; then
    # Use Arrow-based splitter
    python -m cehrbert.data_processing.patient_splitter_arrow \
        --input_folder "$OUTPUT_FOLDER" \
        --output_folder "$OUTPUT_FOLDER" \
        --train_ratio 0.8 \
        --seed 42
else
    # Use regular splitter
    python -m cehrbert.data_processing.patient_splitter \
        --input_folder "$OUTPUT_FOLDER" \
        --train_ratio 0.8 \
        --seed 42
fi

if [ $? -ne 0 ]; then
    echo "Error: Failed to create patient splits"
    exit 1
fi

echo ""
echo "Data preprocessing completed successfully!"
echo "Output files are in: $OUTPUT_FOLDER"