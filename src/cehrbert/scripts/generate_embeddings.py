import argparse
import pandas as pd
import torch
from datasets import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import logging, set_seed

# Import CEHRBERT specific classes from the project.
# Ensure the 'cehrbert' source code is in your Python path.
from cehrbert.data_generators.hf_data_generator.hf_dataset import create_cehrbert_pretraining_dataset
from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator
from cehrbert.models.hf_models.hf_cehrbert import CehrBertForPreTraining
from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments


def generate_embeddings(args):
    """
    Loads a pre-trained CEHRBERT model and generates patient embeddings from a parquet file.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # --- Configuration ---
    set_seed(42)
    logging.set_verbosity_info()
    BATCH_SIZE = 8  # Adjust based on your available memory

    # --- Determine Output File Path ---
    if args.output_file is None:
        input_path = Path(args.input_file)
        # Inserts .emb before the final .parquet suffix
        output_path = input_path.with_name(f"{input_path.stem}.emb.parquet")
    else:
        output_path = Path(args.output_file)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)


    # --- 1. Load Model and Tokenizer ---
    print(f"Loading pre-trained CEHRBERT model and tokenizer from {args.model_name_or_path}...")
    tokenizer = CehrBertTokenizer.from_pretrained(args.model_name_or_path)
    model = CehrBertForPreTraining.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # --- 2. Load and Preprocess Data ---
    print(f"Loading data from {args.input_file}...")
    patient_sequence_df = pd.read_parquet(args.input_file)
    dataset = Dataset.from_pandas(patient_sequence_df)

    print("Preprocessing and tokenizing the dataset...")
    data_args = DataTrainingArguments(
        data_folder=None,
        dataset_prepared_path=None,
        is_data_in_meds=False,
        streaming=False
    )

    processed_dataset = create_cehrbert_pretraining_dataset(
        dataset=dataset,
        concept_tokenizer=tokenizer,
        data_args=data_args
    )
    processed_dataset.set_format(type="torch")

    # --- 3. Create DataLoader ---
    print("Setting up DataLoader...")
    data_collator = CehrBertDataCollator(
        tokenizer=tokenizer,
        max_length=model.config.max_position_embeddings,
        is_pretraining=False
    )
    dataloader = DataLoader(
        processed_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator
    )

    # --- 4. Generate Embeddings ---
    print("Generating patient embeddings...")
    all_embeddings = []
    all_person_ids = []

    with torch.no_grad():
        # Wrap dataloader with tqdm for a progress bar
        for batch in tqdm(dataloader, desc="Generating Embeddings"):
            all_person_ids.extend(batch.pop('person_id').tolist())
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model.bert(**inputs)
            patient_embeddings = outputs.pooler_output
            all_embeddings.append(patient_embeddings.cpu())

    final_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    final_embeddings_list = final_embeddings_tensor.numpy().tolist()

    # --- 5. Save Output to Parquet ---
    print(f"Saving embeddings to {output_path}...")
    output_df = pd.DataFrame({
        'person_id': all_person_ids,
        'embedding': final_embeddings_list
    })
    output_df.to_parquet(output_path, index=False)

    print("\n✅ Processing complete.")
    print(f"Generated and saved {len(output_df)} embeddings to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate patient embeddings using a pre-trained CEHRBERT model.")
    parser.add_argument(
        "-m", "--model_name_or_path",
        type=str,
        required=True,
        help="Path or Hugging Face Hub name of the pre-trained CEHRBERT model."
    )
    parser.add_argument(
        "-i", "--input_file",
        type=str,
        required=True,
        help="Path to the input patient_sequence.parquet file."
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        default=None,
        help="Path to save the output embeddings. Defaults to input file name with .emb.parquet suffix."
    )

    args = parser.parse_args()
    generate_embeddings(args)