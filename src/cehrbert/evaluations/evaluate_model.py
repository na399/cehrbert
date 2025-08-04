"""
Evaluation script for fine-tuned CEHR-BERT models.

This script provides comprehensive evaluation metrics and visualizations
for fine-tuned CEHR-BERT models on test datasets.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc as sklearn_auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader

from cehrbert.models.hf_models.hf_cehrbert import CehrBertForClassification
from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator
from cehrbert.data_generators.hf_data_generator.hf_dataset import create_cehrbert_finetuning_dataset
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments


def load_model_and_tokenizer(model_path: str):
    """Load fine-tuned model and tokenizer."""
    print(f"📂 Loading model from: {model_path}")
    
    model = CehrBertForClassification.from_pretrained(model_path)
    tokenizer = CehrBertTokenizer.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully on {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    return model, tokenizer, device


def prepare_test_data(test_data_path: str, tokenizer, prepared_path: Optional[str] = None):
    """Prepare test dataset for evaluation."""
    print(f"\n📊 Loading test data from: {test_data_path}")
    
    # Check if prepared dataset already exists
    if prepared_path and Path(prepared_path).exists():
        print(f"✅ Loading prepared dataset from: {prepared_path}")
        return load_from_disk(prepared_path)
    
    # Load raw test data
    from datasets import load_dataset
    if os.path.isdir(test_data_path):
        # Load from directory with parquet files
        test_dataset = load_dataset('parquet', data_files=f"{test_data_path}/*.parquet")['train']
    else:
        # Load single file
        test_dataset = load_dataset('parquet', data_files=test_data_path)['train']
    
    print(f"   Loaded {len(test_dataset)} test samples")
    
    # Create data args for processing
    data_args = DataTrainingArguments(
        data_folder=test_data_path,
        dataset_prepared_path=prepared_path or "temp_test_prepared",
        preprocessing_num_workers=4,
        preprocessing_batch_size=1000,
        streaming=False,
        att_function_type='cehrbert',
        include_auxiliary_token=True,
        include_demographic_prompt=False,
        min_num_tokens=1
    )
    
    # Process the dataset
    print("⚙️  Processing test dataset...")
    processed_dataset = create_cehrbert_finetuning_dataset(
        test_dataset,
        tokenizer,
        data_args
    )
    
    # Save prepared dataset if path provided
    if prepared_path:
        processed_dataset.save_to_disk(prepared_path)
        print(f"💾 Saved prepared dataset to: {prepared_path}")
    
    return processed_dataset


def evaluate_model(model, tokenizer, test_dataset, batch_size: int = 32):
    """Run model evaluation and return predictions."""
    print("\n🔬 Running evaluation...")
    
    # Create data collator
    data_collator = CehrBertDataCollator(
        tokenizer=tokenizer,
        max_length=512,
        is_pretraining=False,
        mlm_probability=0.0
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    device = next(model.parameters()).device
    all_predictions = []
    all_labels = []
    all_logits = []
    
    # Run inference
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Extract labels
            labels = batch.pop('classifier_label', None)
            if labels is not None:
                all_labels.extend(labels.cpu().numpy())
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits
            
            # Store logits for later processing
            all_logits.append(logits.cpu().numpy())
    
    # Concatenate all logits
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.array(all_labels)
    
    # Handle both binary and multi-class cases
    if all_logits.shape[1] == 1:
        # Binary classification with single output
        probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
        pred_labels = (probs > 0.5).astype(int).flatten()
        pred_probs = probs.flatten()
    else:
        # Multi-class classification
        # Softmax
        exp_logits = np.exp(all_logits - np.max(all_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        pred_labels = np.argmax(all_logits, axis=1)
        # For binary, get probability of positive class
        if all_logits.shape[1] == 2:
            pred_probs = probs[:, 1]
        else:
            pred_probs = np.max(probs, axis=1)
    
    return pred_labels, pred_probs, all_labels


def calculate_metrics(true_labels, pred_labels, pred_probs) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics."""
    print("\n📈 Calculating metrics...")
    
    # Basic metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Handle binary vs multi-class
    num_classes = len(np.unique(true_labels))
    if num_classes == 2:
        # Binary classification
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary'
        )
        try:
            auc_score = roc_auc_score(true_labels, pred_probs)
        except:
            auc_score = None
    else:
        # Multi-class classification
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, average='macro'
        )
        try:
            auc_score = roc_auc_score(true_labels, pred_probs, multi_class='ovr')
        except:
            auc_score = None
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Per-class metrics
    class_report = classification_report(true_labels, pred_labels, output_dict=True)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc_score) if auc_score is not None else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'num_samples': len(true_labels),
        'num_classes': num_classes
    }
    
    return metrics


def plot_results(true_labels, pred_labels, pred_probs, metrics, output_dir):
    """Generate and save visualization plots."""
    print("\n📊 Generating visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(len(cm)), 
                yticklabels=range(len(cm)))
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve (for binary classification)
    if metrics['num_classes'] == 2 and metrics['auc'] is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {metrics["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Class distribution
    plt.figure(figsize=(10, 6))
    unique_labels, counts = np.unique(true_labels, return_counts=True)
    x_pos = np.arange(len(unique_labels))
    
    # True vs Predicted distribution
    pred_unique, pred_counts = np.unique(pred_labels, return_counts=True)
    
    width = 0.35
    plt.bar(x_pos - width/2, counts, width, label='True', alpha=0.8)
    plt.bar(x_pos + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution: True vs Predicted', fontsize=16)
    plt.xticks(x_pos, unique_labels)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Metrics summary plot
    plt.figure(figsize=(8, 6))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1']
    ]
    
    if metrics['auc'] is not None:
        metric_names.append('AUC-ROC')
        metric_values.append(metrics['auc'])
    
    y_pos = np.arange(len(metric_names))
    plt.barh(y_pos, metric_values, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for i, v in enumerate(metric_values):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=11)
    
    plt.ylabel('Metrics', fontsize=12)
    plt.xlabel('Score', fontsize=12)
    plt.title('Model Performance Metrics', fontsize=16)
    plt.yticks(y_pos, metric_names)
    plt.xlim(0, 1.1)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to: {output_dir}")


def print_results(metrics):
    """Print evaluation results to console."""
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nDataset size: {metrics['num_samples']:,} samples")
    print(f"Number of classes: {metrics['num_classes']}")
    
    print("\n🎯 Overall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1']:.3f}")
    if metrics['auc'] is not None:
        print(f"  AUC-ROC:   {metrics['auc']:.3f}")
    
    print("\n📋 Confusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(cm)
    
    print("\n📊 Per-Class Performance:")
    class_report = metrics['classification_report']
    for class_label in sorted([k for k in class_report.keys() if k.isdigit()]):
        class_metrics = class_report[class_label]
        print(f"\nClass {class_label}:")
        print(f"  Precision: {class_metrics['precision']:.3f}")
        print(f"  Recall:    {class_metrics['recall']:.3f}")
        print(f"  F1-Score:  {class_metrics['f1-score']:.3f}")
        print(f"  Support:   {class_metrics['support']}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned CEHR-BERT model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model directory')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data (parquet file or directory)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--prepared_data_path', type=str, default=None,
                        help='Path to save/load prepared test dataset')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating visualization plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 CEHR-BERT Model Evaluation")
    print("="*60)
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model_path)
    
    # Prepare test data
    test_dataset = prepare_test_data(args.test_data, tokenizer, args.prepared_data_path)
    
    # Run evaluation
    pred_labels, pred_probs, true_labels = evaluate_model(
        model, tokenizer, test_dataset, args.batch_size
    )
    
    # Calculate metrics
    metrics = calculate_metrics(true_labels, pred_labels, pred_probs)
    
    # Save metrics to JSON
    metrics_file = output_dir / 'evaluation_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Metrics saved to: {metrics_file}")
    
    # Generate plots
    if not args.no_plots:
        plot_results(true_labels, pred_labels, pred_probs, metrics, output_dir)
    
    # Print results
    print_results(metrics)
    
    # Save predictions
    predictions_file = output_dir / 'predictions.npz'
    np.savez(predictions_file,
             true_labels=true_labels,
             pred_labels=pred_labels,
             pred_probs=pred_probs)
    print(f"💾 Predictions saved to: {predictions_file}")
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()