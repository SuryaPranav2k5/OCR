"""
Fine-tune TrOCR on IAM Handwriting Dataset
============================================

This script fine-tunes the Microsoft TrOCR model on the IAM handwriting dataset
to improve recognition accuracy for cursive handwriting.

Requirements:
    pip install transformers datasets torch torchvision accelerate evaluate jiwer

Usage:
    python src/finetune_trocr.py

The fine-tuned model will be saved to: Models/trocr-finetuned/
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from datasets import load_dataset
from PIL import Image
import evaluate
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

# Configuration
MODEL_NAME = "microsoft/trocr-base-handwritten"  # Start with base for faster training
OUTPUT_DIR = "Models/trocr-finetuned"
BATCH_SIZE = 4  # Reduce if you run out of memory
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_LENGTH = 64

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


class IAMDataset(Dataset):
    """Custom dataset for IAM handwriting data"""
    
    def __init__(self, dataset, processor, max_length=64):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get image and text
        image = item['image']
        text = item['text']
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Process text (labels)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


@dataclass
class DataCollatorForTrOCR:
    """Custom data collator for TrOCR"""
    processor: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


def compute_metrics(pred):
    """Compute Character Error Rate (CER) metric"""
    cer_metric = evaluate.load("cer")
    
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    # Replace -100 with pad token id
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    
    # Compute CER
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"cer": cer}


def main():
    global processor  # Make processor available for compute_metrics
    
    print("=" * 60)
    print("TrOCR Fine-tuning on IAM Dataset")
    print("=" * 60)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    if device == "cpu":
        print("⚠️  WARNING: Training on CPU will be very slow!")
        print("   Consider using Google Colab or a GPU machine.")
    
    # Load processor and model
    print(f"\nLoading model: {MODEL_NAME}")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    
    # Set decoder start token
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Load IAM dataset from Hugging Face
    print("\nLoading IAM dataset from Hugging Face...")
    print("(This may take a few minutes on first run)")
    
    try:
        # Load the IAM handwriting dataset
        dataset = load_dataset("nielsr/iam_handwriting", trust_remote_code=True)
        print(f"Dataset loaded successfully!")
        print(f"  Train samples: {len(dataset['train'])}")
        print(f"  Test samples: {len(dataset['test'])}")
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nTrying alternative dataset...")
        
        # Try alternative IAM dataset
        try:
            dataset = load_dataset("Teklia/IAM-line", trust_remote_code=True)
            print(f"Alternative dataset loaded!")
            print(f"  Train samples: {len(dataset['train'])}")
            print(f"  Test samples: {len(dataset['test'])}")
        except Exception as e2:
            print(f"Error: {e2}")
            print("\nPlease ensure you have internet access and try again.")
            return
    
    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = IAMDataset(dataset['train'], processor, MAX_LENGTH)
    eval_dataset = IAMDataset(dataset['test'], processor, MAX_LENGTH)
    
    # Use a subset for faster training (optional, remove for full training)
    # Uncomment these lines for quick testing:
    # train_dataset.dataset = train_dataset.dataset.select(range(min(1000, len(train_dataset))))
    # eval_dataset.dataset = eval_dataset.dataset.select(range(min(200, len(eval_dataset))))
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Evaluation samples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForTrOCR(processor=processor)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision on GPU
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # Save the final model
    print("\nSaving fine-tuned model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {OUTPUT_DIR}")
    print("\nTo use the fine-tuned model, update inference.py to load from:")
    print(f'  model_name = "{OUTPUT_DIR}"')
    print("=" * 60)


if __name__ == "__main__":
    main()
