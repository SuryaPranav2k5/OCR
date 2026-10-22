"""
TrOCR Fine-tuning on IAM Word Database - Kaggle Version
========================================================

Uses the Kaggle dataset: nibinv23/iam-handwriting-word-database
This dataset has BOTH images AND labels (words.txt)

Instructions:
1. Create a new Kaggle notebook
2. Add dataset: "nibinv23/iam-handwriting-word-database" 
3. Enable GPU: Settings > Accelerator > GPU T4 x2
4. Copy this code and run!

Training time: ~2-3 hours on Kaggle GPU
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow/CUDA warnings

import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Dict, List, Any
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

# Kaggle dataset path - UPDATE THIS to match your dataset
DATASET_PATH = "/kaggle/input/iam-handwriting-word-database"

# Model configuration
MODEL_NAME = "microsoft/trocr-base-handwritten"
OUTPUT_DIR = "/kaggle/working/trocr-finetuned"

# Training hyperparameters
BATCH_SIZE = 8          # Reduce to 4 if OOM error
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_LENGTH = 32         # Words are shorter than lines
MAX_SAMPLES = None      # Set to 10000 for quick test

# ============================================================
# EXPLORE DATASET STRUCTURE
# ============================================================

print("=" * 60)
print("TrOCR FINE-TUNING ON IAM WORD DATABASE")
print("=" * 60)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🖥️  Device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

print(f"\n📂 Exploring dataset: {DATASET_PATH}")

# List contents
if os.path.exists(DATASET_PATH):
    print("\nDataset contents:")
    for item in os.listdir(DATASET_PATH)[:20]:
        item_path = os.path.join(DATASET_PATH, item)
        if os.path.isdir(item_path):
            count = len(os.listdir(item_path))
            print(f"  📁 {item}/ ({count} items)")
        else:
            size = os.path.getsize(item_path) / 1024
            print(f"  📄 {item} ({size:.1f} KB)")
else:
    print(f"❌ Path not found: {DATASET_PATH}")
    print("\nAvailable paths in /kaggle/input/:")
    for item in os.listdir("/kaggle/input/"):
        print(f"  📁 {item}/")
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

# ============================================================
# PARSE DATASET
# ============================================================

def find_and_parse_labels(base_path):
    """
    Find and parse label file (words.txt or lines.txt)
    IAM format: id ok/err graylevel x y w h transcription
    """
    samples = []
    
    # Define common label file locations
    label_files = [
        os.path.join(base_path, "words.txt"),
        os.path.join(base_path, "ascii", "words.txt"),
        os.path.join(base_path, "iam_words", "words.txt"),
        os.path.join(base_path, "lines.txt"),
        os.path.join(base_path, "ascii", "lines.txt"),
    ]
    
    label_file = None
    
    # 1. Check specific locations first (FAST)
    print("   Searching known locations...")
    for lf in label_files:
        if os.path.exists(lf):
            label_file = lf
            break
            
    # 2. If not found, search recursively (SLOW)
    if label_file is None:
        print("   Checking known paths failed. Starting recursive search (this may take a while)...")
        found_txt = glob.glob(os.path.join(base_path, "**", "words.txt"), recursive=True)
        found_txt += glob.glob(os.path.join(base_path, "**", "lines.txt"), recursive=True)
        
        for lf in found_txt:
            if os.path.exists(lf):
                label_file = lf
                break
    
    if label_file is None:
        print("❌ No words.txt or lines.txt found!")
        print("\nSearching for any .txt files...")
        all_txt = glob.glob(os.path.join(base_path, "**", "*.txt"), recursive=True)[:10]
        for txt in all_txt:
            print(f"  📄 {txt}")
        return []
    
    print(f"\n✅ Found label file: {label_file}")

    # Find image directory
    image_dirs = [
        os.path.join(base_path, "words"),
        os.path.join(base_path, "iam_words", "words"),
        os.path.join(base_path, "lines"),
        os.path.join(base_path, "images"),
        base_path,
    ]
    
    image_base = None
    
    # 1. Check specific locations first (FAST)
    print("   Searching keys locations for images...")
    for idir in image_dirs:
        if os.path.exists(idir) and os.path.isdir(idir):
            # Check if it has images
            # Only check first 50 files to be fast
            has_images = False
            has_subdirs = False
            try:
                files = os.listdir(idir)[:50] 
                has_images = any(f.lower().endswith('.png') or f.lower().endswith('.jpg') for f in files)
                has_subdirs = any(os.path.isdir(os.path.join(idir, f)) for f in files)
            except:
                pass
                
            # Accept if it has images OR if it has subdirectories (IAM structure has images in subfolders)
            if has_images or has_subdirs:
                image_base = idir
                break
                
    # 2. If not found, search recursively (SLOW)
    if image_base is None:
        print("   Checking known image paths failed. Starting recursive search...")
        # Add any directory that contains images
        for dir_path in glob.glob(os.path.join(base_path, "**/"), recursive=True):
            if any(f.lower().endswith('.png') for f in os.listdir(dir_path)):
                image_dirs.append(dir_path)
    
    image_base = None
    for idir in image_dirs:
        if os.path.exists(idir) and os.path.isdir(idir):
            # Check if it has images or subdirectories with images
            has_images = any(f.endswith('.png') for f in os.listdir(idir) if os.path.isfile(os.path.join(idir, f)))
            has_subdirs = any(os.path.isdir(os.path.join(idir, d)) for d in os.listdir(idir))
            if has_images or has_subdirs:
                image_base = idir
                break
    
    print(f"📁 Image base directory: {image_base}")
    
    # Parse label file
    print(f"\n📖 Parsing labels...")
    with open(label_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    print(f"   Total lines in file: {len(lines)}")
    
    # Show sample lines
    print("\n   Sample lines from label file:")
    for line in lines[:5]:
        if not line.startswith('#'):
            print(f"   {line.strip()[:80]}")
    
    # Parse each line
    for line in tqdm(lines, desc="Parsing labels"):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) < 9:
            continue
        
        # Format: id ok/err graylevel x y w h text(s)
        img_id = parts[0]
        status = parts[1]
        text = parts[-1].replace('|', ' ')  # Last part is the text
        
        if status == 'err':
            continue
        
        # Build image path
        # a01-000u-00-00 -> a01/a01-000u/a01-000u-00-00.png
        id_parts = img_id.split('-')
        
        possible_paths = []
        
        if len(id_parts) >= 2:
            folder1 = id_parts[0]
            folder2 = f"{id_parts[0]}-{id_parts[1]}"
            
            if image_base:
                possible_paths = [
                    os.path.join(image_base, folder1, folder2, f"{img_id}.png"),
                    os.path.join(image_base, folder1, f"{img_id}.png"),
                    os.path.join(image_base, folder2, f"{img_id}.png"),
                    os.path.join(image_base, f"{img_id}.png"),
                ]
        
        # Also try direct path
        possible_paths.append(os.path.join(base_path, f"{img_id}.png"))
        
        for img_path in possible_paths:
            if os.path.exists(img_path):
                samples.append((img_path, text))
                break
    
    return samples

# Parse the dataset
samples = find_and_parse_labels(DATASET_PATH)

if len(samples) == 0:
    print("\n❌ No samples found with standard parsing!")
    print("\n🔍 Trying alternative: find all images with text files...")
    
    # Alternative: look for image + text pairs
    for img_path in glob.glob(os.path.join(DATASET_PATH, "**", "*.png"), recursive=True)[:10000]:
        # Check for .txt file with same name
        txt_path = img_path.replace('.png', '.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                text = f.read().strip()
            if text:
                samples.append((img_path, text))
    
    # Alternative: check for CSV/TSV files
    if len(samples) == 0:
        for csv_file in glob.glob(os.path.join(DATASET_PATH, "**", "*.csv"), recursive=True):
            print(f"\nFound CSV: {csv_file}")
            import csv
            with open(csv_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                print(f"   Headers: {header}")
                for row in list(reader)[:5]:
                    print(f"   Sample: {row}")

print(f"\n📊 Total samples found: {len(samples)}")

if len(samples) == 0:
    raise ValueError("No samples found! Please check dataset structure.")

# Show sample
print("\n   Sample entries:")
for img_path, text in samples[:3]:
    print(f"   📷 {os.path.basename(img_path)}: '{text}'")

# ============================================================
# DATASET AND TRAINING
# ============================================================

# Limit samples if needed
if MAX_SAMPLES and len(samples) > MAX_SAMPLES:
    samples = samples[:MAX_SAMPLES]
    print(f"\n⚡ Using {MAX_SAMPLES} samples for faster training")

# Split data
train_samples, val_samples = train_test_split(samples, test_size=0.1, random_state=42)
print(f"\n   Training: {len(train_samples)}")
print(f"   Validation: {len(val_samples)}")


class IAMDataset(Dataset):
    def __init__(self, samples, processor, max_length=32):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            # Resize tiny images to avoid "ambiguous channel dimension" warnings
            if image.size[0] < 16 or image.size[1] < 16:
                image = image.resize((max(32, image.size[0]), max(32, image.size[1])), Image.Resampling.BICUBIC)
            
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
            labels = self.processor.tokenizer(
                text, padding="max_length", max_length=self.max_length,
                truncation=True, return_tensors="pt"
            ).input_ids.squeeze()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels}
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.samples))


@dataclass
class DataCollator:
    processor: Any
    def __call__(self, features):
        return {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features])
        }


# Load model
print(f"\n🤖 Loading model: {MODEL_NAME}")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.to(device)

# Create datasets
train_dataset = IAMDataset(train_samples, processor, MAX_LENGTH)
val_dataset = IAMDataset(val_samples, processor, MAX_LENGTH)

# Training
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=500,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollator(processor=processor),
    tokenizer=processor.tokenizer,
)

print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)

trainer.train()

# Save
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

import shutil
shutil.make_archive("/kaggle/working/trocr-finetuned-model", 'zip', OUTPUT_DIR)

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("Download: /kaggle/working/trocr-finetuned-model.zip")
