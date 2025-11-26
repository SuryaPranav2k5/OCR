import os
import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_loader import load_iam_dataset, split_dataset
from data_generator import DataGenerator
from model import build_model, CTCLoss

# Configuration
DATASET_PATH = "Datasets/IAM"  # Your IAM folder
MODEL_DIR = "Models/trained_model"
IMG_WIDTH = 256
IMG_HEIGHT = 64
BATCH_SIZE = 16  # Reduced batch size for better training stability
EPOCHS = 100  # Increased epochs for better convergence

# Create model directory
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
print("="*60)
print("Loading IAM Dataset")
print("="*60)

dataset, vocab, max_len = load_iam_dataset(DATASET_PATH)

print(f"\nDataset Statistics:")
print(f"  Total samples: {len(dataset)}")
print(f"  Vocabulary size: {len(vocab)}")
print(f"  Max label length: {max_len}")

# Split dataset into train/val/test
train_data, val_data, test_data = split_dataset(
    dataset, 
    train_ratio=0.8, 
    val_ratio=0.1, 
    random_state=42
)

print(f"\nDataset Split:")
print(f"  Training: {len(train_data)}")
print(f"  Validation: {len(val_data)}")
print(f"  Test: {len(test_data)}")

# Optional: Use smaller subset for quick testing
# Uncomment these lines for faster training during testing
# print("\n⚠️  Using small subset for quick testing...")
# train_data = train_data[:500]
# val_data = val_data[:100]

# Create data generators
print("\nCreating data generators...")
train_generator = DataGenerator(
    train_data, vocab, max_len, 
    batch_size=BATCH_SIZE,
    img_width=IMG_WIDTH, 
    img_height=IMG_HEIGHT,
    shuffle=True
)

val_generator = DataGenerator(
    val_data, vocab, max_len, 
    batch_size=BATCH_SIZE,
    img_width=IMG_WIDTH, 
    img_height=IMG_HEIGHT,
    shuffle=False
)

# Build model
print("\nBuilding model...")
model = build_model(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),
    num_classes=len(vocab)
)

# Compile model with a lower learning rate for better convergence
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Reduced learning rate
    loss=CTCLoss
)

model.summary()

# Define callbacks with more aggressive learning rate reduction
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,  # Increased patience
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'best_model.keras'),  # Changed to .keras format
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,  # Reduced patience for more frequent adjustments
        min_lr=1e-7,  # Lower minimum learning rate
        verbose=1
    )
]

# Train model
print("\n" + "="*60)
print("Starting Training")
print("="*60)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save final model
print("\nSaving model...")
model.save(os.path.join(MODEL_DIR, 'final_model.keras'))  # Changed to .keras format

# Save configuration
config = {
    'vocab': vocab,
    'char_to_num': train_generator.char_to_num,
    'num_to_char': train_generator.num_to_char,
    'max_len': max_len,
    'img_width': IMG_WIDTH,
    'img_height': IMG_HEIGHT,
    'dataset_size': len(dataset),
    'train_size': len(train_data),
    'val_size': len(val_data),
    'test_size': len(test_data)
}

with open(os.path.join(MODEL_DIR, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)

print(f"\n{'='*60}")
print(f"Training Complete!")
print(f"{'='*60}")
print(f"Model saved to: {MODEL_DIR}")
print(f"  - final_model.keras")
print(f"  - config.pkl")
print(f"  - best_model.keras")