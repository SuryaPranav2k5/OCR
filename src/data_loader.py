import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_iam_dataset(dataset_root):
    """
    Load IAM dataset with gt_test.txt format
    
    Args:
        dataset_root: Path to IAM folder (contains image/, gt_test.txt)
    
    Returns:
        dataset: List of [image_path, label] pairs
        vocab: Set of unique characters
        max_len: Maximum label length
    """
    dataset = []
    vocab = set()
    max_len = 0
    
    # Path to annotation file
    gt_file = os.path.join(dataset_root, 'gt_test.txt')
    image_dir = os.path.join(dataset_root, 'image')
    
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    print(f"Loading dataset from {gt_file}")
    
    # Read annotation file
    with open(gt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse each line
    for line in tqdm(lines, desc="Loading dataset"):
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Split by tab
        parts = line.split('\t')
        
        if len(parts) < 2:
            # Try space as separator if tab doesn't work
            parts = line.split(' ', 1)
        
        if len(parts) >= 2:
            image_name = parts[0].strip()
            label = parts[1].strip()
            
            # Construct full image path
            image_path = os.path.join(image_dir, image_name)
            
            # Check if image exists
            if os.path.exists(image_path):
                dataset.append([image_path, label])
                vocab.update(list(label))
                max_len = max(max_len, len(label))
            else:
                # Try with different extensions
                for ext in ['.png', '.jpg', '.jpeg']:
                    test_path = os.path.join(image_dir, image_name + ext)
                    if os.path.exists(test_path):
                        dataset.append([test_path, label])
                        vocab.update(list(label))
                        max_len = max(max_len, len(label))
                        break
    
    return dataset, sorted(vocab), max_len

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        dataset: List of [image_path, label] pairs
        train_ratio: Proportion for training (default 0.8)
        val_ratio: Proportion for validation (default 0.1)
        random_state: Random seed for reproducibility
    
    Returns:
        train_data, val_data, test_data
    """
    # First split: train vs (val + test)
    train_data, temp_data = train_test_split(
        dataset, 
        test_size=(1 - train_ratio), 
        random_state=random_state
    )
    
    # Second split: val vs test
    val_size = val_ratio / (1 - train_ratio)
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=(1 - val_size), 
        random_state=random_state
    )
    
    return train_data, val_data, test_data

def load_vocabulary_file(vocab_file_path):
    """
    Load vocabulary from gpt2.dict.txt if needed
    
    Args:
        vocab_file_path: Path to gpt2.dict.txt
    
    Returns:
        vocab: List of characters
    """
    if os.path.exists(vocab_file_path):
        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f.readlines()]
        return vocab
    return None

# Test the loader
if __name__ == "__main__":
    # Update this path to your dataset location
    dataset_root = "Datasets/IAM"
    
    # Load dataset
    print("="*60)
    print("Loading IAM Dataset")
    print("="*60)
    
    dataset, vocab, max_len = load_iam_dataset(dataset_root)
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Max label length: {max_len}")
    print(f"  Vocabulary: {''.join(vocab)[:100]}...")
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(dataset)
    
    print(f"\nDataset Split:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    # Show sample entries
    print(f"\nSample Entries:")
    for i in range(min(3, len(dataset))):
        img_path, label = dataset[i]
        print(f"  {i+1}. Image: {os.path.basename(img_path)}")
        print(f"     Label: '{label}'")
