import cv2
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    """Generate batches of data for training"""
    
    def __init__(self, data, vocab, max_len, batch_size=32, 
                 img_width=256, img_height=64, shuffle=True):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.shuffle = shuffle
        
        # Create character to index mapping (start from 1, reserve 0 for CTC blank)
        self.char_to_num = {char: idx + 1 for idx, char in enumerate(vocab)}
        self.num_to_char = {idx + 1: char for idx, char in enumerate(vocab)}
        self.num_to_char[0] = ''  # Blank character for CTC
        
        self.indexes = np.arange(len(self.data))
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.floor(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y
    
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def preprocess_image(self, img_path):
        """Load and preprocess image"""
        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to fixed size
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # Expand dimensions for model input
        img = np.expand_dims(img, axis=-1)
        
        return img
    
    def encode_label(self, label):
        """Encode text label to numbers"""
        encoded = [self.char_to_num.get(char, 0) for char in label]
        
        # Pad to max length (use 0 for padding, which is the CTC blank token)
        encoded = encoded + [0] * (self.max_len - len(encoded))
        return encoded[:self.max_len]
    
    def __data_generation(self, indexes):
        """Generate data for batch"""
        # Initialize arrays
        X = np.zeros((self.batch_size, self.img_height, self.img_width, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, self.max_len), dtype=np.int32)
        
        # Generate data
        for i, idx in enumerate(indexes):
            img_path, label = self.data[idx]
            
            # Process image
            X[i] = self.preprocess_image(img_path)
            
            # Encode label
            y[i] = self.encode_label(label)
        
        return X, y