import cv2
import pickle
import numpy as np
import os
from tensorflow import keras
import tensorflow as tf

class HandwritingRecognizer:
    """Load and use trained model for inference"""
    
    def __init__(self, model_dir):
        # Load configuration
        with open(f'{model_dir}/config.pkl', 'rb') as f:
            self.config = pickle.load(f)
        
        # Load model - try .keras first, then fallback to .h5
        model_path = f'{model_dir}/final_model.keras'
        if not os.path.exists(model_path):
            model_path = f'{model_dir}/final_model.h5'
        
        # Load model with safe_mode=False to handle Lambda layers
        self.model = keras.models.load_model(
            model_path,
            safe_mode=False,
            compile=False
        )
        
        self.vocab = self.config['vocab']
        self.num_to_char = self.config['num_to_char']
        self.char_to_num = self.config['char_to_num']  # Also load char_to_num for verification
        self.img_width = self.config['img_width']
        self.img_height = self.config['img_height']
        
        # Debug: Print some info about the loaded model and config
        print(f"Model input shape: {self.model.input_shape}")
        print(f"Model output shape: {self.model.output_shape}")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Num to char mapping size: {len(self.num_to_char)}")
        # Print first few mappings for verification
        print(f"First 10 num_to_char mappings: {dict(list(self.num_to_char.items())[:10])}")
        # Print some specific mappings that might be relevant
        if 58 in self.num_to_char:
            print(f"Index 58 maps to: '{self.num_to_char[58]}'")
    
    def preprocess_image(self, img_path):
        """Preprocess image for prediction"""
        # Debug: Print image path
        print(f"Preprocessing image: {img_path}")
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Debug: Check if image was loaded successfully
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            return None
            
        print(f"Original image shape: {img.shape}")
        
        img = cv2.resize(img, (self.img_width, self.img_height))
        print(f"Resized image shape: {img.shape}")
        
        img = np.expand_dims(img, axis=-1)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        print(f"Final preprocessed image shape: {img.shape}")
        
        return img
    
    def decode_prediction(self, pred):
        """Decode model output using CTC decoder"""
        print(f"Prediction shape: {pred.shape}")
        
        # Analyze prediction probabilities
        batch_size, time_steps, num_classes = pred.shape
        print(f"Analyzing prediction probabilities:")
        
        # Show probabilities for first few time steps
        for t in range(min(5, time_steps)):  # Show 5 time steps instead of 3
            time_step_probs = pred[0, t, :]
            top_5_indices = np.argsort(time_step_probs)[-5:][::-1]  # Top 5 indices
            print(f"  Time step {t}: Top 5 classes - ", end="")
            for idx in top_5_indices:
                prob = time_step_probs[idx]
                char = self.num_to_char.get(idx, f"UNK({idx})")
                print(f"[{idx}:'{char}'={prob:.6f}] ", end="")
            print()
        
        # Try different CTC decoding approaches
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        
        # Method 1: Greedy decoding (original)
        try:
            results = keras.backend.ctc_decode(
                pred, 
                input_length=input_len, 
                greedy=True
            )[0][0]
        except Exception as e:
            print(f"Greedy decoding failed: {e}")
            # Fallback: Manual greedy decoding
            return self._manual_greedy_decode(pred)
        
        # Convert to text
        output_text = []
        for i, res in enumerate(results):
            res = res.numpy()
            print(f"Decoded indices for result {i}: {res}")
            
            # Convert indices to characters, handling the mapping correctly
            text_chars = []
            prev_idx = None
            for idx in res:
                int_idx = int(idx)
                # Skip blank tokens (0) and duplicates for CTC
                if int_idx != 0 and int_idx != prev_idx:
                    if int_idx in self.num_to_char:
                        char = self.num_to_char[int_idx]
                        print(f"  Index {int_idx} -> Char '{char}'")
                        # Only add non-empty characters (not just whitespace)
                        if char and char.strip():  # Check if character is not just whitespace
                            text_chars.append(char)
                    else:
                        print(f"  Index {int_idx} not in num_to_char mapping")
                    prev_idx = int_idx
                elif int_idx == 0:
                    print(f"  Skipping blank token (index 0)")
                    prev_idx = None  # Reset previous index for blank
                else:
                    print(f"  Skipping duplicate index {int_idx}")
            
            text = ''.join(text_chars)
            print(f"  Final text for result {i}: '{text}'")
            output_text.append(text)
        
        result = output_text[0] if output_text else ""
        print(f"Final decoded text: '{result}'")
        return result
    
    def _manual_greedy_decode(self, pred):
        """Manual greedy decoding as fallback"""
        print("Using manual greedy decoding")
        batch_size, time_steps, num_classes = pred.shape
        decoded_texts = []
        
        for b in range(batch_size):
            # Get the class with maximum probability for each time step
            max_indices = np.argmax(pred[b], axis=1)
            print(f"Manual decoding - Max indices: {max_indices}")
            
            # Show what the max probabilities are
            max_probs = np.max(pred[b], axis=1)
            print(f"Max probabilities: {max_probs[:10]}...")  # First 10 values
            
            # Apply CTC collapsing rules
            text_chars = []
            prev_idx = None
            for i, idx in enumerate(max_indices):
                prob = pred[b, i, idx]
                print(f"  Time {i}: idx={idx}, prob={prob:.6f}")
                # Skip blank tokens (0) and duplicates
                if idx != 0 and idx != prev_idx:
                    if idx in self.num_to_char:
                        char = self.num_to_char[idx]
                        # Only add non-empty characters (not just whitespace)
                        if char and char.strip():
                            text_chars.append(char)
                            print(f"    Added char '{char}'")
                    prev_idx = idx
                elif idx == 0:
                    print(f"    Skipping blank token")
                    prev_idx = None  # Reset for blank
                else:
                    print(f"    Skipping duplicate")
            
            text = ''.join(text_chars)
            decoded_texts.append(text)
            print(f"Manual decoded text: '{text}'")
        
        return decoded_texts[0] if decoded_texts else ""
    
    def predict(self, img_path):
        """Predict text from image"""
        print(f"\n--- Processing image: {img_path} ---")
        
        img = self.preprocess_image(img_path)
        if img is None:
            return "Error: Could not load image"
            
        pred = self.model.predict(img, verbose=0)
        text = self.decode_prediction(pred)
        return text

# Usage
if __name__ == "__main__":
    recognizer = HandwritingRecognizer('Models/trained_model')
    
    # Test on an image
    result = recognizer.predict('test_image.png')
    print(f"Predicted text: {result}")