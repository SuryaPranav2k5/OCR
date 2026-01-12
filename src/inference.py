import easyocr
import cv2
import numpy as np
import os
from PIL import Image

# TrOCR imports
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


class EasyOCRRecognizer:
    """Handwriting recognition using EasyOCR"""
    
    def __init__(self):
        print("Initializing EasyOCR...")
        try:
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print("EasyOCR initialized successfully")
        except Exception as e:
            print(f"EasyOCR initialization error: {e}")
            self.reader = easyocr.Reader(['en'], gpu=False)
        self.name = "EasyOCR"
    
    def predict(self, img_path):
        """Predict text from image using EasyOCR"""
        print(f"\n[EasyOCR] Processing: {img_path}")
        
        if not os.path.exists(img_path):
            return "Error: Image file not found"
        
        try:
            # Read and preprocess
            img = cv2.imread(img_path)
            if img is None:
                return "Error: Could not load image"
            
            # Convert to grayscale and enhance
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Run OCR
            results = self.reader.readtext(
                enhanced,
                detail=1,
                paragraph=True,
                text_threshold=0.6,
                low_text=0.3,
            )
            
            # Extract text
            texts = [r[1] for r in results if len(r) >= 2]
            full_text = ' '.join(texts)
            
            print(f"[EasyOCR] Result: '{full_text}'")
            return full_text if full_text.strip() else "(No text detected)"
            
        except Exception as e:
            print(f"[EasyOCR] Error: {e}")
            return f"Error: {str(e)}"


class TrOCRRecognizer:
    """
    Handwriting recognition using Microsoft TrOCR (Transformer-based)
    Includes line segmentation for multi-line documents
    """
    
    def __init__(self):
        print("Initializing TrOCR (this may take a moment on first run)...")
        try:
            # Use TrOCR handwritten model - specifically trained for handwriting
            model_name = "microsoft/trocr-base-handwritten"
            
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            print(f"TrOCR initialized successfully (device: {self.device})")
        except Exception as e:
            print(f"TrOCR initialization error: {e}")
            self.processor = None
            self.model = None
        
        self.name = "TrOCR"
    
    def segment_lines(self, img):
        """
        Segment image into individual text lines using horizontal projection
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate horizontal projection (sum of black pixels per row)
        h_proj = np.sum(binary, axis=1)
        
        # Find line boundaries using projection profile
        threshold = np.max(h_proj) * 0.1  # 10% of max as threshold
        in_line = False
        line_boundaries = []
        start = 0
        
        for i, val in enumerate(h_proj):
            if not in_line and val > threshold:
                in_line = True
                start = i
            elif in_line and val <= threshold:
                in_line = False
                # Add some padding
                pad = 5
                line_boundaries.append((max(0, start - pad), min(len(h_proj), i + pad)))
        
        # Handle case where line extends to end of image
        if in_line:
            line_boundaries.append((start, len(h_proj)))
        
        # Merge very close lines (likely same line with gap)
        merged_boundaries = []
        for start, end in line_boundaries:
            if merged_boundaries and start - merged_boundaries[-1][1] < 10:
                merged_boundaries[-1] = (merged_boundaries[-1][0], end)
            else:
                merged_boundaries.append((start, end))
        
        # Extract line images
        line_images = []
        for start, end in merged_boundaries:
            if end - start > 10:  # Minimum line height
                line_img = img[start:end, :]
                line_images.append(line_img)
        
        print(f"[TrOCR] Segmented into {len(line_images)} lines")
        return line_images
    
    def predict_single_line(self, line_img):
        """Predict text from a single line image"""
        try:
            # Convert numpy array to PIL Image
            if isinstance(line_img, np.ndarray):
                if len(line_img.shape) == 2:  # Grayscale
                    line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
                elif line_img.shape[2] == 4:  # RGBA
                    line_img = cv2.cvtColor(line_img, cv2.COLOR_RGBA2RGB)
                elif line_img.shape[2] == 3:  # BGR
                    line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(line_img)
            else:
                pil_img = line_img
            
            # Process image
            pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text with conservative parameters to reduce hallucination
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=64,              # Shorter max length
                    num_beams=5,                # Beam search for better quality
                    no_repeat_ngram_size=3,     # Prevent repetition
                    length_penalty=1.0,         # Neutral length preference
                    early_stopping=True,        # Stop when all beams finish
                )
            
            # Decode
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
            
        except Exception as e:
            print(f"[TrOCR] Line prediction error: {e}")
            return ""
    
    def predict(self, img_path):
        """Predict text from image using TrOCR with line segmentation"""
        print(f"\n[TrOCR] Processing: {img_path}")
        
        if self.processor is None or self.model is None:
            return "Error: TrOCR not initialized"
        
        if not os.path.exists(img_path):
            return "Error: Image file not found"
        
        try:
            # Load image with OpenCV for line segmentation
            img = cv2.imread(img_path)
            if img is None:
                return "Error: Could not load image"
            
            # Segment into lines
            line_images = self.segment_lines(img)
            
            if not line_images:
                # Fallback: try processing the whole image as one line
                print("[TrOCR] No lines detected, processing as single image")
                pil_img = Image.open(img_path).convert("RGB")
                pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values, max_length=128)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return text if text.strip() else "(No text detected)"
            
            # Process each line
            all_texts = []
            for i, line_img in enumerate(line_images):
                line_text = self.predict_single_line(line_img)
                if line_text:
                    all_texts.append(line_text)
                    print(f"[TrOCR] Line {i+1}: '{line_text}'")
            
            # Join all lines
            full_text = ' '.join(all_texts)
            
            print(f"[TrOCR] Full result: '{full_text}'")
            return full_text if full_text.strip() else "(No text detected)"
            
        except Exception as e:
            print(f"[TrOCR] Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"


class HandwritingRecognizer:
    """
    Combined handwriting recognizer using both EasyOCR and TrOCR
    Returns results from both models for comparison
    """
    
    def __init__(self, model_dir=None):
        print("="*60)
        print("Initializing Dual Handwriting Recognizer")
        print("="*60)
        
        # Initialize both recognizers
        self.easyocr = EasyOCRRecognizer()
        self.trocr = TrOCRRecognizer()
        
        print("="*60)
        print("Dual Recognizer Ready!")
        print("="*60)
    
    def predict(self, img_path):
        """
        Returns prediction from primary recognizer (EasyOCR for backward compatibility)
        Use predict_both() to get results from both models
        """
        return self.easyocr.predict(img_path)
    
    def predict_both(self, img_path):
        """
        Get predictions from both EasyOCR and TrOCR
        
        Returns:
            dict with 'easyocr' and 'trocr' results
        """
        print(f"\n{'='*60}")
        print(f"Processing with BOTH models: {img_path}")
        print('='*60)
        
        easyocr_result = self.easyocr.predict(img_path)
        trocr_result = self.trocr.predict(img_path)
        
        print(f"\n{'='*60}")
        print("COMPARISON:")
        print(f"  EasyOCR: {easyocr_result}")
        print(f"  TrOCR:   {trocr_result}")
        print('='*60)
        
        return {
            'easyocr': easyocr_result,
            'trocr': trocr_result
        }


# Usage example
if __name__ == "__main__":
    recognizer = HandwritingRecognizer()
    
    # Test on a sample image if available
    test_images = [
        'Datasets/IAM/image/c04-110-01.jpg',
        'test_image.png',
    ]
    
    for test_img in test_images:
        if os.path.exists(test_img):
            print(f"\n{'='*60}")
            print(f"Testing on: {test_img}")
            print('='*60)
            results = recognizer.predict_both(test_img)
            print(f"\nFinal Results:")
            print(f"  EasyOCR: {results['easyocr']}")
            print(f"  TrOCR:   {results['trocr']}")
            break
    else:
        print("No test images found.")