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
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Denoise significantly to remove paper texture
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Run OCR with optimized parameters for handwriting
            results = self.reader.readtext(
                enhanced,
                detail=1,
                paragraph=True,
                text_threshold=0.5,   # Lower threshold to catch faint text
                low_text=0.3,
                link_threshold=0.2,   # Lower link threshold to merge handwritten words
                mag_ratio=1.5,        # Magnify image to detect strokes better
                beamWidth=5           # Use beam search for better decoding
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
            # Check if fine-tuned model exists, otherwise use pre-trained
            # Try multiple possible locations for the fine-tuned model
            possible_paths = [
                "Models/trocr-finetuned",
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models", "trocr-finetuned"),
                "trocr-finetuned-model",
            ]
            
            finetuned_path = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
                    finetuned_path = path
                    break
            
            # IMPORTANT: Fine-tuned model was trained on WORD images only (IAM word database)
            # It does NOT work well for full sentences/paragraphs - outputs subword tokens
            # Use pre-trained model for general handwriting recognition
            use_finetuned = False  # Set to True only for single-word recognition
            
            if use_finetuned and finetuned_path:
                model_name = finetuned_path
                print(f"🔧 Loading FINE-TUNED model from: {model_name}")
                print("   ⚠️ Note: Fine-tuned on WORD images - best for single words only")
            else:
                # Pre-trained models work better for general handwriting
                # base = faster, large = more accurate (but slower)
                model_name = "microsoft/trocr-base-handwritten"
                print(f"🤖 Loading pre-trained model: {model_name}")
                print("   (Better for full sentences and paragraphs)")
            
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
        Advanced segmentation:
        1. Create a mask of rules lines
        2. Subtract it from binary image (keeping text)
        3. Find text blobs
        """
        # Work on a copy
        clean_img = img.copy()
        gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
        
        # Binary thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Create a mask for ruled lines
        lines_mask = np.zeros_like(binary)
        
        # Edge detection for finding lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=20)
        
        if lines is not None:
            print(f"[TrOCR] Detected {len(lines)} ruled lines")
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check angle
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 5:
                    # Draw on lines_mask (white lines on black bg)
                    cv2.line(lines_mask, (x1, y1), (x2, y2), 255, 3)
        
        # Subtract lines mask from binary image
        # Pixels present in ONLY binary (text) and NOT in lines_mask (lines) will remain
        # This keeps vertical strokes that cross the line, while removing the horizontal line line
        text_only = cv2.subtract(binary, lines_mask)
        
        # Dilate to connect text horizontally
        # Use (30,1) to prioritize horizontal merging
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        dilated = cv2.dilate(text_only, kernel, iterations=5)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lines_bboxes = []
        img_h, img_w = img.shape[:2]
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter noise
            if h > 15 and w > 30: 
                lines_bboxes.append((x, y, w, h))
        
        # Sort top-to-bottom
        lines_bboxes.sort(key=lambda b: b[1])
        
        # Extract
        line_images = []
        debug_dir = "debug_lines"
        os.makedirs(debug_dir, exist_ok=True)
        for f in os.listdir(debug_dir):
            if f.endswith('.png'): os.remove(os.path.join(debug_dir, f))
            
        for i, (x, y, w, h) in enumerate(lines_bboxes):
            # Expand box slightly to catch ascenders/descenders
            pad_y = 10  # Increased from 5
            pad_x = 20  # Increased from 10
            y1 = max(0, y - pad_y)
            y2 = min(img_h, y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(img_w, x + w + pad_x)
            
            line_img = img[y1:y2, x1:x2]
            line_images.append(line_img)
            cv2.imwrite(os.path.join(debug_dir, f"line_{i}.png"), line_img)
            
        print(f"[TrOCR] Segmented into {len(line_images)} lines (Hough method) (saved to {debug_dir}/)")
        return line_images
    
    def predict_single_line(self, line_img):
        """Predict text from a single line image"""
        try:
            # Apply selective preprocessing (line removal + contrast boost)
            if isinstance(line_img, np.ndarray):
                line_img = self.preprocess_image(line_img)
            
            # Convert numpy array to PIL Image
            if isinstance(line_img, np.ndarray):
                # Ensure RGB format
                if len(line_img.shape) == 2:
                    line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
                elif line_img.shape[2] == 4:
                    line_img = cv2.cvtColor(line_img, cv2.COLOR_RGBA2RGB)
                elif line_img.shape[2] == 3:
                    # OpenCV is BGR, PIL needs RGB
                    line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
                
                pil_img = Image.fromarray(line_img)
            else:
                pil_img = line_img
            
            # Don't resize too aggressively - TrOCR expects 384x384 but handles various sizes
            # Only resize if image is too small or too large
            min_height = 32
            max_height = 128
            
            if pil_img.height < min_height:
                aspect_ratio = pil_img.width / pil_img.height
                new_height = min_height
                new_width = int(new_height * aspect_ratio)
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            elif pil_img.height > max_height:
                aspect_ratio = pil_img.width / pil_img.height
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Process image through the processor (handles resizing to model input size)
            pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text with proper parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_new_tokens=128,       # Max tokens to generate
                    num_beams=5,              # Beam search for better results
                    no_repeat_ngram_size=0,   # Allow repetitions (common in OCR)
                    length_penalty=1.0,
                    early_stopping=True,
                )
            
            # Decode the generated tokens to text
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up the result
            text = text.strip()
            
            # Filter out garbage (too many special characters)
            if text:
                # Check if result is mostly garbage (subword tokens)
                if text.count('#') > len(text) / 4:  # More than 25% are # symbols
                    print(f"[TrOCR] Warning: Detected subword tokens, result may be unreliable")
                    return ""
            
            return text
            
        except Exception as e:
            print(f"[TrOCR] Line prediction error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def preprocess_image(self, img):
        """
        TrOCR-friendly preprocessing:
        - Remove horizontal ruled lines (helps with notebook paper)
        - Light contrast enhancement (helps with faint pencil strokes)
        - NO binarization (TrOCR needs grayscale richness, not black & white)
        - NO heavy blur (preserves fine stroke details)
        """
        # Convert to grayscale for line detection
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # --- Step 1: Remove horizontal ruled lines ---
        # Morphological opening with a wide horizontal kernel
        # Only structures wider than 40px survive (ruled lines, not letters)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Create a binary mask of the detected lines
        _, line_mask = cv2.threshold(detected_lines, 200, 255, cv2.THRESH_BINARY)
        
        # Replace ruled line pixels with white (background)
        gray_cleaned = gray.copy()
        gray_cleaned[line_mask < 255] = 255
        
        # --- Step 2: Light contrast enhancement ---
        # Use CLAHE with conservative clipLimit to boost faint strokes
        # without amplifying noise or destroying natural gradients
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_cleaned)
        
        # --- NO blur (preserves thin pen/pencil strokes) ---
        # --- NO binarization (TrOCR was trained on natural grayscale images) ---
        
        # Convert back to 3-channel BGR for consistency with rest of pipeline
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        print("[TrOCR] Preprocessed: ruled lines removed, contrast enhanced (grayscale preserved)")
        return result
    
    def predict(self, img_path):
        """
        Hybrid prediction:
        1. Use EasyOCR to DETECT text lines (bounding boxes)
        2. Use TrOCR to RECOGNIZE text in those boxes
        """
        print(f"\n[TrOCR-Hybrid] Processing: {img_path}")
        
        if self.processor is None or self.model is None:
            return "Error: TrOCR not initialized"
        
        try:
            # Load full image
            img = cv2.imread(img_path)
            if img is None: return "Error loading image"
            
            # Use EasyOCR to get bounding boxes ONLY
            print("[TrOCR-Hybrid] identifying text lines with EasyOCR...")
            import easyocr
            # using cpu to avoid locking gpu memory if needed, or stick to auto
            reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            
            # Efficient detection
            # Use paragraph=False to get words, then we sort them manually into lines
            boxes = reader.readtext(
                img, 
                text_threshold=0.5, 
                low_text=0.3, 
                link_threshold=0.2,
                mag_ratio=1.5,
                canvas_size=2048,
                paragraph=False
            )
            
            # Custom logic to group words into lines
            # 1. Sort all boxes by Y coordinate
            boxes.sort(key=lambda r: r[0][0][1])
            
            lines_of_boxes = []
            current_line = []
            last_y = -100
            
            for box in boxes:
                y = box[0][0][1]
                h = box[0][2][1] - y
                
                # If new box is significantly lower, start new line
                if y > last_y + (h * 0.5):
                    if current_line:
                        # Sort previous line by X
                        current_line.sort(key=lambda r: r[0][0][0])
                        lines_of_boxes.append(current_line)
                    current_line = [box]
                    last_y = y
                else:
                    current_line.append(box)
                    
            if current_line:
                current_line.sort(key=lambda r: r[0][0][0])
                lines_of_boxes.append(current_line)
            
            # Merge boxes in each line to form the line crop
            line_images = []
            img_h, img_w = img.shape[:2]
            
            debug_dir = "debug_lines"
            os.makedirs(debug_dir, exist_ok=True)
            for f in os.listdir(debug_dir): os.remove(os.path.join(debug_dir, f))
            
            for i, line_boxes in enumerate(lines_of_boxes):
                # Find bounding box of the whole line
                all_xs = []
                all_ys = []
                for box in line_boxes:
                    poly = box[0]
                    all_xs.extend([p[0] for p in poly])
                    all_ys.extend([p[1] for p in poly])
                
                x1 = max(0, int(min(all_xs)) - 20)
                x2 = min(img_w, int(max(all_xs)) + 20)
                y1 = max(0, int(min(all_ys)) - 5)
                y2 = min(img_h, int(max(all_ys)) + 5)
                
                # Filter noise
                if (y2 - y1) < 15 or (x2 - x1) < 20: continue
                
                crop = img[y1:y2, x1:x2]
                line_images.append(crop)
                cv2.imwrite(os.path.join(debug_dir, f"line_{i}.png"), crop)
            
            print(f"[TrOCR-Hybrid] Detected {len(line_images)} lines")
            
            if not line_images:
                return "No text detected"
            
            # Feed crops to TrOCR
            all_texts = []
            
            for i, line_img in enumerate(line_images):
                line_text = self.predict_single_line(line_img)
                
                if not line_text or len(line_text) < 2: continue
                # Filter mostly-symbol garbage
                alnum_rate = sum(c.isalnum() for c in line_text) / len(line_text)
                if len(line_text) > 5 and alnum_rate < 0.3: continue
                
                print(f"[TrOCR] Line {i+1}: '{line_text}'")
                all_texts.append(line_text)
                
            full_text = ' '.join(all_texts)
            return full_text if full_text.strip() else "(No text detected)"
            
        except Exception as e:
            print(f"Hybrid Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {e}"


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