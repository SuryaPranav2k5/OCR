"""
TrOCR Model Evaluation Script
=============================

This script evaluates the OCR models and generates metrics to verify CV claims:
- Character Error Rate (CER) / Word Error Rate (WER)
- Accuracy comparison: EasyOCR vs TrOCR vs Hybrid
- Processing time benchmarks
- Preprocessing noise reduction analysis

Run: python src/evaluate_model.py
"""

import os
import sys
import time
import json
import glob
import cv2
import numpy as np
from datetime import datetime
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from jiwer import cer, wer
    HAS_JIWER = True
except ImportError:
    print("Warning: jiwer not installed. Installing...")
    os.system("pip install jiwer")
    from jiwer import cer, wer
    HAS_JIWER = True

from src.inference import EasyOCRRecognizer, TrOCRRecognizer, HandwritingRecognizer


class OCRBenchmark:
    """Benchmark OCR models and generate metrics for CV claims"""
    
    def __init__(self, output_dir="evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("=" * 60)
        print("OCR MODEL EVALUATION BENCHMARK")
        print("=" * 60)
        
        # Initialize recognizers
        print("\n📦 Loading models...")
        self.easyocr = EasyOCRRecognizer()
        self.trocr = TrOCRRecognizer()
        self.hybrid = HandwritingRecognizer()
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "easyocr": {"predictions": [], "times": [], "cer": [], "wer": []},
            "trocr": {"predictions": [], "times": [], "cer": [], "wer": []},
            "hybrid": {"predictions": [], "times": [], "cer": [], "wer": []},
        }
    
    def find_test_samples(self, test_dir=None):
        """Find test images with ground truth labels"""
        samples = []
        
        # Check common locations
        possible_dirs = [
            test_dir,
            "data/test",
            "Datasets/IAM/test",
            "test_images",
            "data",
        ]
        
        for dir_path in possible_dirs:
            if dir_path and os.path.exists(dir_path):
                # Look for images
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    for img_path in glob.glob(os.path.join(dir_path, "**", ext), recursive=True):
                        # Check for corresponding text file
                        txt_path = img_path.rsplit('.', 1)[0] + '.txt'
                        if os.path.exists(txt_path):
                            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                                ground_truth = f.read().strip()
                            samples.append((img_path, ground_truth))
        
        return samples
    
    def evaluate_single(self, img_path, ground_truth, model_name, model):
        """Evaluate a single image with one model"""
        # Time the prediction
        start_time = time.time()
        
        if model_name == "hybrid":
            result = model.predict_both(img_path)
            prediction = result.get('trocr', '')  # Use TrOCR output from hybrid
        else:
            prediction = model.predict(img_path)
        
        elapsed_time = time.time() - start_time
        
        # Calculate error rates
        if ground_truth and prediction and prediction != "(No text detected)":
            char_error = cer(ground_truth.lower(), prediction.lower())
            word_error = wer(ground_truth.lower(), prediction.lower())
        else:
            char_error = 1.0
            word_error = 1.0
        
        return {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "time": elapsed_time,
            "cer": char_error,
            "wer": word_error,
            "accuracy": 1.0 - char_error,
        }
    
    def run_benchmark(self, samples=None, max_samples=50):
        """Run full benchmark on test samples"""
        if samples is None:
            samples = self.find_test_samples()
        
        if not samples:
            print("\n⚠️ No test samples found with ground truth labels!")
            print("Creating synthetic benchmark with sample images...")
            return self.run_synthetic_benchmark()
        
        samples = samples[:max_samples]
        print(f"\n📊 Evaluating {len(samples)} samples...")
        
        for i, (img_path, ground_truth) in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] {os.path.basename(img_path)}")
            print(f"   Ground Truth: '{ground_truth[:50]}...'")
            
            # EasyOCR
            result = self.evaluate_single(img_path, ground_truth, "easyocr", self.easyocr)
            self.results["easyocr"]["predictions"].append(result)
            self.results["easyocr"]["times"].append(result["time"])
            self.results["easyocr"]["cer"].append(result["cer"])
            self.results["easyocr"]["wer"].append(result["wer"])
            print(f"   EasyOCR: CER={result['cer']:.2%}, Time={result['time']:.2f}s")
            
            # TrOCR
            result = self.evaluate_single(img_path, ground_truth, "trocr", self.trocr)
            self.results["trocr"]["predictions"].append(result)
            self.results["trocr"]["times"].append(result["time"])
            self.results["trocr"]["cer"].append(result["cer"])
            self.results["trocr"]["wer"].append(result["wer"])
            print(f"   TrOCR:   CER={result['cer']:.2%}, Time={result['time']:.2f}s")
        
        return self.generate_report()
    
    def run_synthetic_benchmark(self):
        """Run benchmark without ground truth (timing only)"""
        # Find any available images
        test_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            test_images.extend(glob.glob(f"**/{ext}", recursive=True))
        
        test_images = [img for img in test_images if 'debug' not in img.lower()][:10]
        
        if not test_images:
            print("❌ No images found for testing!")
            return None
        
        print(f"\n⏱️ Running timing benchmark on {len(test_images)} images...")
        
        for img_path in test_images:
            print(f"\n   Processing: {img_path}")
            
            # EasyOCR timing
            start = time.time()
            self.easyocr.predict(img_path)
            easyocr_time = time.time() - start
            self.results["easyocr"]["times"].append(easyocr_time)
            
            # TrOCR timing
            start = time.time()
            self.trocr.predict(img_path)
            trocr_time = time.time() - start
            self.results["trocr"]["times"].append(trocr_time)
            
            # Hybrid timing
            start = time.time()
            self.hybrid.predict_both(img_path)
            hybrid_time = time.time() - start
            self.results["hybrid"]["times"].append(hybrid_time)
            
            print(f"   EasyOCR: {easyocr_time:.2f}s | TrOCR: {trocr_time:.2f}s | Hybrid: {hybrid_time:.2f}s")
        
        return self.generate_report()
    
    def measure_preprocessing_quality(self, img_path):
        """Measure noise reduction from preprocessing"""
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Original noise estimate (using Laplacian variance)
        original_noise = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # After preprocessing
        preprocessed = self.trocr.preprocess_image(img)
        preprocessed_gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
        processed_noise = cv2.Laplacian(preprocessed_gray, cv2.CV_64F).var()
        
        # Calculate noise reduction
        if original_noise > 0:
            noise_reduction = ((original_noise - processed_noise) / original_noise) * 100
        else:
            noise_reduction = 0
        
        return {
            "original_noise": original_noise,
            "processed_noise": processed_noise,
            "noise_reduction_percent": noise_reduction,
        }
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        report = {
            "timestamp": self.results["timestamp"],
            "summary": {},
            "cv_claims_verification": {},
        }
        
        # Calculate averages
        for model in ["easyocr", "trocr", "hybrid"]:
            data = self.results[model]
            
            avg_time = np.mean(data["times"]) if data["times"] else 0
            avg_cer = np.mean(data["cer"]) if data["cer"] else None
            avg_wer = np.mean(data["wer"]) if data["wer"] else None
            
            report["summary"][model] = {
                "num_samples": len(data["times"]),
                "avg_processing_time_sec": round(avg_time, 3),
                "avg_cer": round(avg_cer, 4) if avg_cer else None,
                "avg_wer": round(avg_wer, 4) if avg_wer else None,
                "character_accuracy": round((1 - avg_cer) * 100, 2) if avg_cer else None,
            }
        
        # Verify CV claims
        easyocr_acc = report["summary"]["easyocr"].get("character_accuracy")
        trocr_acc = report["summary"]["trocr"].get("character_accuracy")
        hybrid_time = report["summary"]["hybrid"].get("avg_processing_time_sec", 0)
        
        if easyocr_acc and trocr_acc:
            improvement = trocr_acc - easyocr_acc
            report["cv_claims_verification"]["accuracy_improvement"] = {
                "easyocr_accuracy": easyocr_acc,
                "trocr_accuracy": trocr_acc,
                "improvement_percent": round(improvement, 2),
                "claim_35_percent": improvement >= 35,
            }
        
        report["cv_claims_verification"]["processing_time"] = {
            "hybrid_avg_time_sec": hybrid_time,
            "claim_under_1_2_sec": hybrid_time < 1.2 if hybrid_time else "Not tested",
        }
        
        if trocr_acc:
            report["cv_claims_verification"]["character_accuracy"] = {
                "trocr_accuracy": trocr_acc,
                "claim_92_percent": trocr_acc >= 92,
            }
        
        # Save report
        report_path = os.path.join(self.output_dir, "evaluation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 EVALUATION REPORT")
        print("=" * 60)
        
        print("\n📈 Model Performance Summary:")
        for model, stats in report["summary"].items():
            print(f"\n   {model.upper()}:")
            print(f"      Samples tested: {stats['num_samples']}")
            print(f"      Avg processing time: {stats['avg_processing_time_sec']:.3f}s")
            if stats['character_accuracy']:
                print(f"      Character accuracy: {stats['character_accuracy']:.2f}%")
        
        print("\n" + "-" * 60)
        print("✅ CV CLAIMS VERIFICATION:")
        print("-" * 60)
        
        claims = report["cv_claims_verification"]
        
        if "accuracy_improvement" in claims:
            acc = claims["accuracy_improvement"]
            status = "✅ VERIFIED" if acc["claim_35_percent"] else "⚠️ NOT MET"
            print(f"\n   '35% accuracy improvement': {status}")
            print(f"      EasyOCR: {acc['easyocr_accuracy']:.2f}%")
            print(f"      TrOCR:   {acc['trocr_accuracy']:.2f}%")
            print(f"      Actual improvement: {acc['improvement_percent']:.2f}%")
        
        if "processing_time" in claims:
            pt = claims["processing_time"]
            if isinstance(pt["claim_under_1_2_sec"], bool):
                status = "✅ VERIFIED" if pt["claim_under_1_2_sec"] else "⚠️ NOT MET"
                print(f"\n   'Processing under 1.2 seconds': {status}")
                print(f"      Actual avg time: {pt['hybrid_avg_time_sec']:.3f}s")
        
        if "character_accuracy" in claims:
            ca = claims["character_accuracy"]
            status = "✅ VERIFIED" if ca["claim_92_percent"] else "⚠️ NOT MET"
            print(f"\n   '92% character accuracy': {status}")
            print(f"      Actual accuracy: {ca['trocr_accuracy']:.2f}%")
        
        print("\n" + "=" * 60)
        print(f"📁 Full report saved to: {report_path}")
        print("=" * 60)
        
        return report


def create_test_samples():
    """Create test samples with ground truth for evaluation"""
    test_dir = "data/test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Check if we already have samples
    existing = glob.glob(os.path.join(test_dir, "*.txt"))
    if existing:
        print(f"Found {len(existing)} existing test samples")
        return
    
    print("\n📝 To properly evaluate the models, you need test images with ground truth.")
    print("   Please add images to 'data/test/' folder with corresponding .txt files.")
    print("   Example:")
    print("      data/test/sample1.png  (handwriting image)")
    print("      data/test/sample1.txt  (contains the text in the image)")
    print("\n   You can use IAM dataset images for accurate evaluation.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 HANDWRITING OCR EVALUATION SUITE")
    print("=" * 60)
    
    # Create test directory structure
    create_test_samples()
    
    # Run benchmark
    benchmark = OCRBenchmark()
    report = benchmark.run_benchmark()
    
    if report:
        print("\n✅ Evaluation complete!")
    else:
        print("\n⚠️ Evaluation could not be completed. Please add test samples.")
