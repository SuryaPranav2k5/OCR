"""
Processing Time Benchmark
=========================

Quick benchmark to verify the "processing in under 1.2 seconds" CV claim.
Tests the Flask grading app's full pipeline timing.

Run: python src/benchmark_timing.py
"""

import os
import sys
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.inference import HandwritingRecognizer


def benchmark_processing_time(num_iterations=5):
    """Benchmark the full OCR + grading pipeline"""
    
    print("=" * 60)
    print("⏱️  PROCESSING TIME BENCHMARK")
    print("=" * 60)
    
    # Find test images
    test_images = []
    for root, dirs, files in os.walk("."):
        if "debug" in root.lower() or "__pycache__" in root:
            continue
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(root, f))
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    test_images = test_images[:3]  # Test on 3 images
    
    print(f"\n📷 Testing on {len(test_images)} images, {num_iterations} iterations each")
    print("-" * 60)
    
    # Initialize model (this takes time, but only happens once)
    print("\n🔧 Initializing recognizer...")
    init_start = time.time()
    recognizer = HandwritingRecognizer()
    init_time = time.time() - init_start
    print(f"   Model initialization: {init_time:.2f}s (one-time cost)")
    
    all_times = []
    
    for img_path in test_images:
        print(f"\n📷 {os.path.basename(img_path)}")
        
        times = []
        for i in range(num_iterations):
            start = time.time()
            result = recognizer.predict_both(img_path)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"   Iteration {i+1}: {elapsed:.3f}s")
        
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"   📊 Avg: {avg_time:.3f}s | Min: {min_time:.3f}s | Max: {max_time:.3f}s")
        all_times.extend(times)
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("📊 OVERALL RESULTS")
    print("=" * 60)
    
    overall_avg = statistics.mean(all_times)
    overall_min = min(all_times)
    overall_max = max(all_times)
    overall_std = statistics.stdev(all_times) if len(all_times) > 1 else 0
    
    print(f"\n   Total measurements: {len(all_times)}")
    print(f"   Average time:       {overall_avg:.3f}s")
    print(f"   Minimum time:       {overall_min:.3f}s")
    print(f"   Maximum time:       {overall_max:.3f}s")
    print(f"   Std deviation:      {overall_std:.3f}s")
    
    print("\n" + "-" * 60)
    print("✅ CV CLAIM VERIFICATION: 'Processing under 1.2 seconds'")
    print("-" * 60)
    
    if overall_avg < 1.2:
        print(f"   ✅ VERIFIED: Average processing time is {overall_avg:.3f}s (< 1.2s)")
    else:
        print(f"   ⚠️ NOT MET: Average processing time is {overall_avg:.3f}s (> 1.2s)")
        print(f"   Note: This may vary based on:")
        print(f"      - GPU availability (current: {'CUDA' if 'cuda' in str(recognizer.trocr.device) else 'CPU'})")
        print(f"      - Image size and complexity")
        print(f"      - First-run model loading overhead")
    
    print("=" * 60)
    
    return {
        "avg_time": overall_avg,
        "min_time": overall_min,
        "max_time": overall_max,
        "meets_claim": overall_avg < 1.2,
    }


if __name__ == "__main__":
    benchmark_processing_time()
