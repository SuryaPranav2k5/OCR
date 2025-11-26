import cv2
import os
import pickle
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from inference import HandwritingRecognizer

class AssignmentGrader:
    """Grade handwritten assignments"""
    
    def __init__(self, model_dir):
        self.recognizer = HandwritingRecognizer(model_dir)
    
    def preprocess_answer_sheet(self, img_path):
        """Preprocess scanned answer sheet"""
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def extract_text_from_region(self, img_path, bbox=None):
        """Extract text from specific region or whole image"""
        if bbox:
            img = cv2.imread(img_path)
            x1, y1, x2, y2 = bbox
            cropped = img[y1:y2, x1:x2]
            temp_path = 'temp_crop.jpg'
            cv2.imwrite(temp_path, cropped)
            text = self.recognizer.predict(temp_path)
            os.remove(temp_path)
        else:
            text = self.recognizer.predict(img_path)
        
        return text
    
    def calculate_similarity(self, student_answer, reference_answer):
        """Calculate similarity between answers"""
        # Method 1: Sequence Matching
        seq_score = SequenceMatcher(None, student_answer.lower(), 
                                   reference_answer.lower()).ratio()
        
        # Method 2: TF-IDF Cosine Similarity
        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([reference_answer, student_answer])
            tfidf_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except:
            tfidf_score = 0.0
        
        # Combined score
        final_score = (0.6 * seq_score + 0.4 * tfidf_score)
        
        return {
            'sequence_match': seq_score,
            'tfidf_similarity': tfidf_score,
            'final_score': final_score
        }
    
    def grade_answer(self, student_answer, reference_answer, max_score=10):
        """Grade a single answer"""
        similarity = self.calculate_similarity(student_answer, reference_answer)
        grade = round(similarity['final_score'] * max_score, 2)
        
        return {
            'student_answer': student_answer,
            'reference_answer': reference_answer,
            'grade': grade,
            'max_score': max_score,
            'percentage': round(similarity['final_score'] * 100, 2)
        }
    
    def grade_assignment(self, img_path, reference_answers, question_regions=None):
        """
        Grade complete assignment
        
        Args:
            img_path: Path to scanned assignment
            reference_answers: Dict of {question_num: reference_answer}
            question_regions: Dict of {question_num: (x1,y1,x2,y2)} bounding boxes
        """
        results = []
        
        # Preprocess image
        processed = self.preprocess_answer_sheet(img_path)
        temp_processed = 'temp_processed.jpg'
        cv2.imwrite(temp_processed, processed)
        
        for q_num, ref_answer in reference_answers.items():
            # Extract text for this question
            if question_regions and q_num in question_regions:
                bbox = question_regions[q_num]
                student_text = self.extract_text_from_region(temp_processed, bbox)
            else:
                student_text = self.extract_text_from_region(temp_processed)
            
            # Grade answer
            result = self.grade_answer(student_text, ref_answer)
            result['question'] = q_num
            results.append(result)
        
        # Cleanup
        if os.path.exists(temp_processed):
            os.remove(temp_processed)
        
        # Calculate total score
        total_grade = sum([r['grade'] for r in results])
        max_total = sum([r['max_score'] for r in results])
        
        return {
            'results': results,
            'total_grade': total_grade,
            'max_total': max_total,
            'percentage': round((total_grade / max_total) * 100, 2)
        }

# Usage Example
if __name__ == "__main__":
    grader = AssignmentGrader('Models/trained_model')
    
    # Define reference answers
    reference_answers = {
        'Q1': 'photosynthesis is the process by which plants make food',
        'Q2': 'machine learning is a branch of artificial intelligence',
        'Q3': 'python is a high level programming language'
    }
    
    # Grade assignment
    results = grader.grade_assignment(
        img_path='student_assignment.jpg',
        reference_answers=reference_answers
    )
    
    # Display results
    print(f"\n{'='*60}")
    print(f"ASSIGNMENT GRADING RESULTS")
    print(f"{'='*60}\n")
    
    for result in results['results']:
        print(f"Question: {result['question']}")
        print(f"Student Answer: {result['student_answer']}")
        print(f"Grade: {result['grade']}/{result['max_score']} ({result['percentage']}%)")
        print(f"{'-'*60}\n")
    
    print(f"{'='*60}")
    print(f"TOTAL: {results['total_grade']}/{results['max_total']} ({results['percentage']}%)")
    print(f"{'='*60}\n")
    
    # Save to CSV
    df = pd.DataFrame(results['results'])
    df.to_csv('grading_results.csv', index=False)
    print("Results saved to grading_results.csv")