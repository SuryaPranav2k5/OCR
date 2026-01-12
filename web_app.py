import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import our handwriting recognizer
from src.inference import HandwritingRecognizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Initialize the recognizer
print("Initializing HandwritingRecognizer...")
try:
    recognizer = HandwritingRecognizer()
    print("HandwritingRecognizer initialized successfully")
except Exception as e:
    print(f"Error initializing HandwritingRecognizer: {e}")
    recognizer = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_similarity(student_answer, reference_answer):
    """Calculate similarity between student answer and reference answer"""
    # Normalize texts
    student_lower = student_answer.lower().strip()
    reference_lower = reference_answer.lower().strip()
    
    # Method 1: Sequence Matching (character-level similarity)
    seq_score = SequenceMatcher(None, student_lower, reference_lower).ratio()
    
    # Method 2: TF-IDF Cosine Similarity (word-level semantic similarity)
    try:
        if student_lower and reference_lower:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([reference_lower, student_lower])
            tfidf_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        else:
            tfidf_score = 0.0
    except Exception as e:
        print(f"TF-IDF error: {e}")
        tfidf_score = 0.0
    
    # Combined score (weighted average)
    final_score = (0.6 * seq_score + 0.4 * tfidf_score)
    
    return {
        'sequence_match': seq_score,
        'tfidf_similarity': tfidf_score,
        'final_score': final_score
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename or 'uploaded_image.jpg')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Check if recognizer is available
            if recognizer is None:
                return jsonify({'error': 'Recognizer not initialized'})
            
            print(f"Processing image: {filepath}")
            
            # Get predictions from BOTH models
            results = recognizer.predict_both(filepath)
            
            easyocr_text = results['easyocr']
            trocr_text = results['trocr']
            
            print(f"EasyOCR: '{easyocr_text}'")
            print(f"TrOCR: '{trocr_text}'")
            
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'easyocr_text': easyocr_text if easyocr_text else "(No text detected)",
                'trocr_text': trocr_text if trocr_text else "(No text detected)",
                # For backward compatibility, also include predicted_text
                'predicted_text': trocr_text if trocr_text else easyocr_text
            })
        except Exception as e:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"Error processing image: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/grade', methods=['POST'])
def grade_answer():
    """Grade a student's answer against a reference answer"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'})
        
        student_answer = data.get('student_answer', '')
        reference_answer = data.get('reference_answer', '')
        max_score = data.get('max_score', 10)
        
        if not student_answer:
            return jsonify({'error': 'No student answer provided'})
        
        if not reference_answer:
            return jsonify({'error': 'No reference answer provided'})
        
        # Calculate similarity
        similarity = calculate_similarity(student_answer, reference_answer)
        
        # Calculate grade
        grade = round(similarity['final_score'] * max_score, 2)
        
        return jsonify({
            'success': True,
            'student_answer': student_answer,
            'reference_answer': reference_answer,
            'grade': grade,
            'max_score': max_score,
            'percentage': round(similarity['final_score'] * 100, 2),
            'sequence_match': similarity['sequence_match'],
            'tfidf_similarity': similarity['tfidf_similarity']
        })
        
    except Exception as e:
        print(f"Grading error: {str(e)}")
        return jsonify({'error': f'Grading error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)