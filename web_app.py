import os
import cv2
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow import keras
import tensorflow as tf

# Import our handwriting recognizer
from src.inference import HandwritingRecognizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize the recognizer
print("Initializing HandwritingRecognizer...")
try:
    recognizer = HandwritingRecognizer('Models/trained_model')
    print("HandwritingRecognizer initialized successfully")
except Exception as e:
    print(f"Error initializing HandwritingRecognizer: {e}")
    recognizer = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        filename = secure_filename(file.filename or '')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Check if recognizer is available
            if recognizer is None:
                return jsonify({'error': 'Recognizer not initialized'})
            
            print(f"Processing image: {filepath}")
            # Process the image and get prediction
            predicted_text = recognizer.predict(filepath)
            
            # Debug: Print the predicted text
            print(f"Predicted text for {filename}: '{predicted_text}'")
            
            # Clean up uploaded file
            os.remove(filepath)
            
            # Provide more informative feedback
            if not predicted_text:
                display_text = "(No text detected - this may indicate the model needs more training or the image quality is not suitable)"
            elif predicted_text == "(No text detected)":
                display_text = "(No text detected - this may indicate the model needs more training or the image quality is not suitable)"
            else:
                display_text = predicted_text
            
            return jsonify({
                'success': True,
                'filename': filename,
                'predicted_text': display_text
            })
        except Exception as e:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            print(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)