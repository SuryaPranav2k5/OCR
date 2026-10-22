import os
import re
import string
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sentence-Transformers for true semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    HAS_SBERT = True
    print("✅ Sentence-Transformers loaded (all-MiniLM-L6-v2)")
except Exception as e:
    print(f"⚠️ Sentence-Transformers unavailable ({e}), semantic grading disabled")
    SBERT_MODEL = None
    HAS_SBERT = False

# Fuzzy matching for OCR-error tolerance
try:
    from fuzzywuzzy import fuzz
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

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

# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

_CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "wouldn't": "would not",
    "couldn't": "could not", "shouldn't": "should not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "it's": "it is", "i'm": "i am", "you're": "you are",
    "they're": "they are", "we're": "we are", "that's": "that is",
    "there's": "there is", "what's": "what is", "who's": "who is",
    "let's": "let us",
}

# Common OCR-confused character pairs for tolerance
_OCR_PAIRS = [
    ("0", "o"), ("1", "l"), ("1", "i"), ("5", "s"),
    ("rn", "m"), ("cl", "d"), ("vv", "w"),
]

def normalize_text(text: str) -> str:
    """Aggressive normalisation to reduce surface-level noise."""
    t = text.lower().strip()
    # Expand contractions
    for contr, expanded in _CONTRACTIONS.items():
        t = t.replace(contr, expanded)
    # Remove punctuation
    t = t.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_keywords(text: str, min_len: int = 4) -> list[str]:
    """
    Extract meaningful keywords from a reference answer.
    Drops stop-words, generic verbs, and very short tokens.
    """
    STOP_WORDS = {
        # determiners / pronouns / prepositions / conjunctions
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "because", "but", "and", "or",
        "if", "while", "about", "up", "its", "it", "this", "that", "these",
        "those", "which", "who", "whom", "what", "their", "them", "they",
        "also", "any",
        # generic verbs — too vague to be meaningful keywords
        "use", "uses", "used", "using", "make", "makes", "made", "making",
        "turn", "turns", "turned", "turning", "get", "gets", "got",
        "give", "gives", "gave", "given", "take", "takes", "took", "taken",
        "come", "comes", "came", "go", "goes", "went", "gone",
        "put", "puts", "let", "lets", "say", "says", "said",
        "know", "knows", "knew", "known", "think", "thinks", "thought",
        "see", "sees", "saw", "seen", "find", "finds", "found",
        "call", "calls", "called", "help", "helps", "helped",
        "become", "becomes", "became", "keep", "keeps", "kept",
        "begin", "begins", "began", "show", "shows", "showed", "shown",
        "like", "well", "also", "many", "much", "often", "still",
        # filler / structure words
        "however", "therefore", "thus", "hence", "moreover", "furthermore",
        "result", "results", "resulting", "example", "examples",
        "based", "called", "known", "refer", "refers", "referred",
        "include", "includes", "including", "involve", "involves",
        "form", "forms", "formed",
    }
    norm = normalize_text(text)
    words = norm.split()
    keywords = []
    seen = set()
    for w in words:
        if len(w) >= min_len and w not in STOP_WORDS and w not in seen:
            keywords.append(w)
            seen.add(w)
    return keywords


# ---------------------------------------------------------------------------
# Synonym / abbreviation map  (bidirectional)
# ---------------------------------------------------------------------------
# Each group contains words/phrases that should be treated as equivalent.
# When checking keyword coverage, if a keyword matches ANY synonym of a word
# the student used, it counts as matched.

_SYNONYM_GROUPS = [
    {"co2", "carbon dioxide", "carbondioxide"},
    {"o2", "oxygen"},
    {"h2o", "water"},
    {"glucose", "sugar"},
    {"light", "sunlight", "solar", "sun"},
    {"energy", "power", "atp"},
    {"plant", "plants", "vegetation", "flora"},
    {"animal", "animals", "fauna"},
    {"cell", "cells", "cellular"},
    {"dna", "deoxyribonucleic acid"},
    {"rna", "ribonucleic acid"},
    {"temp", "temperature"},
    {"govt", "government"},
    {"info", "information"},
    {"amt", "amount"},
    {"approx", "approximately"},
    {"equation", "formula", "expression"},
    {"increase", "rise", "grow", "growth"},
    {"decrease", "reduce", "decline", "drop", "fall"},
    {"produce", "create", "generate", "synthesize", "synthesis"},
    {"absorb", "absorbs", "absorption", "absorbing"},
    {"release", "emit", "releases", "emits"},
    {"convert", "converts", "conversion", "transform", "transforms"},
    {"important", "significant", "crucial", "essential", "vital"},
    {"because", "since", "due", "cause", "caused", "reason"},
]

# Build a lookup: word -> set of synonyms (including itself)
_SYN_LOOKUP: dict[str, set[str]] = {}
for group in _SYNONYM_GROUPS:
    for word in group:
        _SYN_LOOKUP.setdefault(word, set()).update(group)


def _get_synonyms(word: str) -> set[str]:
    """Return known synonyms of a word (including itself)."""
    return _SYN_LOOKUP.get(word, {word})


# ---------------------------------------------------------------------------
# Individual scoring dimensions
# ---------------------------------------------------------------------------

def score_semantic(student: str, reference: str) -> float:
    """Sentence-Transformers cosine similarity (0-1)."""
    if not HAS_SBERT or not SBERT_MODEL:
        return -1.0  # sentinel: unavailable
    emb = SBERT_MODEL.encode([student, reference], convert_to_tensor=True)
    cos = st_util.cos_sim(emb[0], emb[1]).item()
    return max(0.0, cos)  # clamp negatives


def score_keyword_coverage(student: str, reference: str) -> dict:
    """
    Extract keywords from the reference, then check how many appear in the
    student answer.  Uses three matching strategies:
      1. Exact substring match
      2. Synonym / abbreviation match
      3. Fuzzy match with partial-string tolerance (OCR errors)
    Returns dict with 'score', 'matched', 'missed', 'total'.
    """
    ref_kw = extract_keywords(reference)
    if not ref_kw:
        return {"score": 1.0, "matched": [], "missed": [], "total": 0}

    student_norm = normalize_text(student)
    student_words = set(student_norm.split())
    matched = []
    missed = []

    for kw in ref_kw:
        # --- Strategy 1: Exact substring match ---
        if kw in student_norm:
            matched.append(kw)
            continue

        # --- Strategy 2: Synonym / abbreviation match ---
        syns = _get_synonyms(kw)
        syn_found = False
        for syn in syns:
            if syn in student_norm:
                matched.append(kw)
                syn_found = True
                break
        if syn_found:
            continue

        # --- Strategy 3: Fuzzy match (OCR tolerance) ---
        if HAS_FUZZY:
            best_ratio = 0
            for sw in student_words:
                # Word-level fuzzy
                ratio = fuzz.ratio(kw, sw) / 100.0
                if ratio > best_ratio:
                    best_ratio = ratio
            # Also try partial match against full text (handles multi-word)
            partial = fuzz.partial_ratio(kw, student_norm) / 100.0
            best_ratio = max(best_ratio, partial)

            if best_ratio >= 0.80:
                matched.append(kw)
                continue

        missed.append(kw)

    coverage = len(matched) / len(ref_kw) if ref_kw else 0.0
    return {
        "score": coverage,
        "matched": matched,
        "missed": missed,
        "total": len(ref_kw),
    }


def score_ngram_overlap(student: str, reference: str) -> float:
    """
    ROUGE-L-style longest common subsequence ratio.
    Works at word level so word-order matters but minor insertions are ok.
    """
    s_words = normalize_text(student).split()
    r_words = normalize_text(reference).split()
    if not s_words or not r_words:
        return 0.0

    # LCS length via DP
    m, n = len(r_words), len(s_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if r_words[i - 1] == s_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    precision = lcs_len / n if n else 0
    recall = lcs_len / m if m else 0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def score_sequence(student: str, reference: str) -> float:
    """SequenceMatcher on normalised text."""
    return SequenceMatcher(
        None, normalize_text(student), normalize_text(reference)
    ).ratio()


def score_tfidf(student: str, reference: str) -> float:
    """TF-IDF cosine similarity on normalised text."""
    s, r = normalize_text(student), normalize_text(reference)
    if not s or not r:
        return 0.0
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([r, s])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main grading function
# ---------------------------------------------------------------------------

def calculate_grade(student_answer: str, reference_answer: str, max_score: float = 10):
    """
    Multi-dimensional grading engine.

    Dimensions & weights (when all methods available):
      Semantic (SBERT)     35%
      Keyword Coverage     25%
      N-gram Overlap       20%
      Sequence Match       10%
      TF-IDF               10%

    If SBERT is unavailable the weight is redistributed to other dimensions.
    """
    # --- Compute individual dimensions ----
    sem = score_semantic(student_answer, reference_answer)
    kw = score_keyword_coverage(student_answer, reference_answer)
    ngram = score_ngram_overlap(student_answer, reference_answer)
    seq = score_sequence(student_answer, reference_answer)
    tfidf = score_tfidf(student_answer, reference_answer)

    # --- Weighted combination ----
    if sem >= 0:  # SBERT available
        weights = {
            "semantic": 0.35,
            "keyword": 0.25,
            "ngram": 0.20,
            "sequence": 0.10,
            "tfidf": 0.10,
        }
        final = (
            weights["semantic"] * sem
            + weights["keyword"] * kw["score"]
            + weights["ngram"] * ngram
            + weights["sequence"] * seq
            + weights["tfidf"] * tfidf
        )
    else:
        # Fallback: redistribute semantic weight
        weights = {
            "semantic": 0.0,
            "keyword": 0.35,
            "ngram": 0.30,
            "sequence": 0.15,
            "tfidf": 0.20,
        }
        final = (
            weights["keyword"] * kw["score"]
            + weights["ngram"] * ngram
            + weights["sequence"] * seq
            + weights["tfidf"] * tfidf
        )

    # Clamp to [0, 1]
    final = max(0.0, min(1.0, final))
    grade = round(final * max_score, 2)
    percentage = round(final * 100, 2)

    # Letter grade + feedback
    if percentage >= 90:
        letter, feedback = "A+", "Outstanding! Thorough and accurate answer."
    elif percentage >= 80:
        letter, feedback = "A", "Excellent work — covers the key concepts well."
    elif percentage >= 70:
        letter, feedback = "B", "Good answer — most important points are present."
    elif percentage >= 60:
        letter, feedback = "C", "Satisfactory — some key concepts are missing."
    elif percentage >= 50:
        letter, feedback = "D", "Needs improvement — review the topic and add more detail."
    else:
        letter, feedback = "F", "Significant gaps — please revisit the material."

    return {
        "grade": grade,
        "max_score": max_score,
        "percentage": percentage,
        "letter_grade": letter,
        "feedback": feedback,
        # Detailed dimensions
        "dimensions": {
            "semantic_similarity": round(sem * 100, 1) if sem >= 0 else None,
            "keyword_coverage": round(kw["score"] * 100, 1),
            "ngram_overlap": round(ngram * 100, 1),
            "sequence_match": round(seq * 100, 1),
            "tfidf_similarity": round(tfidf * 100, 1),
        },
        "weights_used": {k: round(v * 100) for k, v in weights.items()},
        "keywords": {
            "matched": kw["matched"],
            "missed": kw["missed"],
            "total": kw["total"],
        },
    }


# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

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
                # For backward compatibility
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
        
        # Use the new multi-dimensional grading engine
        result = calculate_grade(student_answer, reference_answer, max_score)
        result['success'] = True
        result['student_answer'] = student_answer
        result['reference_answer'] = reference_answer
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Grading error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Grading error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)