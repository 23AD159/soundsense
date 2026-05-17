from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os
import sys
import json
from datetime import datetime, timedelta
from collections import Counter

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from web_utils import preprocess_for_inference, get_direction, CONFIDENCE_THRESHOLD, load_stereo_audio

app = Flask(__name__, template_folder='../../frontend/web/templates', static_folder='../../frontend/web/static')

# ---------------------------------------------------------------------------
# Project root and configuration paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Model files
INDIAN_MODEL_PATH = os.path.join(PROJECT_ROOT, 'backend', 'models', 'transfer_model_indian.h5')
ROBUST_MODEL_PATH = os.path.join(PROJECT_ROOT, 'backend', 'models', 'transfer_model_robust.h5')
SIAMESE_MODEL_PATH = os.path.join(PROJECT_ROOT, 'backend', 'models', 'siamese_model.h5')

# Class label files
INDIAN_CLASSES_PATH = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'indian_classes.npy')
ROBUST_CLASSES_PATH = os.path.join(PROJECT_ROOT, 'backend', 'data', 'processed', 'classes.npy')

# Directory for custom‑sound samples (single definition)
CUSTOM_SOUNDS_DIR = os.path.join(PROJECT_ROOT, 'data', 'custom_sounds')

# ---------------------------------------------------------------------------
# Choose which classifier to use
# ---------------------------------------------------------------------------
if os.path.exists(INDIAN_MODEL_PATH) and os.path.exists(INDIAN_CLASSES_PATH):
    MODEL_PATH = INDIAN_MODEL_PATH
    CLASSES_PATH = INDIAN_CLASSES_PATH
    print('[INFO] Using curated Indian model.')
else:
    MODEL_PATH = ROBUST_MODEL_PATH
    CLASSES_PATH = ROBUST_CLASSES_PATH
    print('[WARN] Indian model not found – falling back to robust model.')

# ---------------------------------------------------------------------------
# Global objects – model, siamese model, class list
# ---------------------------------------------------------------------------
model = None
siamese_model = None
list_classes = []

# Load the main classifier
print(f'Loading model from {MODEL_PATH} ...')
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print('[OK] Classifier loaded.')
except Exception as e:
    print(f'[ERROR] Failed to load classifier: {e}')
    model = None

# Load Siamese model if it exists
if os.path.exists(SIAMESE_MODEL_PATH):
    try:
        siamese_model = tf.keras.models.load_model(
            SIAMESE_MODEL_PATH,
            custom_objects={'tf': tf},
            compile=False,
            safe_mode=False
        )
        print('[OK] Siamese model loaded.')
    except Exception as e:
        print(f'[ERROR] Failed to load Siamese model: {e}')
        siamese_model = None
else:
    print('[INFO] No Siamese model found – personalization disabled.')

# Load class labels – fallback to empty list if missing
print(f'Loading class labels from {CLASSES_PATH} ...')
try:
    list_classes = np.load(CLASSES_PATH, allow_pickle=True).tolist()
    print(f'[OK] Loaded {len(list_classes)} class labels.')
except Exception as e:
    print(f'[ERROR] Could not load class labels: {e}')
    list_classes = []

# Cache for custom sound features to prevent preprocessing on every request
custom_sound_features_cache = {}

def load_custom_sounds_into_cache():
    if not siamese_model or not os.path.isdir(CUSTOM_SOUNDS_DIR):
        return
    for sound_name in os.listdir(CUSTOM_SOUNDS_DIR):
        sound_path = os.path.join(CUSTOM_SOUNDS_DIR, sound_name)
        if not os.path.isdir(sound_path):
            continue
        if sound_name not in custom_sound_features_cache:
            custom_sound_features_cache[sound_name] = []
        for sample_file in os.listdir(sound_path):
            if not sample_file.lower().endswith('.webm') and not sample_file.lower().endswith('.wav'):
                continue
            sample_path = os.path.join(sound_path, sample_file)
            
            # check if we already have it
            if any(s.get('path') == sample_path for s in custom_sound_features_cache[sound_name]):
                continue
                
            sample_feat = preprocess_for_inference(sample_path)
            if sample_feat is not None:
                custom_sound_features_cache[sound_name].append({
                    'path': sample_path,
                    'feat': sample_feat[0]
                })

# Load cache on startup
load_custom_sounds_into_cache()

# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
DETECTIONS_FILE = os.path.join(PROJECT_ROOT, 'data', 'detections.json')
FEEDBACK_FILE = os.path.join(PROJECT_ROOT, 'data', 'feedback.json')

EMERGENCY_SOUNDS = {
    'fire_alarm', 'siren', 'car_horn', 'crying_baby', 'glass_breaking',
    'dog', 'auto_rickshaw_horn', 'motorcycle_horn'
}

def _load_json(path):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f'[WARN] Failed loading JSON from {path}: {e}')
    return []

def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)

def _save_detection(label, confidence, direction):
    if label in ('Uncertain', 'Unknown'):
        return
    records = _load_json(DETECTIONS_FILE)
    records.append({
        'class': label,
        'confidence': round(confidence, 4),
        'direction': direction,
        'priority': 'emergency' if label in EMERGENCY_SOUNDS else 'normal',
        'timestamp': datetime.now().isoformat()
    })
    _save_json(DETECTIONS_FILE, records[-500:])

# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    ext = audio_file.filename.rsplit('.', 1)[-1] if '.' in audio_file.filename else 'wav'
    temp_path = f'temp_audio.{ext}'
    audio_file.save(temp_path)
    
    # If the file is webm (e.g. from browser), convert to wav first to prevent librosa/audioread hangs
    if ext.lower() == 'webm':
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(temp_path, format="webm")
            wav_path = 'temp_audio.wav'
            audio.export(wav_path, format="wav")
            # Remove original webm and update temp_path
            if os.path.exists(temp_path):
                os.remove(temp_path)
            temp_path = wav_path
        except Exception as e:
            print(f"[ERROR] WebM conversion failed in app.py: {e}")
            # Fall back to trying to load it directly, though it might hang

    try:
        # Load audio once for both inference and direction to save time and memory
        y_stereo = load_stereo_audio(temp_path)
        if y_stereo is None:
            raise RuntimeError('Failed to load audio file')
            
        # Mono for inference
        if y_stereo.shape[0] > 1:
            y_mono = np.mean(y_stereo, axis=0)
        else:
            y_mono = y_stereo[0]
            
        input_data = preprocess_for_inference(y_mono, from_file=False)

        if input_data is None:
            raise RuntimeError('Preprocessing failed – returned None')
        pred = model.predict(input_data, verbose=0)
        pred_idx = int(np.argmax(pred[0]))
        confidence = float(np.max(pred[0]))
        predicted_label = list_classes[pred_idx] if list_classes else f'class_{pred_idx}'
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_label = 'Uncertain'
        
        # Pass the already loaded stereo audio for direction
        direction = get_direction(y_stereo, from_file=False)
        
        # Personalisation via Siamese
        custom_sound_detected = None
        if siamese_model and os.path.isdir(CUSTOM_SOUNDS_DIR):
            live_feat = input_data[0]
            # Ensure cache is up to date (in case new sounds were learned)
            load_custom_sounds_into_cache()
            for sound_name, samples in custom_sound_features_cache.items():
                for sample_data in samples:
                    sample_feat = sample_data['feat']
                    score = siamese_model.predict([
                        np.expand_dims(live_feat, 0),
                        np.expand_dims(sample_feat, 0)
                    ], verbose=0)[0][0]
                    if score > 0.85:
                        custom_sound_detected = sound_name.replace('_', ' ')
                        predicted_label = f'Personal: {custom_sound_detected}'
                        confidence = float(score)
                        break
                if custom_sound_detected:
                    break
        _save_detection(predicted_label, confidence, direction)
        result = {
            'class': predicted_label,
            'confidence': confidence,
            'direction': direction,
            'is_custom': custom_sound_detected is not None,
            'all_scores': pred[0].tolist() if not custom_sound_detected else []
        }
        return jsonify(result)
    except Exception as exc:
        import traceback
        err_msg = f'Prediction error: {exc}'
        print(err_msg)
        print(traceback.format_exc())
        return jsonify({'error': err_msg}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

@app.route('/learn', methods=['POST'])
def learn_sound():
    """Accept three samples for a custom sound and store them for later Siamese comparison."""
    sound_name = request.form.get('name')
    if not sound_name:
        return jsonify({'error': 'Missing sound name'}), 400
    os.makedirs(CUSTOM_SOUNDS_DIR, exist_ok=True)
    sound_dir = os.path.join(CUSTOM_SOUNDS_DIR, sound_name.replace(' ', '_'))
    os.makedirs(sound_dir, exist_ok=True)
    for i in range(1, 4):
        key = f'sample_{i}'
        if key in request.files:
            file = request.files[key]
            dst = os.path.join(sound_dir, f'sample_{i}.webm')
            file.save(dst)
            print(f'Saved custom sample: {dst}')
    return jsonify({'status': 'success', 'message': f'Learned {sound_name}'})

# ---------------------------------------------------------------------------
# History / analytics endpoints
# ---------------------------------------------------------------------------
@app.route('/history')
def history():
    records = _load_json(DETECTIONS_FILE)
    return jsonify(list(reversed(records[-50:])))

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json(force=True)
    records = _load_json(FEEDBACK_FILE)
    records.append({
        'detected': data.get('detected'),
        'correct': data.get('correct'),
        'actual': data.get('actual', ''),
        'timestamp': datetime.now().isoformat()
    })
    _save_json(FEEDBACK_FILE, records)
    return jsonify({'status': 'saved'})

@app.route('/analytics')
def analytics():
    detections = _load_json(DETECTIONS_FILE)
    feedbacks = _load_json(FEEDBACK_FILE)
    # Top‑10 sounds
    counts = Counter(rec.get('class') for rec in detections)
    top10 = [{'class': k, 'count': v} for k, v in counts.most_common(10)]
    # Daily stats (last 7 days)
    today = datetime.now().date()
    daily = {(today - timedelta(days=i)).isoformat(): 0 for i in range(7)}
    for rec in detections:
        try:
            day = rec['timestamp'][:10]
            if day in daily:
                daily[day] += 1
        except Exception:
            pass
    # Accuracy from feedback
    total_fb = len(feedbacks)
    correct = sum(1 for f in feedbacks if f.get('correct'))
    accuracy = round(correct / total_fb * 100, 1) if total_fb else None
    emergency_count = sum(1 for r in detections if r.get('priority') == 'emergency')
    avg_conf = round(sum(r['confidence'] for r in detections) / len(detections) * 100, 1) if detections else 0
    return jsonify({
        'total': len(detections),
        'emergency_count': emergency_count,
        'avg_confidence': avg_conf,
        'accuracy': accuracy,
        'top10': top10,
        'daily': [{'date': d, 'count': c} for d, c in daily.items()]
    })

@app.route('/custom_sounds')
def custom_sounds():
    """List stored custom sound names and the number of samples for each."""
    result = []
    if os.path.isdir(CUSTOM_SOUNDS_DIR):
        for name in os.listdir(CUSTOM_SOUNDS_DIR):
            dir_path = os.path.join(CUSTOM_SOUNDS_DIR, name)
            if os.path.isdir(dir_path):
                sample_count = len([f for f in os.listdir(dir_path) if f.lower().endswith('.webm')])
                result.append({'name': name.replace('_', ' '), 'samples': sample_count})
    return jsonify(result)

# ---------------------------------------------------------------------------
# Application entry‑point – bind to Render's PORT if provided
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, use_reloader=False, threaded=False, port=port)
