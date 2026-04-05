from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import os
import sys
import json
from datetime import datetime, timedelta
from collections import Counter

# Add current directory to path to import web_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from web_utils import preprocess_for_inference, get_direction, CONFIDENCE_THRESHOLD

app = Flask(__name__)

# Config — prefer the curated Indian model if it exists, else fall back to robust
BASE = os.path.join(os.path.dirname(__file__), '..', '..')
_indian_model  = os.path.join(BASE, 'models', 'transfer_model_indian.h5')
_robust_model  = os.path.join(BASE, 'models', 'transfer_model_robust.h5')
_indian_classes = os.path.join(BASE, 'data', 'processed', 'indian_classes.npy')
_robust_classes = os.path.join(BASE, 'data', 'processed', 'classes.npy')

if os.path.exists(_indian_model):
    MODEL_PATH   = _indian_model
    CLASSES_PATH = _indian_classes
    print("🇮🇳 Using curated Indian model.")
else:
    MODEL_PATH   = _robust_model
    CLASSES_PATH = _robust_classes
    print("⚠️  Indian model not found — using robust model as fallback.")

SIAMESE_MODEL_PATH = os.path.join(BASE, 'models', 'siamese_model.h5')
list_classes = []
siamese_model = None

# Load Model
print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Classifier loaded.")
    
    if os.path.exists(SIAMESE_MODEL_PATH):
        def l1_dist(tensors):
            return tf.abs(tensors[0] - tensors[1])
        siamese_model = tf.keras.models.load_model(
            SIAMESE_MODEL_PATH,
            custom_objects={'tf': tf},
            safe_mode=False
        )
        print("✅ Siamese model loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Load Classes
print(f"Loading classes from {CLASSES_PATH}...")
try:
    list_classes = np.load(CLASSES_PATH, allow_pickle=True)
    print(f"✅ Classes loaded: {list_classes}")
except Exception as e:
    print(f"❌ Error loading classes: {e}")
    list_classes = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
        
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # Save temp file with proper extension
    ext = audio_file.filename.split('.')[-1] if '.' in audio_file.filename else 'wav'
    temp_path = f'temp_audio.{ext}'
    audio_file.save(temp_path)
    
    try:
        print(f"Processing audio file: {temp_path}")
        
        # Preprocess
        input_data = preprocess_for_inference(temp_path)
        
        if input_data is None:
            raise ValueError("Failed to preprocess audio - input_data is None")
        
        print(f"Input shape: {input_data.shape}")
        
        # Predict
        prediction = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        
        predicted_label = "Unknown"
        if len(list_classes) > 0:
            predicted_label = list_classes[predicted_index]
        
        # Low confidence → don't mislead with a wrong class
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_label = "Uncertain"

        # Get direction (Localization)
        direction = get_direction(temp_path)
        
        # --- Check against Custom Sounds (Siamese) ---
        custom_sound_detected = None
        custom_sounds_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'custom_sounds')
        
        if siamese_model and os.path.exists(custom_sounds_dir):
            for sound_name in os.listdir(custom_sounds_dir):
                sound_dir = os.path.join(custom_sounds_dir, sound_name)
                if not os.path.isdir(sound_dir): continue
                
                # Load samples and compare
                for sample_file in os.listdir(sound_dir):
                    if not sample_file.endswith('.webm'): continue
                    sample_path = os.path.join(sound_dir, sample_file)
                    
                    # Prepare pairs for Siamese (Input A = Live, Input B = Sample)
                    live_feat = input_data[0] # (64, 157, 1)
                    sample_feat_group = preprocess_for_inference(sample_path)
                    
                    if sample_feat_group is None:
                        print(f"Skipping custom sound {sample_file} due to load error (FFmpeg missing?)")
                        continue
                        
                    sample_feat = sample_feat_group[0]
                    
                    # In a real app, precompute sample_feat
                    score = siamese_model.predict([
                        np.expand_dims(live_feat, 0), 
                        np.expand_dims(sample_feat, 0)
                    ], verbose=0)[0][0]
                    
                    if score > 0.85: # High similarity
                        custom_sound_detected = sound_name.replace("_", " ")
                        predicted_label = f"Personal: {custom_sound_detected}"
                        confidence = float(score)
                        break
                if custom_sound_detected: break

        print(f"✅ Prediction: {predicted_label} ({confidence:.2%}) | Direction: {direction}")

        # Auto-save detection to history
        _save_detection(predicted_label, confidence, direction)

        result = {
            'class': predicted_label,
            'confidence': confidence,
            'direction': direction,
            'is_custom': custom_sound_detected is not None,
            'all_scores': prediction[0].tolist() if not custom_sound_detected else []
        }

        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_msg = f"Prediction error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.route('/learn', methods=['POST'])
def learn_sound():
    """
    Personalization: Receives 3 samples of a custom sound and saves them.
    The system will then use the Siamese network to compare real-time audio 
    against these samples.
    """
    sound_name = request.form.get('name')
    if not sound_name:
        return jsonify({'error': 'Sound name missing'}), 400
        
    # Create directory for custom sounds if not exists
    custom_sounds_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'custom_sounds')
    if not os.path.exists(custom_sounds_dir):
        os.makedirs(custom_sounds_dir)
        
    sound_dir = os.path.join(custom_sounds_dir, sound_name.replace(" ", "_"))
    if not os.path.exists(sound_dir):
        os.makedirs(sound_dir)
        
    # Save the 3 samples
    for i in range(1, 4):
        sample_key = f'sample_{i}'
        if sample_key in request.files:
            file = request.files[sample_key]
            file_path = os.path.join(sound_dir, f'sample_{i}.webm')
            file.save(file_path)
            print(f"Saved custom sample: {file_path}")
            
    return jsonify({'status': 'success', 'message': f'Learned {sound_name}'})

# ── Data helpers ───────────────────────────────────────────────────────────────
DETECTIONS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'detections.json')
FEEDBACK_FILE   = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'feedback.json')

EMERGENCY_SOUNDS = {
    'fire_alarm', 'siren', 'car_horn', 'crying_baby', 'glass_breaking',
    'dog', 'auto_rickshaw_horn', 'motorcycle_horn'
}

def _load_json(path):
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
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
        'class':      label,
        'confidence': round(confidence, 4),
        'direction':  direction,
        'priority':   'emergency' if label in EMERGENCY_SOUNDS else 'normal',
        'timestamp':  datetime.now().isoformat()
    })
    # Keep last 500 only
    _save_json(DETECTIONS_FILE, records[-500:])


# ── New endpoints ───────────────────────────────────────────────────────────────
@app.route('/history')
def history():
    records = _load_json(DETECTIONS_FILE)
    return jsonify(list(reversed(records[-50:])))


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json(force=True)
    records = _load_json(FEEDBACK_FILE)
    records.append({
        'detected':   data.get('detected'),
        'correct':    data.get('correct'),      # True / False
        'actual':     data.get('actual', ''),   # correct label if wrong
        'timestamp':  datetime.now().isoformat()
    })
    _save_json(FEEDBACK_FILE, records)
    return jsonify({'status': 'saved'})


@app.route('/analytics')
def analytics():
    records   = _load_json(DETECTIONS_FILE)
    feedbacks = _load_json(FEEDBACK_FILE)

    # Top sounds
    counts = Counter(r['class'] for r in records)
    top10  = [{'class': k, 'count': v}
               for k, v in counts.most_common(10)]

    # Daily counts — last 7 days
    today  = datetime.now().date()
    daily  = {}
    for i in range(6, -1, -1):
        day = (today - timedelta(days=i)).isoformat()
        daily[day] = 0
    for r in records:
        try:
            day = r['timestamp'][:10]
            if day in daily:
                daily[day] += 1
        except Exception:
            pass

    # Accuracy from feedback
    total_fb  = len(feedbacks)
    correct   = sum(1 for f in feedbacks if f.get('correct'))
    accuracy  = round(correct / total_fb * 100, 1) if total_fb else None

    emergency_count = sum(1 for r in records if r.get('priority') == 'emergency')
    avg_conf = round(sum(r['confidence'] for r in records) / len(records) * 100, 1) if records else 0

    return jsonify({
        'total':           len(records),
        'emergency_count': emergency_count,
        'avg_confidence':  avg_conf,
        'accuracy':        accuracy,
        'top10':           top10,
        'daily':           [{'date': k, 'count': v} for k, v in daily.items()]
    })


@app.route('/custom_sounds')
def custom_sounds():
    """List all saved custom sounds."""
    custom_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'custom_sounds')
    result = []
    if os.path.exists(custom_dir):
        for name in os.listdir(custom_dir):
            sound_dir = os.path.join(custom_dir, name)
            if os.path.isdir(sound_dir):
                samples = len([f for f in os.listdir(sound_dir)])
                result.append({'name': name.replace('_', ' '), 'samples': samples})
    return jsonify(result)


if __name__ == '__main__':
    # Disable reloader and threading — Keras C++ crashes on Windows threaded Flask
    app.run(debug=True, use_reloader=False, threaded=False, port=5000)
