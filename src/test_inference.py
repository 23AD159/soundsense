import os
import sys
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.processing.audio_loader import load_audio, extract_log_mel_spectrogram

import shutil

def test_inference():
    MODEL_PATH = "models/transfer_model_robust.h5"
    TEMP_MODEL_PATH = "models/transfer_model_tmp.h5"
    CLASSES_PATH = "data/processed/classes.npy"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    print(f"Copying model for safe loading...")
    shutil.copy2(MODEL_PATH, TEMP_MODEL_PATH)

    print(f"Loading model: {TEMP_MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(TEMP_MODEL_PATH)
    finally:
        # We can theoretically delete after load, but some h5 engines keep it open
        pass
    
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    
    # Pick a few sample sounds from data/raw to test
    test_samples = []
    raw_dir = "data/raw"
    
    for root, dirs, files in os.walk(raw_dir):
        wav_files = [f for f in files if f.endswith('.wav') or f.endswith('.mp3')]
        if wav_files:
            category = os.path.basename(root)
            test_samples.append((os.path.join(root, wav_files[0]), category))
            if len(test_samples) >= 3: break

    if not test_samples:
        print("❌ No raw samples found to test.")
        return

    print("\n--- Smoke Test Results ---")
    for file_path, actual_label in test_samples:
        y = load_audio(file_path, duration=3.0)
        if y is None: continue
        
        # Preprocess
        spec = extract_log_mel_spectrogram(y)[..., np.newaxis]
        spec = np.clip((spec + 80.0) / 80.0, 0, 1) # Normalize
        input_data = np.expand_dims(spec, axis=0)
        
        # Predict
        preds = model.predict(input_data, verbose=0)
        idx = np.argmax(preds[0])
        pred_label = classes[idx]
        confidence = preds[0][idx]
        
        print(f"File: {os.path.basename(file_path)}")
        print(f"  Actual: {actual_label}")
        print(f"  Predicted: {pred_label} ({confidence:.2%})")
        print("-" * 20)

if __name__ == "__main__":
    test_inference()
