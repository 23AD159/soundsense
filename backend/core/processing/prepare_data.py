import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.core.processing.audio_loader import load_audio, extract_log_mel_spectrogram

# Config
DATASET_PATH = "dataset-ESC-50/ESC-50-master"
META_FILE = os.path.join(DATASET_PATH, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(DATASET_PATH, "audio")
PROCESSED_DIR = "backend/data/processed"

def prepare_dataset():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    X = [] # Features
    Y = [] # Labels
    categories = []

    # 1. Process Custom Indian Dataset (data/raw)
    RAW_DIR = "backend/data/raw"
    if os.path.exists(RAW_DIR):
        print(f"Scanning {RAW_DIR} for custom Indian sounds...")
        for root, dirs, files in os.walk(RAW_DIR):
            category = os.path.basename(root)
            if category == "raw": continue # Skip root dir
            
            wav_files = [f for f in files if f.endswith('.wav')]
            if not wav_files: continue
            
            print(f"Processing {len(wav_files)} files in category: {category}")
            for filename in wav_files:
                file_path = os.path.join(root, filename)
                y = load_audio(file_path, duration=3.0)
                if y is None: continue
                
                spec = extract_log_mel_spectrogram(y)[..., np.newaxis]
                X.append(spec)
                categories.append(category)

    # 2. Process ESC-50 (Fallback/Baseline)
    if os.path.exists(META_FILE):
        print("Loading ESC-50 metadata...")
        df = pd.read_csv(META_FILE)
        print(f"Processing {len(df)} ESC-50 audio files...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            file_path = os.path.join(AUDIO_DIR, row['filename'])
            category = row['category']
            
            y = load_audio(file_path, duration=3.0)
            if y is None: continue
            
            spec = extract_log_mel_spectrogram(y)[..., np.newaxis]
            X.append(spec)
            categories.append(category)
    else:
        print(f"Skipping ESC-50: Metadata not found at {META_FILE}")

    if not X:
        print("❌ Error: No data found in data/raw or ESC-50.")
        return

    X = np.array(X)
    # Normalize
    X = np.clip((X + 80.0) / 80.0, 0, 1)
    
    # Encode Labels
    le = LabelEncoder()
    Y_encoded = le.fit_transform(categories)
    classes = le.classes_
    
    print(f"Normalized Data Range: Min={X.min():.2f}, Max={X.max():.2f}")
    print(f"Features shape: {X.shape}")
    print(f"Total Categories: {len(classes)}")
    
    np.save(os.path.join(PROCESSED_DIR, "X.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "Y.npy"), Y_encoded)
    np.save(os.path.join(PROCESSED_DIR, "classes.npy"), classes)
    
    print(f"✅ Data saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    prepare_dataset()
