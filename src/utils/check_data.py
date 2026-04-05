import os
import pandas as pd
import librosa
import numpy as np

# Updated path based on user's folder structure
# Running from root directory
DATASET_PATH = "dataset-ESC-50/ESC-50-master"
META_FILE = os.path.join(DATASET_PATH, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(DATASET_PATH, "audio")

def check_dataset():
    print(f"Checking dataset at: {DATASET_PATH}")
    
    # 1. Check Metadata
    if not os.path.exists(META_FILE):
        print(f"❌ Metadata file not found at {META_FILE}")
        return
    
    try:
        df = pd.read_csv(META_FILE)
        print(f"✅ Metadata loaded. Found {len(df)} samples.")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Classes: {df['category'].nunique()} unique sound classes.")
    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return

    # 2. Check Audio Files
    if not os.path.exists(AUDIO_DIR):
        print(f"❌ Audio directory not found at {AUDIO_DIR}")
        return
        
    audio_files = os.listdir(AUDIO_DIR)
    print(f"✅ Audio directory found. Contains {len(audio_files)} files.")
    
    # 3. Verify Match
    meta_files = set(df['filename'].values)
    disk_files = set(audio_files)
    
    missing_files = meta_files - disk_files
    if missing_files:
        print(f"❌ Missing {len(missing_files)} audio files referenced in metadata.")
    else:
        print("✅ All metadata files exist on disk.")

    # 4. Sample Audio Load Test
    print("\nTesting sample audio load...")
    sample_file = os.path.join(AUDIO_DIR, df.iloc[0]['filename'])
    try:
        y, sr = librosa.load(sample_file, sr=None)
        print(f"✅ Successfully loaded {df.iloc[0]['filename']}")
        print(f"   Sample Rate: {sr}Hz, Duration: {len(y)/sr:.2f}s")
    except Exception as e:
        print(f"❌ Error loading sample audio: {e}")

if __name__ == "__main__":
    check_dataset()
