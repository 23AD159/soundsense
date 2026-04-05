import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import pandas as pd
import sys

# Add src to path to import local modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.processing.audio_loader import load_audio, extract_log_mel_spectrogram

# Config
# Running from root directory of execution
DATASET_PATH = "dataset-ESC-50/ESC-50-master"
META_FILE = os.path.join(DATASET_PATH, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(DATASET_PATH, "audio")
OUTPUT_DIR = "visualizations"

def visualize_samples(n_samples=5):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if not os.path.exists(META_FILE):
        print(f"Error: Metadata not found at {META_FILE}. Run from project root.")
        return

    df = pd.read_csv(META_FILE)
    
    # Pick random samples
    samples = df.sample(n_samples)
    
    plt.figure(figsize=(15, 10))
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        file_path = os.path.join(AUDIO_DIR, row['filename'])
        category = row['category']
        
        # Process
        y = load_audio(file_path)
        if y is None: continue
        
        log_mel = extract_log_mel_spectrogram(y)
        
        # Plot
        plt.subplot(n_samples, 2, 2*i + 1)
        librosa.display.waveshow(y, sr=16000)
        plt.title(f"Waveform: {category}")
        
        plt.subplot(n_samples, 2, 2*i + 2)
        librosa.display.specshow(log_mel, sr=16000, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel-Spectrogram (Input to CNN): {category}")
        
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "spectrogram_samples.png")
    plt.savefig(output_path)
    print(f"✅ Visualization saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    visualize_samples()
