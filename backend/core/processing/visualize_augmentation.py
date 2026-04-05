import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.processing.audio_loader import load_audio, extract_log_mel_spectrogram
from src.processing.augmentation import add_noise, time_stretch, pitch_shift

# Config
DATASET_PATH = "dataset-ESC-50/ESC-50-master"
META_FILE = os.path.join(DATASET_PATH, "meta", "esc50.csv")
AUDIO_DIR = os.path.join(DATASET_PATH, "audio")
OUTPUT_DIR = "visualizations"

def visualize_augmentation():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = pd.read_csv(META_FILE)
    sample_row = df.sample(1).iloc[0]
    file_path = os.path.join(AUDIO_DIR, sample_row['filename'])
    category = sample_row['category']
    
    print(f"Visualizing augmentation for: {category}")
    
    y = load_audio(file_path, duration=5.0)
    
    # Augmentations
    y_noise = add_noise(y)
    y_stretch = time_stretch(y, rate=1.2)
    # Fix length for stretch
    if len(y_stretch) > len(y): y_stretch = y_stretch[:len(y)]
    else: y_stretch = np.pad(y_stretch, (0, len(y) - len(y_stretch)))
    
    y_pitch = pitch_shift(y, n_steps=4)
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    # Original
    plt.subplot(4, 2, 1)
    librosa.display.waveshow(y, sr=16000)
    plt.title("Original Waveform")
    plt.subplot(4, 2, 2)
    librosa.display.specshow(extract_log_mel_spectrogram(y), sr=16000, x_axis='time', y_axis='mel')
    plt.title("Original Spectrogram")
    
    # Noise
    plt.subplot(4, 2, 3)
    librosa.display.waveshow(y_noise, sr=16000)
    plt.title("Noise Injection")
    plt.subplot(4, 2, 4)
    librosa.display.specshow(extract_log_mel_spectrogram(y_noise), sr=16000, x_axis='time', y_axis='mel')
    plt.title("Noisy Spectrogram")
    
    # Stretch
    plt.subplot(4, 2, 5)
    librosa.display.waveshow(y_stretch, sr=16000)
    plt.title("Time Stretch (Fast)")
    plt.subplot(4, 2, 6)
    librosa.display.specshow(extract_log_mel_spectrogram(y_stretch), sr=16000, x_axis='time', y_axis='mel')
    plt.title("Stretched Spectrogram")
    
    # Pitch
    plt.subplot(4, 2, 7)
    librosa.display.waveshow(y_pitch, sr=16000)
    plt.title("Pitch Shift (Higher)")
    plt.subplot(4, 2, 8)
    librosa.display.specshow(extract_log_mel_spectrogram(y_pitch), sr=16000, x_axis='time', y_axis='mel')
    plt.title("Shifted Spectrogram")
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "augmentation_samples.png")
    plt.savefig(output_path)
    print(f"✅ Augmentation visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_augmentation()
