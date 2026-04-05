import librosa
import numpy as np
import os

def load_audio(file_path, sr=16000, duration=5.0):
    """
    Load audio file, resample to target sample rate, and fix duration.
    If shorter, pad with zeros. If longer, truncate.
    """
    try:
        # Load audio
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        # Padding if necessary
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
            
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_log_mel_spectrogram(y, sr=16000, n_mels=64, n_fft=1024, hop_length=512):
    """
    Convert audio waveform to Log Mel-Spectrogram.
    Returns shape: (n_mels, time_steps)
    """
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    
    # Convert to Log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

if __name__ == "__main__":
    # Test block
    print("Audio loader module ready.")
