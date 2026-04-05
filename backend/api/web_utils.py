import librosa
import numpy as np
import os
import io

# Audio Configuration (Must match training)
SAMPLE_RATE = 16000
DURATION = 3.0   # Training uses duration=3.0 (see train_transfer_robust.py line 62)
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
CONFIDENCE_THRESHOLD = 0.12  # Below this, report as 'Uncertain'

def load_stereo_audio(file_path):
    """
    Load stereo audio file and resample.
    """
    try:
        # Load stereo
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=False, duration=DURATION)
        
        # If mono, convert to stereo by duplicating
        if len(y.shape) == 1:
            y = np.array([y, y])
            
        # Reshape if necessary (channels, samples)
        if y.shape[0] > 2:
            y = y[:2]
            
        # Target length
        target_length = int(SAMPLE_RATE * DURATION)
        
        # Padding/Truncating for both channels
        processed_channels = []
        for i in range(2):
            channel = y[i]
            if len(channel) < target_length:
                channel = np.pad(channel, (0, target_length - len(channel)))
            else:
                channel = channel[:target_length]
            processed_channels.append(channel)
            
        return np.array(processed_channels)
    except Exception as e:
        print(f"Error loading stereo audio: {e}")
        return None

def gcc_phat(sig, refsig, fs=16000, interp=16):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    max_shift = int(interp * n / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau

def get_direction(file_path):
    """
    Estimate sound direction from stereo file.
    """
    y_stereo = load_stereo_audio(file_path)
    if y_stereo is None or y_stereo.shape[0] < 2:
        return "CENTER"
        
    mic_left = y_stereo[0]
    mic_right = y_stereo[1]
    
    tau = gcc_phat(mic_left, mic_right, fs=SAMPLE_RATE)
    
    if tau < -0.0001:
        return "RIGHT"
    elif tau > 0.0001:
        return "LEFT"
    else:
        return "CENTER"

def load_audio_from_file(file_path):
    """
    Load audio file, resample, and fix duration to 5 seconds.
    Handles both WAV and WebM files.
    """
    try:
        print(f"[DEBUG] Attempting to load: {file_path}")
        print(f"[DEBUG] File exists: {os.path.exists(file_path)}")
        
        # For WAV files, use librosa directly (fast and simple)
        if file_path.endswith('.wav'):
            print("[DEBUG] WAV file detected, loading directly...")
            y, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
            print(f"[DEBUG] WAV loaded successfully: shape={y.shape}")
        
        # For WebM files, try conversion (requires FFmpeg)
        elif file_path.endswith('.webm'):
            print("[DEBUG] WebM file detected - NOTE: This requires FFmpeg!")
            print("[DEBUG] Consider using WAV format instead for better compatibility.")
            try:
                from pydub import AudioSegment
                import tempfile
                
                print("[DEBUG] Attempting WebM conversion with pydub...")
                audio = AudioSegment.from_file(file_path, format="webm")
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    temp_wav_path = temp_wav.name
                    audio.export(temp_wav_path, format="wav")
                
                y, _ = librosa.load(temp_wav_path, sr=SAMPLE_RATE, duration=DURATION)
                os.unlink(temp_wav_path)
                print(f"[DEBUG] WebM converted and loaded: shape={y.shape}")
                
            except Exception as e:
                print(f"[ERROR] WebM conversion failed: {type(e).__name__}: {e}")
                print("[ERROR] Install FFmpeg or use WAV format instead!")
                raise
        else:
            # For other formats
            print(f"[DEBUG] Unknown format, trying librosa directly...")
            y, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Padding/Truncating
        target_length = int(SAMPLE_RATE * DURATION)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        
        print(f"[DEBUG] Final audio shape: {y.shape}")
        return y
    except Exception as e:
        print(f"[ERROR] Error loading audio: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def normalize_audio(y):
    """
    RMS-normalize audio to a fixed target loudness.
    ESC-50 training files have consistent loudness. External files (YouTube, etc.)
    often have very different levels which breaks the mel-spectrogram.
    This ensures the audio is always at the same RMS level before feature extraction.
    """
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 1e-6:  # Avoid division by zero on silent audio
        y = y / rms * 0.1  # Target RMS of 0.1 (same as typical ESC-50 level)
    return y

def extract_features(y):
    """
    Convert audio waveform to Log Mel-Spectrogram.
    Normalization MUST match training pipeline in train_transfer_robust.py.
    """
    if y is None: return None

    # Normalize loudness so external recordings match ESC-50 training levels
    y = normalize_audio(y)

    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Convert to Log scale (dB) - use ref=np.max to match training
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Add channel dimension (H, W) -> (H, W, 1)
    log_mel_spec = log_mel_spec[..., np.newaxis]

    return log_mel_spec

def preprocess_for_inference(file_path):

    """
    Full pipeline: Load -> Extract -> Normalize -> Batch Dimension.
    Returns: (1, 64, 157, 1) ready for model.predict()
    """
    # For model prediction, we use mono
    y = load_audio_from_file(file_path)
    if y is None: return None
    
    # Extract features
    features = extract_features(y)
    
    # Normalize (Fixed range -80dB to 0dB -> [0, 1])
    # librosa.power_to_db(ref=np.max) scales the top to 0dB and floor to -80dB by default
    features = np.clip((features + 80.0) / 80.0, 0, 1)
        
    # Add batch dimension: (1, H, W, C)
    features = np.expand_dims(features, axis=0)
    
    return features
