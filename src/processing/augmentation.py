import librosa
import numpy as np

def add_noise(data, noise_factor=0.005):
    """
    Adds random white noise to the audio.
    """
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def time_stretch(data, rate=0.8):
    """
    Stretches the time (speeds up or slows down) without changing pitch.
    """
    return librosa.effects.time_stretch(y=data, rate=rate)

def pitch_shift(data, sr=16000, n_steps=2.0):
    """
    Shifts the pitch up or down without changing speed.
    """
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def augment_audio(y, sr=16000):
    """
    Applies a random augmentation to the audio clip.
    Returns the augmented audio array.
    """
    choice = np.random.choice(['noise', 'stretch', 'pitch', 'none'])
    
    if choice == 'noise':
        return add_noise(y)
    elif choice == 'stretch':
        rate = np.random.uniform(0.8, 1.2)
        # Handle length change
        y_stretched = time_stretch(y, rate)
        # Fix duration back to original
        target_len = len(y)
        if len(y_stretched) > target_len:
            return y_stretched[:target_len]
        else:
            return np.pad(y_stretched, (0, target_len - len(y_stretched)))
    elif choice == 'pitch':
        steps = np.random.uniform(-2, 2)
        return pitch_shift(y, sr, steps)
    else:
        return y
