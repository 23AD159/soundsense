import numpy as np
import matplotlib.pyplot as plt

def generate_stereo_signal(frequency=440, fs=16000, duration=1.0, lag_samples=10):
    """
    Generates a stereo signal with a time delay (lag) between channels.
    Positive lag means Right channel is delayed (Sound is coming from LEFT).
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Add noise
    signal += np.random.normal(0, 0.1, signal.shape)
    
    left = signal
    # Shift right channel
    if lag_samples > 0:
        right = np.concatenate([np.zeros(lag_samples), signal[:-lag_samples]])
    elif lag_samples < 0:
        right = np.concatenate([signal[-lag_samples:], np.zeros(-lag_samples)])
    else:
        right = signal
        
    return left, right

def gcc_phat_test(left, right, max_lag=20):
    """
    Simplified Cross-Correlation (similar to our Kotlin implementation).
    """
    best_lag = 0
    max_corr = -1.0
    
    # Narrow window cross-correlation
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr = np.sum(left[lag:] * right[:-lag])
        elif lag < 0:
            corr = np.sum(left[:lag] * right[-lag:])
        else:
            corr = np.sum(left * right)
            
        if corr > max_corr:
            max_corr = corr
            best_lag = lag
            
    return best_lag

def run_simulation():
    print("Testing Sound Localization Simulator...")
    fs = 16000
    
    test_cases = [
        ("Sound from LEFT", -10),  # Negative lag in our generator = Left first
        ("Sound from CENTER", 0),
        ("Sound from RIGHT", 10)
    ]
    
    for label, lag in test_cases:
        left, right = generate_stereo_signal(fs=fs, lag_samples=lag)
        detected_lag = gcc_phat_test(left, right)
        
        direction = "CENTER"
        if detected_lag > 3: direction = "RIGHT"
        elif detected_lag < -3: direction = "LEFT"
        
        status = "✅" if direction in label.upper() else "❌"
        print(f"{label}: Lag {lag} -> Detected {detected_lag} ({direction}) {status}")

if __name__ == "__main__":
    run_simulation()
