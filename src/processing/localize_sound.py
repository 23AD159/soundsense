import numpy as np
import matplotlib.pyplot as plt

def gcc_phat(sig, refsig, fs=16000, max_tau=None, interp=16):
    '''
    Generalized Cross Correlation - Phase Transform (GCC-PHAT)
    Calculates time delay between two signals.
    '''
    # Make sure length of signals is the same
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # Find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    
    return tau, cc

def simulate_and_localize():
    """
    Simulates a stereo recording (Left/Right Mics) and estimates direction.
    """
    fs = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(fs*duration))
    
    # Create a synthetic sound source (Sine wave burst)
    source_signal = np.sin(2 * np.pi * 440 * t) * np.exp(-10 * t)
    
    print("--- Simulation 1: Sound from LEFT ---")
    # Left mic hears it first (0ms delay), Right mic hears it later (e.g., 0.5ms delay)
    # Delay of 0.5ms approx 17cm distance (width of a head)
    delay_samples = int(0.0005 * fs) 
    
    mic_left = source_signal
    mic_right = np.roll(source_signal, delay_samples) # Delayed
    
    tau, cc = gcc_phat(mic_left, mic_right, fs=fs)
    print(f"Detected Delay: {tau*1000:.4f} ms")
    
    if tau < -0.0001:
        print("Direction: ➡️ RIGHT")
    elif tau > 0.0001:
        print("Direction: ⬅️ LEFT")
    else:
        print("Direction: ⬆️ CENTER")
        
    print("\n--- Simulation 2: Sound from RIGHT ---")
    # Right mic hears it first
    mic_right_2 = source_signal
    mic_left_2 = np.roll(source_signal, delay_samples)
    
    tau2, _ = gcc_phat(mic_left_2, mic_right_2, fs=fs)
    print(f"Detected Delay: {tau2*1000:.4f} ms")
    
    if tau2 < -0.0001:
        print("Direction: ➡️ RIGHT")
    elif tau2 > 0.0001:
        print("Direction: ⬅️ LEFT")
    else:
        print("Direction: ⬆️ CENTER")

if __name__ == "__main__":
    simulate_and_localize()
