import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import librosa

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.transfer_learning import create_transfer_model
from src.processing.augmentation import augment_audio
from src.processing.audio_loader import load_audio, extract_log_mel_spectrogram

# Config
DATA_DIR = "data/processed"
MODELS_DIR = "models"
BATCH_SIZE = 32
HEAD_EPOCHS = 40
FINE_TUNE_EPOCHS = 40

def get_background_sounds():
    """
    Identifies 'noisy' classes from ESC-50 to use as background noise.
    """
    META_FILE = "dataset-ESC-50/ESC-50-master/meta/esc50.csv"
    import pandas as pd
    df = pd.read_csv(META_FILE)
    
    # Selecting classes that work well as background noise
    bg_classes = ['rain', 'wind', 'insects', 'sea_waves', 'crackling_fire', 'thunderstorm']
    bg_files = df[df['category'].isin(bg_classes)]['filename'].tolist()
    
    AUDIO_DIR = "dataset-ESC-50/ESC-50-master/audio"
    bg_paths = [os.path.join(AUDIO_DIR, f) for f in bg_files]
    return bg_paths

def load_robust_dataset():
    """
    Creates a robust dataset with heavy noise injection and augmentation.
    """
    import pandas as pd
    META_FILE = "dataset-ESC-50/ESC-50-master/meta/esc50.csv"
    AUDIO_DIR = "dataset-ESC-50/ESC-50-master/audio"
    
    df = pd.read_csv(META_FILE)
    bg_paths = get_background_sounds()
    
    X = []
    Y = []
    
    # Use native mapping from classes.npy
    classes = list(np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True))
    
    print(f"Building Robust Dataset (2000 base files)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_path = os.path.join(AUDIO_DIR, row['filename'])
        label = row['target'] # Use native target ID
        
        # 1. Load Original (3s)
        y = load_audio(file_path, duration=3.0)
        if y is None: continue
        
        # Process and save clean
        spec = extract_log_mel_spectrogram(y)[..., np.newaxis]
        spec = np.clip((spec + 80.0) / 80.0, 0, 1)
        X.append(spec)
        Y.append(label)
        
        # 2. Add Noisy/Augmented Version
        # We add ONE augmented version per file to keep dataset size manageable (4000 total)
        y_aug = y.copy()
        
        # Randomly choose augmentation
        choice = random.random()
        if choice < 0.3:
            y_aug = augment_audio(y_aug) # Pitch/Stretch
        elif choice < 0.6:
            # Noise injection from another class
            bg_y = load_audio(random.choice(bg_paths), duration=3.0)
            if bg_y is not None:
                snr = random.uniform(5, 15) # 5dB to 15dB
                # Simple mix: y_aug = y + noise * scale
                # Avoid division by zero
                y_std = np.std(y_aug) if np.std(y_aug) > 0 else 1e-6
                bg_std = np.std(bg_y) if np.std(bg_y) > 0 else 1e-6
                noise_scale = np.sqrt(y_std**2 / (bg_std**2 * (10**(snr/10))))
                y_aug = y_aug + bg_y * noise_scale
        else:
            # Combined
            y_aug = augment_audio(y_aug)
            bg_y = load_audio(random.choice(bg_paths), duration=3.0)
            if bg_y is not None:
                snr = random.uniform(10, 25)
                y_std = np.std(y_aug) if np.std(y_aug) > 0 else 1e-6
                bg_std = np.std(bg_y) if np.std(bg_y) > 0 else 1e-6
                noise_scale = np.sqrt(y_std**2 / (bg_std**2 * (10**(snr/10))))
                y_aug = y_aug + bg_y * noise_scale
                
        # Normalize amplitude after mixing
        if np.max(np.abs(y_aug)) > 0:
            y_aug = y_aug / np.max(np.abs(y_aug))
            
        spec_aug = extract_log_mel_spectrogram(y_aug)[..., np.newaxis]
        spec_aug = np.clip((spec_aug + 80.0) / 80.0, 0, 1)
        X.append(spec_aug)
        Y.append(label)
        
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def train_robust():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    X, Y = load_robust_dataset()
    classes = np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True)
    num_classes = len(classes)
    
    # Split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
    
    input_shape = X_train.shape[1:]
    model = create_transfer_model(input_shape, num_classes)
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODELS_DIR, "transfer_model_robust.h5"), 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1
    )
    
    # Cosine Decay Schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=(len(X_train) // BATCH_SIZE) * HEAD_EPOCHS
    )
    
    print(f"\nPhase 1: Head Training ({HEAD_EPOCHS} epochs)...")
    
    # Calculate class weights for unbalanced augmented data
    classes_labels = np.unique(Y_train)
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight(class_weight='balanced', classes=classes_labels, y=Y_train)
    class_weight_dict = {classes_labels[i]: weights[i] for i in range(len(classes_labels))}
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=HEAD_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[checkpoint]
    )
    
    print("\nPhase 2: Fine-Tuning Top 30 layers (60 epochs)...")
    base_model = model.layers[2]
    base_model.trainable = True
    
    # Fine tune more layers for better specialization
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    # Lower learning rate for fine tuning
    lr_schedule_ft = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,
        decay_steps=(len(X_train) // BATCH_SIZE) * FINE_TUNE_EPOCHS
    )
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_ft),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=FINE_TUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[checkpoint]
    )
    
    print("✅ Robust training complete. Model saved to models/transfer_model_robust.h5")

if __name__ == "__main__":
    train_robust()
