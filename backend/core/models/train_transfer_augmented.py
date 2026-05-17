import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.transfer_learning import create_transfer_model
from src.processing.augmentation import augment_audio
from src.processing.audio_loader import extract_log_mel_spectrogram

# Config
DATA_DIR = "data/processed"
MODELS_DIR = "models"
BATCH_SIZE = 32
EPOCHS = 30

def load_and_augment_data():
    """
    Loads raw ESC-50 data and creates an augmented dataset via raw audio transformations.
    """
    import pandas as pd
    from src.processing.audio_loader import load_audio, extract_log_mel_spectrogram
    
    META_FILE = "dataset-ESC-50/ESC-50-master/meta/esc50.csv"
    AUDIO_DIR = "dataset-ESC-50/ESC-50-master/audio"
    
    print("Loading metadata for raw augmentation...")
    df = pd.read_csv(META_FILE)
    
    X = []
    Y = []
    
    # Load alphabetical classes for consistent mapping (matches prepare_data)
    classes = list(np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True))
    
    print(f"Generating augmented samples for 2000 files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_path = os.path.join(AUDIO_DIR, row['filename'])
        category = row['category']
        label = classes.index(category) # Map string to alphabetical index
        
        # 1. Load Clean
        y = load_audio(file_path, duration=3.0)
        if y is None: continue
        
        # Extract Clean
        spec = extract_log_mel_spectrogram(y)[..., np.newaxis]
        # Normalize
        spec = np.clip((spec + 80.0) / 80.0, 0, 1)
        X.append(spec)
        Y.append(label)
        
        # 2. Add Augmented Version (100% chance for robustness)
        y_aug = y.copy()
        y_aug = augment_audio(y_aug)
        
        # Extract Augmented
        spec_aug = extract_log_mel_spectrogram(y_aug)[..., np.newaxis]
        # Normalize
        spec_aug = np.clip((spec_aug + 80.0) / 80.0, 0, 1)
        X.append(spec_aug)
        Y.append(label)
        
    X = np.array(X)
    Y = np.array(Y)
    
    print(f"Combined shape: {X.shape}")
    return X, Y

def train_augmented():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    X, Y = load_and_augment_data()
    classes = np.load(os.path.join(DATA_DIR, "classes.npy"), allow_pickle=True)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)
    
    input_shape = X_train.shape[1:]
    num_classes = len(classes)
    
    model = create_transfer_model(input_shape, num_classes)
    
    # Higher learning rate for initial phase might help
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("Starting Augmented Training...")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODELS_DIR, "transfer_model_aug.h5"), 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1
    )
    
    print("Phase 1: Training Classification Head...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=15, # Shorter initial phase
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint]
    )
    
    # Phase 2: Fine-Tuning
    print("\nPhase 2: Fine-Tuning Top Layers...")
    # Unfreeze the base model
    base_model = model.layers[2] # MobileNetV2 is the 3rd layer (after Input and Conv2D-3ch)
    base_model.trainable = True
    
    # Let's freeze everything except the last 20 layers
    fine_tune_at = len(base_model.layers) - 20
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), # Tiny learning rate
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    history_fine = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=15,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint]
    )
    
    print("✅ Augmented model training complete.")

if __name__ == "__main__":
    train_augmented()
