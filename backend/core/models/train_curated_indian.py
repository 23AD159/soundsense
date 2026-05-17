"""
Curated Indian SED Training Script
===================================
30 classes for maximum real-world impact for deaf people in India.

ESC-50 classes (40 files × 5 aug each = ~200 samples per class):
  crying_baby, dog, glass_breaking, car_horn, siren,
  door_wood_knock, clock_alarm, toilet_flush, washing_machine,
  pouring_water, crackling_fire, clapping, laughing, coughing,
  sneezing, footsteps, rain, thunderstorm, wind, chirping_birds,
  cat, rooster, train, engine, fireworks

FreeSound classes (augmented to 160 samples each):
  auto_rickshaw_horn, motorcycle_horn, temple_bells,
  fire_alarm, traditional_bell (doorbell)
"""

import os
import sys
import numpy as np
import tensorflow as tf
import librosa
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.transfer_learning import create_transfer_model

# ── Config ─────────────────────────────────────────────────────────────────────
ESC50_DIR       = "dataset-ESC-50/ESC-50-master/audio"
ESC50_META      = "dataset-ESC-50/ESC-50-master/meta/esc50.csv"
RAW_DIR         = "data/raw"
MODELS_DIR      = "models"
DATA_DIR        = "data/processed"
SAMPLE_RATE     = 16000
DURATION        = 3.0
N_MELS          = 64
N_FFT           = 1024
HOP_LENGTH      = 512
BATCH_SIZE      = 32
HEAD_EPOCHS     = 40
FINETUNE_EPOCHS = 30

# ── Class Mapping ──────────────────────────────────────────────────────────────
ESC50_CLASSES = [
    'dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow',
    'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops',
    'wind', 'pouring_water', 'toilet_flush', 'thunderstorm', 'crying_baby', 'sneezing',
    'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth',
    'snoring', 'drinking_sipping', 'door_wood_knock', 'mouse_click', 'keyboard_typing',
    'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm',
    'clock_tick', 'glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn', 
    'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
]

FREESOUND_CLASSES = [
    ("auto_rickshaw_horn", "auto_rickshaw_horn"),
    ("motorcycle_horn",    "motorcycle_horn"),
    ("temple_bells",       "temple_bells"),
    ("fire_alarm",         "fire_alarm"),
    ("traditional_bell",   "traditional_bell"),
]

ALL_CLASSES = ESC50_CLASSES + [name for name, _ in FREESOUND_CLASSES]
NUM_CLASSES = len(ALL_CLASSES)

print(f"Total classes: {NUM_CLASSES}")
print("Classes:", ALL_CLASSES)


# ── Audio Utilities ────────────────────────────────────────────────────────────
def load_audio(path):
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
        target = int(SAMPLE_RATE * DURATION)
        if len(y) < target:
            y = np.pad(y, (0, target - len(y)))
        else:
            y = y[:target]
        return y
    except Exception as e:
        print(f"  [SKIP] {os.path.basename(path)}: {e}")
        return None


def extract_spec(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    spec = np.clip((log_mel + 80.0) / 80.0, 0, 1)
    return spec[..., np.newaxis]  # (64, 94, 1)


def augment(y):
    """Random augmentation: pitch shift, time stretch, or white noise."""
    choice = random.random()
    try:
        if choice < 0.33:
            y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=random.uniform(-2, 2))
        elif choice < 0.66:
            y = librosa.effects.time_stretch(y, rate=random.uniform(0.85, 1.15))
        else:
            y = y + np.random.randn(len(y)) * 0.005
    except Exception:
        pass
    target = int(SAMPLE_RATE * DURATION)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    return y


# ── Dataset Building ───────────────────────────────────────────────────────────
def load_esc50_class(class_name, df, label_idx):
    """Load all ESC-50 audio for a class. Each file augmented 4x → ~200 samples."""
    rows = df[df['category'] == class_name]
    X, Y = [], []
    for _, row in rows.iterrows():
        y = load_audio(os.path.join(ESC50_DIR, row['filename']))
        if y is None:
            continue
        X.append(extract_spec(y))
        Y.append(label_idx)
        for _ in range(4):
            X.append(extract_spec(augment(y.copy())))
            Y.append(label_idx)
    return X, Y


def load_freesound_class(folder, label_idx, augment_to=160):
    """Load FreeSound audio and augment aggressively to reach augment_to samples."""
    X, Y = [], []
    folder_path = os.path.join(RAW_DIR, folder)

    if not os.path.exists(folder_path):
        print(f"  [WARN] Folder not found: {folder_path}")
        return X, Y

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    if not files:
        print(f"  [WARN] No audio files in {folder_path}")
        return X, Y

    originals = []
    for fname in files:
        y = load_audio(os.path.join(folder_path, fname))
        if y is not None:
            originals.append(y)
            X.append(extract_spec(y))
            Y.append(label_idx)

    if not originals:
        return X, Y

    while len(X) < augment_to:
        y_aug = augment(random.choice(originals).copy())
        X.append(extract_spec(y_aug))
        Y.append(label_idx)

    print(f"  [{folder}] {len(files)} files → {len(X)} samples after augmentation")
    return X, Y


def build_dataset():
    print(f"\n{'='*60}")
    print("Building curated Indian SED dataset...")
    print(f"{'='*60}\n")

    df = pd.read_csv(ESC50_META)
    X_all, Y_all = [], []

    print("Loading ESC-50 classes (4x augmentation each)...")
    for i, cls in enumerate(ESC50_CLASSES):
        X, Y = load_esc50_class(cls, df, i)
        print(f"  [{cls}] {len(X)} samples")
        X_all.extend(X)
        Y_all.extend(Y)

    print("\nLoading FreeSound classes (augmented to 160 each)...")
    for i, (name, folder) in enumerate(FREESOUND_CLASSES):
        X, Y = load_freesound_class(folder, len(ESC50_CLASSES) + i)
        X_all.extend(X)
        Y_all.extend(Y)

    X = np.array(X_all, dtype=np.float32)
    Y = np.array(Y_all, dtype=np.int32)

    print(f"\n✅ Dataset: {X.shape[0]} total samples, {NUM_CLASSES} classes")
    print(f"   (~{X.shape[0] // NUM_CLASSES} samples per class)")

    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "indian_classes.npy"), np.array(ALL_CLASSES))
    print(f"✅ Class labels saved → {DATA_DIR}/indian_classes.npy")

    return X, Y


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    X, Y = build_dataset()
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.15, random_state=42, stratify=Y
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    model = create_transfer_model(X_train.shape[1:], NUM_CLASSES)

    labels = np.unique(Y_train)
    weights = compute_class_weight('balanced', classes=labels, y=Y_train)
    class_weight_dict = dict(zip(labels, weights))

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(MODELS_DIR, "transfer_model_indian.h5"),
        monitor='val_accuracy', save_best_only=True, verbose=1
    )

    # ── Phase 1: Head training (no early stopping) ─────────────────────────────
    print(f"\nPhase 1: Training classification head ({HEAD_EPOCHS} epochs)...")
    lr1 = tf.keras.optimizers.schedules.CosineDecay(
        1e-4, decay_steps=(len(X_train) // BATCH_SIZE) * HEAD_EPOCHS
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(lr1),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
              epochs=HEAD_EPOCHS, batch_size=BATCH_SIZE,
              class_weight=class_weight_dict,
              callbacks=[checkpoint])

    # ── Phase 2: Fine-tune top 30 MobileNetV2 layers ──────────────────────────
    print(f"\nPhase 2: Fine-tuning top layers ({FINETUNE_EPOCHS} epochs)...")
    base_model = model.layers[2]
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    lr2 = tf.keras.optimizers.schedules.CosineDecay(
        1e-5, decay_steps=(len(X_train) // BATCH_SIZE) * FINETUNE_EPOCHS
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(lr2),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
              epochs=FINETUNE_EPOCHS, batch_size=BATCH_SIZE,
              class_weight=class_weight_dict,
              callbacks=[
                  checkpoint,
                  tf.keras.callbacks.EarlyStopping(
                      monitor='val_accuracy', patience=12, restore_best_weights=True
                  )
              ])

    loss, acc = model.evaluate(X_val, Y_val, verbose=0)
    print(f"\n{'='*60}")
    print(f"✅ Training complete!  val_accuracy = {acc:.2%}")
    print(f"✅ Model saved → models/transfer_model_indian.h5")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
