import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.baseline_cnn import create_baseline_model

# Config
DATA_DIR = "data/processed"
MODELS_DIR = "models"
BATCH_SIZE = 32
EPOCHS = 30

def train():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    print("Loading data...")
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    Y = np.load(os.path.join(DATA_DIR, "Y.npy"))
    classes = np.load(os.path.join(DATA_DIR, "classes.npy"))
    
    print(f"Data loaded: {X.shape}, {Y.shape}")
    
    # Split data
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Create model
    input_shape = X_train.shape[1:] # e.g. (64, 94, 1)
    num_classes = len(classes)
    
    model = create_baseline_model(input_shape, num_classes)
    model.summary()
    
    # Training
    print("Starting training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "baseline_cnn.h5")
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Plot results
    plot_history(history)

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(os.path.join(MODELS_DIR, "training_history.png"))
    print("✅ Training history saved.")

if __name__ == "__main__":
    train()
