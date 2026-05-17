import tensorflow as tf
import os
import numpy as np

# Config
MODELS_DIR = "models"
KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "transfer_model.h5")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "sound_classifier.tflite")

def convert_model(keras_path, tflite_path):
    if not os.path.exists(keras_path):
        print(f"Error: Model file {keras_path} not found.")
        return

    print(f"Loading Keras model from {keras_path}...")
    try:
        model = tf.keras.models.load_model(keras_path)
    except Exception:
        # Retry with safe_mode=False for custom layers (Lambda)
        model = tf.keras.models.load_model(keras_path, safe_mode=False)
    
    # Initialize converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional: Optimization for mobile (Compression)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    print(f"Converting {os.path.basename(keras_path)} to TFLite...")
    tflite_model = converter.convert()
    
    # Save
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"✅ TFLite model saved to {tflite_path}")
    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"Model Size: {size_kb:.2f} KB")

if __name__ == "__main__":
    # 1. Main Classifier
    convert_model(
        os.path.join(MODELS_DIR, "transfer_model.h5"), 
        os.path.join(MODELS_DIR, "sound_classifier.tflite")
    )
    
    # 2. Siamese Network (Embedding portion)
    siamese_embed_path = os.path.join(MODELS_DIR, "siamese_embedding.h5")
    if os.path.exists(siamese_embed_path):
        convert_model(
            siamese_embed_path,
            os.path.join(MODELS_DIR, "siamese_embedding.tflite")
        )
