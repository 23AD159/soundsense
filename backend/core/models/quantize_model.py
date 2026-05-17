import tensorflow as tf
import os
import numpy as np

# Handle Keras 3 safe mode if needed for Lambda layers
try:
    tf.keras.config.enable_unsafe_deserialization()
except:
    try:
        import keras
        keras.config.enable_unsafe_deserialization()
    except:
        pass

def contrastive_loss(y_true, y_pred):
    """
    Local definition to avoid import issues.
    """
    margin = 1
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def quantize_model(h5_path, tflite_path, classes_path=None):
    """
    Converts and quantizes an H5 model to TFLite (INT8).
    """
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found.")
        return

    print(f"Quantizing {h5_path}...")
    try:
        model = tf.keras.models.load_model(h5_path, custom_objects={'contrastive_loss': contrastive_loss})
    except:
        model = tf.keras.models.load_model(h5_path, compile=False)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
        print(f"[SUCCESS] Quantized model saved to {tflite_path}")
        print(f"Size: {os.path.getsize(tflite_path) / 1024 / 1024:.2f} MB")

    # Export labels for Android
    if classes_path and os.path.exists(classes_path):
        classes = np.load(classes_path, allow_pickle=True)
        with open(os.path.join(os.path.dirname(tflite_path), "labels.txt"), "w", encoding="utf-8") as f:
            for cls in classes:
                f.write(f"{cls}\n")
        print(f"[SUCCESS] labels.txt generated.")

if __name__ == "__main__":
    # 1. Quantize primary classification model
    quantize_model(
        "models/transfer_model_indian.h5", 
        "models/transfer_model_indian.tflite", 
        classes_path="data/processed/indian_classes.npy"
    )
    
    # 2. Quantize siamese model
    # Reconstruct architecture to avoid Lambda deserialization issues
    if os.path.exists("models/siamese_model.h5"):
        print("\nQuantizing Siamese model (via reconstruction)...")
        from siamese_network import create_siamese_network
        
        # Instantiate architecture
        s_model = create_siamese_network((64, 94, 1))
        
        # Load weights only
        try:
            s_model.load_weights("models/siamese_model.h5")
            print("[SUCCESS] Weights loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Weight loading failed, trying standard load: {e}")
            s_model = tf.keras.models.load_model("models/siamese_model.h5", compile=False)

        s_converter = tf.lite.TFLiteConverter.from_keras_model(s_model)
        s_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Support for custom Lambda distance layer
        s_converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        s_tflite_model = s_converter.convert()
        with open("models/siamese_model.tflite", "wb") as f:
            f.write(s_tflite_model)
        print("[SUCCESS] Siamese TFLite model saved.")
