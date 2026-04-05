import numpy as np
import os

def check_project_status():
    print("=== Sound Event Detection Project: Sprint Verification ===")
    
    # 1. Dataset Status
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    
    if os.path.exists(raw_dir):
        categories = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
        print(f"Dataset: {len(categories)} categories found in {raw_dir}")
        for cat in categories:
            files = os.listdir(os.path.join(raw_dir, cat))
            status = "✅" if len(files) > 0 else "❌ Empty"
            print(f"  - {cat}: {len(files)} files {status}")
    else:
        print("❌ Dataset: data/raw directory is missing!")

    # 2. Processed Data
    if os.path.exists(processed_dir):
        files = os.listdir(processed_dir)
        if "X.npy" in files and "classes.npy" in files:
            X = np.load(os.path.join(processed_dir, "X.npy"), allow_pickle=True)
            classes = np.load(os.path.join(processed_dir, "classes.npy"), allow_pickle=True)
            print(f"✅ Preprocessing: {X.shape[0]} samples processed, {len(classes)} classes")
        else:
            print("❌ Preprocessing: Processed files (X.npy, classes.npy) not found.")
    else:
        print("❌ Preprocessing: data/processed directory is missing.")

    # 3. Model Status
    models_dir = "models"
    model_paths = {
        "Baseline (H5)": "transfer_model_robust.h5",
        "TFLite (INT8)": "transfer_model.tflite",
        "Siamese (H5)": "siamese_model.h5"
    }
    print("Models:")
    for name, path in model_paths.items():
        full_path = os.path.join(models_dir, path)
        status = "✅ Found" if os.path.exists(full_path) else "❌ Not trained yet"
        print(f"  - {name}: {status}")

    # 4. Android Infrastructure
    android_path = "src/android/app/src/main/java/com/example/soundalert"
    critical_files = ["LocalizationEngine.kt", "AlertManager.kt", "SiameseClassifier.kt", "VideoActivity.kt"]
    print("Android Infrastructure:")
    for f in critical_files:
        full_path = os.path.join(android_path, f)
        status = "✅ Ready" if os.path.exists(full_path) else "❌ Missing"
        print(f"  - {f}: {status}")

if __name__ == "__main__":
    check_project_status()
