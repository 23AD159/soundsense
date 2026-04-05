import os

def check_file(path, description):
    if os.path.exists(path):
        print(f"✅ [FOUND] {description}: {path}")
        return True
    else:
        print(f"❌ [MISSING] {description}: {path}")
        return False

def verify_project():
    print("--- 🔍 Starting Final Project Verification ---\n")
    
    # 1. Models
    print("Checking AI Models (The 'Brain')...")
    check_file("models/transfer_model.h5", "Transfer Learning Model")
    check_file("models/siamese_model.h5", "Personalization Model")
    check_file("models/sound_classifier.tflite", "Mobile Optimized Model")
    
    # 2. Android
    print("\nChecking Android App (The 'Body')...")
    check_file("src/android/build.gradle", "Root Gradle")
    check_file("src/android/app/src/main/AndroidManifest.xml", "Manifest")
    check_file("src/android/app/src/main/java/com/example/soundalert/MainActivity.kt", "MainActivity")
    check_file("src/android/app/src/main/java/com/example/soundalert/SoundClassifier.kt", "SoundClassifier")
    check_file("src/android/app/src/main/java/com/example/soundalert/AudioProcessor.kt", "AudioProcessor")
    check_file("src/android/app/src/main/java/com/example/soundalert/VibrationHelper.kt", "VibrationHelper")
    
    # 3. Documentation
    print("\nChecking Documentation...")
    check_file("PROJECT_REPORT.md", "Final Project Report")
    check_file("README.md", "ReadMe File")
    
    # 4. Visuals
    print("\nChecking Visualizations...")
    check_file("visualizations/spectrogram_samples.png", "Spectrogram Examples")
    
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_project()
