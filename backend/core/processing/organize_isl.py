import os

def organize_isl_assets():
    """
    Helps the user map sound labels to ISL video files for the Android app.
    """
    labels_path = "backend/models/labels.txt"
    raw_res_dir = "frontend/android/app/src/main/res/raw"
    
    if not os.path.exists(labels_path):
        print(f"❌ Error: {labels_path} not found. Run quantization first.")
        return

    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    print("=== ISL Video Asset Organizer ===")
    print(f"Total Labels: {len(labels)}")
    print(f"Target Directory: {raw_res_dir}\n")

    if not os.path.exists(raw_res_dir):
        os.makedirs(raw_res_dir)
        print(f"Created directory: {raw_res_dir}")

    existing_videos = os.listdir(raw_res_dir)
    missing = []

    print(f"{'Sound Label':<30} | {'Expected Filename':<30} | {'Status'}")
    print("-" * 80)

    for label in labels:
        # Clean label for filename (lowercase, no spaces)
        clean_label = label.lower().replace(" ", "_").replace("-", "_")
        expected_name = f"isl_{clean_label}.mp4"
        
        status = "✅ Found" if expected_name in existing_videos else "❌ Missing"
        if expected_name not in existing_videos:
            missing.append(expected_name)
            
        print(f"{label:<30} | {expected_name:<30} | {status}")

    print("-" * 80)
    print(f"\nSummary: {len(labels) - len(missing)}/{len(labels)} videos present.")
    if missing:
        print(f"Please add the missing .mp4 files to {raw_res_dir}")

if __name__ == "__main__":
    organize_isl_assets()
