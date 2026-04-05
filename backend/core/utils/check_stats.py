import numpy as np
import os

DATA_DIR = "data/processed"

def check_stats():
    x_path = os.path.join(DATA_DIR, "X.npy")
    if not os.path.exists(x_path):
        print("Data not found.")
        return

    X = np.load(x_path)
    print(f"Shape: {X.shape}")
    print(f"Min: {X.min():.4f}")
    print(f"Max: {X.max():.4f}")
    print(f"Mean: {X.mean():.4f}")
    print(f"Std: {X.std():.4f}")
    
    if X.min() < -10 or X.max() > 10:
        print("\n⚠️ WARNING: Data values are outside typical NN range [-1, 1] or [0, 1].")
        print("Recommendation: Apply normalization (e.g., MinMax or Standard scaling).")
    else:
        print("\n✅ Data looks normalized.")

if __name__ == "__main__":
    check_stats()
