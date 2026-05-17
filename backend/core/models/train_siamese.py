import os
import numpy as np
import tensorflow as tf
from siamese_network import create_siamese_network

# Config
DATA_DIR = "data/processed"
MODELS_DIR = "models"

def create_pairs(X, Y):
    """
    Creates positive and negative pairs for Siamese Network training.
    """
    pairs = []
    labels = []
    
    n_classes = len(np.unique(Y))
    digit_indices = [np.where(Y == i)[0] for i in range(n_classes)]
    
    for idx1 in range(len(X)):
        # Positive Pair
        label = Y[idx1]
        idx2 = np.random.choice(digit_indices[label])
        pairs += [[X[idx1], X[idx2]]]
        labels += [1.0]
        
        # Negative Pair
        negative_label = (label + np.random.randint(1, n_classes)) % n_classes
        idx3 = np.random.choice(digit_indices[negative_label])
        pairs += [[X[idx1], X[idx3]]]
        labels += [0.0]
        
    return np.array(pairs), np.array(labels)

def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss from Hadsell-et-al.'06
    """
    margin = 1
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def train_siamese():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    Y = np.load(os.path.join(DATA_DIR, "Y.npy"))
    
    input_shape = X.shape[1:]
    
    print("Creating pairs...")
    X_pairs, Y_labels = create_pairs(X, Y)
    
    # Split for Siamese
    # X_pairs.shape is (N, 2, 64, 94, 1)
    
    model = create_siamese_network(input_shape)
    model.compile(loss=contrastive_loss, optimizer='adam', metrics=['accuracy'])
    
    print("Starting Siamese Training...")
    model.fit(
        [X_pairs[:, 0], X_pairs[:, 1]], Y_labels,
        batch_size=32,
        epochs=30, # Increased epochs for better personalization
        validation_split=0.1
    )
    
    model.save(os.path.join(MODELS_DIR, "siamese_model.h5"))
    print("✅ Siamese model saved.")

if __name__ == "__main__":
    train_siamese()
