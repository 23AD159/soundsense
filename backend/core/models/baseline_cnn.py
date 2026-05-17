import tensorflow as tf
from tensorflow.keras import layers, models

def create_baseline_model(input_shape, num_classes):
    """
    Creates a simple 2D CNN model for audio classification.
    Input shape: (n_mels, time_steps, 1) -> e.g. (64, 94, 1)
    """
    model = models.Sequential([
        # Layer 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Layer 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Layer 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Test model build
    model = create_baseline_model((64, 94, 1), 50)
    model.summary()
