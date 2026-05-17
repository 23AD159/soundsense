import tensorflow as tf
from tensorflow.keras import layers, models, applications

def create_transfer_model(input_shape, num_classes):
    """
    Creates a Transfer Learning model using MobileNetV2.
    Input shape must be adapted because MobileNet expects 3 channels (RGB),
    but our Spectrograms have 1 channel (Grayscale).
    """
    inputs = layers.Input(shape=input_shape)
    
    # 1. Convert 1-channel Spectrogram to 3-channel (RGB-like) image
    # We repeat the grayscale channel 3 times
    x = layers.Conv2D(3, (3, 3), padding='same')(inputs)
    
    # 2. Load MobileNetV2 (Pre-trained on ImageNet)
    # include_top=False means we remove the last validation layers
    base_model = applications.MobileNetV2(
        input_shape=None, # Inferred from input tensor
        include_top=False, 
        weights='imagenet'
    )
    
    # Freeze the base model (don't retrain existing "knowledge")
    base_model.trainable = False
    
    # Pass our data through MobileNet
    x = base_model(x, training=False)
    
    # 3. Add our custom classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Test model build
    model = create_transfer_model((64, 94, 1), 50)
    model.summary()
