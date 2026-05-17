import tensorflow as tf
from tensorflow.keras import layers, Model

def create_base_network(input_shape):
    """
    Base convolutional network for feature extraction (MobileNetV2 features).
    """
    # Use a simpler CNN or the head of MobileNetV2 for feature embeddings
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3), # TFLite prefers RGB-like input for MobileNet
        include_top=False, 
        pooling='avg'
    )
    
    # We'll use a wrapper to handle the 1-channel to 3-channel conversion internally
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(3, (3, 3), padding='same')(inputs) # Convert 1ch to 3ch
    x = base_model(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x) # L2 normalization for embeddings
    
    return Model(inputs, x, name="embedding_net")

def create_siamese_network(input_shape):
    """
    Siamese Network for calculating similarity between two sound samples.
    """
    embedding_net = create_base_network(input_shape)
    
    input_a = layers.Input(shape=input_shape, name="input_a")
    input_b = layers.Input(shape=input_shape, name="input_b")
    
    emb_a = embedding_net(input_a)
    emb_b = embedding_net(input_b)
    
    # Calculate Euclidean distance between embeddings
    distance = layers.Lambda(
        lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True))
    )([emb_a, emb_b])
    
    return Model(inputs=[input_a, input_b], outputs=distance, name="siamese_net")

if __name__ == "__main__":
    # Test architecture
    model = create_siamese_network((64, 94, 1))
    model.summary()
    print("✅ Siamese Network architecture created.")
