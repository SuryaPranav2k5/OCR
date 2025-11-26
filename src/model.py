import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def CTCLoss(y_true, y_pred):
    """CTC Loss function"""
    # Ensure y_true is of integer type
    y_true = tf.cast(y_true, tf.int32)
    
    # Get shapes
    batch_size = tf.shape(y_pred)[0]
    input_length = tf.shape(y_pred)[1]
    
    # Calculate actual label lengths (non-zero elements)
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1)
    
    # Create input length array - one value per sample
    input_length_array = input_length * tf.ones(shape=(batch_size,), dtype=tf.int32)
    
    # Ensure y_pred is in the correct format (log probabilities)
    # Add small epsilon to prevent log(0)
    y_pred = tf.math.log(y_pred + keras.backend.epsilon())
    
    # Compute CTC loss
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=label_length,
        logit_length=input_length_array,
        logits_time_major=False,  # Important: batch is first dimension
        blank_index=0  # Use 0 as blank index
    )
    
    return tf.reduce_mean(loss)

def build_model(input_shape, num_classes):
    """
    Build CNN-LSTM model for handwriting recognition
    
    Args:
        input_shape: (height, width, channels)
        num_classes: number of characters in vocabulary
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name="image")
    
    # Normalize images
    x = layers.Lambda(lambda x: x / 255.0)(inputs)
    
    # CNN Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # CNN Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # CNN Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # CNN Block 4
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Reshape for RNN
    # Get shape and reshape to (batch, time_steps, features)
    new_shape = ((input_shape[0] // 16) * (input_shape[1] // 16), 128)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    
    # Output layer
    # +1 for CTC blank token
    outputs = layers.Dense(num_classes + 1, activation='softmax', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='HandwritingRecognitionModel')
    return model

# Test model creation
if __name__ == "__main__":
    model = build_model(input_shape=(64, 256, 1), num_classes=80)
    model.summary()