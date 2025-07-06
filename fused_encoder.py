#fused_encoder.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16

EMBEDDING_SIZE = 128

def create_fused_encoder():
    # Image input
    image_input = layers.Input(shape=(224, 224, 3), name="image_input")
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=image_input)
    for layer in base_model.layers[:15]:
        layer.trainable = False
    x_img = base_model.output
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.Dense(512, activation='relu')(x_img)

    # Audio input (YAMNet sequence shape: [None, 1024])
    audio_input = layers.Input(shape=(None, 1024), name="audio_input")
    x_audio = layers.Bidirectional(layers.LSTM(128))(audio_input)
    x_audio = layers.Dense(256, activation='relu')(x_audio)

    # Fusion
    x = layers.Concatenate()([x_img, x_audio])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(EMBEDDING_SIZE)(x)
    x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)

    return Model(inputs=[image_input, audio_input], outputs=x, name="fused_encoder")
