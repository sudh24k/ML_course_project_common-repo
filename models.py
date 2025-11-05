"""
Description:
Deep CNN with Residual Blocks, Multi-Scale Convolution, 
Squeeze-Excitation (SE) attention, and CBAM (Channel + Spatial attention)
Designed for learning bilirubin-driven discriminative skin patterns.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


# --------------------- Attention Blocks --------------------- #

def squeeze_excitation(inputs, reduction=16):
    filters = inputs.shape[-1]
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Dense(filters // reduction, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1,1,filters))(se)
    return layers.Multiply()([inputs, se])


def cbam_block(inputs, reduction_ratio=8):
    filters = inputs.shape[-1]

    # Channel Attention
    avg_pool = layers.GlobalAveragePooling2D()(inputs)
    avg_pool = layers.Reshape((1,1,filters))(avg_pool)
    max_pool = layers.GlobalMaxPooling2D()(inputs)
    max_pool = layers.Reshape((1,1,filters))(max_pool)

    mlp = layers.Dense(filters // reduction_ratio, activation='relu')
    mlp_out = layers.Dense(filters)

    avg_out = mlp_out(mlp(avg_pool))
    max_out = mlp_out(mlp(max_pool))

    channel_attention = layers.Activation('sigmoid')(avg_out + max_out)
    channel_refined = layers.Multiply()([inputs, channel_attention])

    # Spatial Attention
    avg_channel = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
    max_channel = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_channel, max_channel])

    spatial = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    refined = layers.Multiply()([channel_refined, spatial])

    return refined


# --------------------- Building Blocks --------------------- #

def conv_block(x, filters, kernel_size, strides=1):
    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def residual_block(x, filters):
    shortcut = x

    x = conv_block(x, filters, 3)
    x = conv_block(x, filters, 3)

    # SE + CBAM
    x = squeeze_excitation(x)
    x = cbam_block(x)

    # Residual connection
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, padding="same")(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def multi_scale_block(x, filters):
    b1 = conv_block(x, filters, 1)
    b2 = conv_block(x, filters, 3)
    b3 = conv_block(x, filters, 5)
    b4 = layers.MaxPooling2D(pool_size=3, strides=1, padding="same")(x)

    x = layers.Concatenate()([b1, b2, b3, b4])
    x = layers.Conv2D(filters, kernel_size=1, padding="same")(x)  # fusion layer
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


# --------------------- Final Custom CNN --------------------- #

def build_custom_jaundice_cnn(input_shape=(224,224,3), num_classes=2):
    inputs = layers.Input(shape=input_shape)

    # Initial Stem
    x = conv_block(inputs, 64, 7, strides=2)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # Multi-Scale Feature Extractor
    x = multi_scale_block(x, 64)

    # Deep Residual Attention Blocks
    x = residual_block(x, 128)
    x = layers.MaxPooling2D(2)(x)
    
    x = residual_block(x, 256)
    x = layers.MaxPooling2D(2)(x)

    x = residual_block(x, 256)
    x = layers.MaxPooling2D(2)(x)

    # High-Level CBAM
    x = cbam_block(x)

    # Global context aggregation
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # Classifier
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="Custom_CBAM_SE_Residual_MultiScale_CNN")

    return model


# --------------------- Build Model --------------------- #

if __name__ == "__main__":
    model = build_custom_jaundice_cnn()
    model.summary()
