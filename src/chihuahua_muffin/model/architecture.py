"""Approved small CNN architecture and compilation policy."""

from __future__ import annotations

import tensorflow as tf

from .config import ModelRunConfig


EXPECTED_TRAINABLE_PARAMETERS = 25_697


def build_model(config: ModelRunConfig) -> tf.keras.Model:
    """Build the exact from-scratch network approved by MODEL-006."""

    inputs = tf.keras.Input(shape=config.architecture.input_shape, name="image")
    x = inputs
    for index, filters in enumerate(config.architecture.conv_filters, start=1):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="relu",
            use_bias=True,
            name=f"conv{index}",
        )(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=2, name=f"pool{index}"
        )(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(32, activation="relu", use_bias=True, name="dense32")(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", use_bias=True, name="muffin_probability"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="small_cnn_v1")
    if model.count_params() != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"model has {model.count_params()} parameters; "
            f"expected {EXPECTED_TRAINABLE_PARAMETERS}"
        )
    return model


def compile_model(model: tf.keras.Model, config: ModelRunConfig) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.training.learning_rate
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        ],
    )
