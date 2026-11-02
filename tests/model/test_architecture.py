from pathlib import Path

import pytest
import tensorflow as tf

from chihuahua_muffin.model.architecture import build_model, compile_model
from chihuahua_muffin.model.config import load_model_config


PROJECT_ROOT = Path(__file__).parents[2]


def test_approved_architecture_shapes_and_parameter_count() -> None:
    config = load_model_config(PROJECT_ROOT / "config/model.toml")
    model = build_model(config)

    assert model.input_shape == (None, 128, 128, 3)
    assert model.get_layer("conv1").output.shape == (None, 128, 128, 16)
    assert model.get_layer("pool1").output.shape == (None, 64, 64, 16)
    assert model.get_layer("conv2").output.shape == (None, 64, 64, 32)
    assert model.get_layer("pool2").output.shape == (None, 32, 32, 32)
    assert model.get_layer("conv3").output.shape == (None, 32, 32, 64)
    assert model.get_layer("pool3").output.shape == (None, 16, 16, 64)
    assert model.get_layer("global_average_pooling").output.shape == (None, 64)
    assert model.get_layer("dense32").output.shape == (None, 32)
    assert model.output_shape == (None, 1)
    assert model.count_params() == 25_697
    assert model.get_layer("muffin_probability").activation == tf.keras.activations.sigmoid


def test_compile_uses_approved_loss_optimizer_and_metrics() -> None:
    config = load_model_config(PROJECT_ROOT / "config/model.toml")
    model = build_model(config)
    compile_model(model, config)

    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(0.001)
    assert isinstance(model.loss, tf.keras.losses.BinaryCrossentropy)


def test_model_summary_records_the_approved_total() -> None:
    config = load_model_config(PROJECT_ROOT / "config/model.toml")
    model = build_model(config)
    lines: list[str] = []

    model.summary(print_fn=lines.append)

    summary = "\n".join(lines)
    assert "Total params: 25,697" in summary
