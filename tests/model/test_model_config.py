from pathlib import Path

import pytest

from chihuahua_muffin.model.config import ModelConfigError, load_model_config


PROJECT_ROOT = Path(__file__).parents[2]


def test_effective_model_config_is_strict_and_approved() -> None:
    config = load_model_config(PROJECT_ROOT / "config/model.toml")

    assert config.seed == 42
    assert config.architecture.expected_trainable_parameters == 25_697
    assert config.training.batch_size == 32
    assert config.training.max_epochs == 15
    assert config.training.threshold == 0.5
    assert config.quality_target.approved is True
    assert config.smoke.allow_cpu_by_default is False
    assert config.compatibility.tensorflow == "2.21.0"


def test_model_config_rejects_architecture_drift(tmp_path: Path) -> None:
    content = (PROJECT_ROOT / "config/model.toml").read_text(encoding="utf-8")
    path = tmp_path / "model.toml"
    path.write_text(
        content.replace("dense_units = 32", "dense_units = 64"),
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigError, match="dense_units"):
        load_model_config(path)


def test_model_config_rejects_unapproved_threshold(tmp_path: Path) -> None:
    content = (PROJECT_ROOT / "config/model.toml").read_text(encoding="utf-8")
    path = tmp_path / "model.toml"
    path.write_text(
        content.replace(
            "classification_threshold = 0.5",
            "classification_threshold = 0.4",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigError, match="classification_threshold"):
        load_model_config(path)
