"""Strict runtime configuration for the approved SPEC 02 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import tomllib
from typing import Any


DEFAULT_MODEL_CONFIG = Path("config/model.toml")


class ModelConfigError(ValueError):
    """Raised when runtime configuration diverges from the approved baseline."""


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    manifest: Path
    index: Path
    quality_report: Path
    manifest_sha256: str
    index_sha256: str


@dataclass(frozen=True, slots=True)
class ArchitectureConfig:
    name: str
    input_shape: tuple[int, int, int]
    conv_filters: tuple[int, int, int]
    dense_units: int
    expected_trainable_parameters: int


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    monitor: str
    patience: int
    min_delta: float
    restore_best_weights: bool


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    monitor: str
    mode: str
    save_best_only: bool


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int
    learning_rate: float
    max_epochs: int
    threshold: float
    require_gpu: bool
    deterministic_ops: bool
    early_stopping: EarlyStoppingConfig
    checkpoint: CheckpointConfig


@dataclass(frozen=True, slots=True)
class QualityTarget:
    validation_accuracy: float
    validation_macro_f1: float
    approved: bool


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    epochs: int
    samples_per_class_per_split: int
    allow_cpu_by_default: bool


@dataclass(frozen=True, slots=True)
class CompatibilityConfig:
    python: str
    tensorflow: str
    keras: str
    tensorflow_build_cuda: str
    tensorflow_build_cudnn: str
    official_tested_cuda: str
    official_tested_cudnn: str


@dataclass(frozen=True, slots=True)
class ModelRunConfig:
    source_path: Path
    source_bytes: bytes
    schema_version: str
    seed: int
    dataset: DatasetPaths
    architecture: ArchitectureConfig
    training: TrainingConfig
    quality_target: QualityTarget
    smoke: SmokeConfig
    runs_dir: Path
    compatibility: CompatibilityConfig


def _table(container: dict[str, Any], key: str, field: str) -> dict[str, Any]:
    value = container.get(key)
    if type(value) is not dict:
        raise ModelConfigError(f"{field} must be a TOML table")
    return value


def _value(
    container: dict[str, Any], key: str, expected_type: type, field: str
) -> Any:
    if key not in container:
        raise ModelConfigError(f"{field} is required")
    value = container[key]
    if type(value) is not expected_type:
        raise ModelConfigError(f"{field} must have type {expected_type.__name__}")
    return value


def _exact(field: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ModelConfigError(
            f"{field} must be {expected!r}; received {actual!r}"
        )


def _relative_path(value: str, field: str) -> Path:
    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ModelConfigError(f"{field} must be a safe project-relative path")
    return Path(*pure.parts)


def _hex_digest(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ModelConfigError(f"{field} must be a lowercase SHA-256 digest")
    return value


def load_model_config(path: str | Path = DEFAULT_MODEL_CONFIG) -> ModelRunConfig:
    """Load the effective config and reject any scientific-policy divergence."""

    source_path = Path(path)
    try:
        source_bytes = source_path.read_bytes()
        document = tomllib.loads(source_bytes.decode("utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ModelConfigError(f"cannot load model config {source_path}: {error}") from error
    if type(document) is not dict:
        raise ModelConfigError("model config must be a TOML document")

    schema = _value(document, "schema_version", str, "schema_version")
    _exact("schema_version", schema, "1.0")
    project = _table(document, "project", "project")
    seed = _value(project, "seed", int, "project.seed")
    _exact("project.seed", seed, 42)

    dataset = _table(document, "dataset", "dataset")
    dataset_paths = DatasetPaths(
        manifest=_relative_path(
            _value(dataset, "manifest", str, "dataset.manifest"),
            "dataset.manifest",
        ),
        index=_relative_path(
            _value(dataset, "index", str, "dataset.index"), "dataset.index"
        ),
        quality_report=_relative_path(
            _value(dataset, "quality_report", str, "dataset.quality_report"),
            "dataset.quality_report",
        ),
        manifest_sha256=_hex_digest(
            _value(dataset, "manifest_sha256", str, "dataset.manifest_sha256"),
            "dataset.manifest_sha256",
        ),
        index_sha256=_hex_digest(
            _value(dataset, "index_sha256", str, "dataset.index_sha256"),
            "dataset.index_sha256",
        ),
    )

    model = _table(document, "model", "model")
    approved_model_values = {
        "architecture": "small_cnn_v1",
        "input_shape": [128, 128, 3],
        "conv_filters": [16, 32, 64],
        "conv_kernel": [3, 3],
        "conv_padding": "same",
        "conv_stride": 1,
        "activation": "relu",
        "pool_size": [2, 2],
        "pool_stride": 2,
        "use_bias": True,
        "global_average_pooling": True,
        "dense_units": 32,
        "output_units": 1,
        "output_activation": "sigmoid",
        "expected_trainable_parameters": 25697,
        "transfer_learning": False,
    }
    for field, expected in approved_model_values.items():
        _exact(f"model.{field}", model.get(field), expected)
    architecture = ArchitectureConfig(
        name="small_cnn_v1",
        input_shape=(128, 128, 3),
        conv_filters=(16, 32, 64),
        dense_units=32,
        expected_trainable_parameters=25697,
    )

    training = _table(document, "training", "training")
    for field, expected in {
        "batch_size": 32,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "loss": "binary_crossentropy",
        "max_epochs": 15,
        "selection_metric": "val_loss",
        "classification_threshold": 0.5,
        "require_gpu": True,
        "deterministic_ops": True,
    }.items():
        _exact(f"training.{field}", training.get(field), expected)
    early = _table(training, "early_stopping", "training.early_stopping")
    for field, expected in {
        "monitor": "val_loss",
        "patience": 3,
        "min_delta": 0.001,
        "restore_best_weights": True,
    }.items():
        _exact(f"training.early_stopping.{field}", early.get(field), expected)
    checkpoint = _table(training, "checkpoint", "training.checkpoint")
    for field, expected in {
        "monitor": "val_loss",
        "mode": "min",
        "save_best_only": True,
    }.items():
        _exact(f"training.checkpoint.{field}", checkpoint.get(field), expected)
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        max_epochs=15,
        threshold=0.5,
        require_gpu=True,
        deterministic_ops=True,
        early_stopping=EarlyStoppingConfig("val_loss", 3, 0.001, True),
        checkpoint=CheckpointConfig("val_loss", "min", True),
    )

    augmentation = _table(document, "augmentation", "augmentation")
    _exact("augmentation.enabled", augmentation.get("enabled"), False)
    metrics = _table(document, "metrics", "metrics")
    _exact("metrics.zero_division", metrics.get("zero_division"), 0.0)
    _exact("metrics.positive_class", metrics.get("positive_class"), "muffin")

    baseline = _table(_table(document, "baseline", "baseline"), "validation", "baseline.validation")
    _exact("baseline.validation.strategy", baseline.get("strategy"), "always_chihuahua")
    _exact("baseline.validation.accuracy", baseline.get("accuracy"), 384 / 710)
    _exact("baseline.validation.macro_f1", baseline.get("macro_f1"), (768 / 1094) / 2)

    target = _table(document, "quality_target", "quality_target")
    _exact("quality_target.comparison", target.get("comparison"), "strictly_greater")
    _exact("quality_target.approved", target.get("approved"), True)
    target_accuracy = _value(target, "accuracy", float, "quality_target.accuracy")
    target_macro_f1 = _value(target, "macro_f1", float, "quality_target.macro_f1")
    _exact("quality_target.accuracy", target_accuracy, 384 / 710)
    _exact("quality_target.macro_f1", target_macro_f1, (768 / 1094) / 2)

    smoke = _table(document, "smoke_train", "smoke_train")
    for field, expected in {
        "epochs": 1,
        "samples_per_class_per_split": 8,
        "allow_cpu_by_default": False,
    }.items():
        _exact(f"smoke_train.{field}", smoke.get(field), expected)

    artifacts = _table(document, "artifacts", "artifacts")
    runs_dir = _relative_path(
        _value(artifacts, "runs_dir", str, "artifacts.runs_dir"),
        "artifacts.runs_dir",
    )
    compatibility = _table(document, "compatibility", "compatibility")
    expected_compatibility = {
        "python": "3.12",
        "tensorflow": "2.21.0",
        "keras": "3.15.1",
        "tensorflow_build_cuda": "12.5.1",
        "tensorflow_build_cudnn": "9",
        "official_tested_cuda": "12.5",
        "official_tested_cudnn": "9.3",
    }
    for field, expected in expected_compatibility.items():
        _exact(f"compatibility.{field}", compatibility.get(field), expected)

    return ModelRunConfig(
        source_path=source_path,
        source_bytes=source_bytes,
        schema_version=schema,
        seed=seed,
        dataset=dataset_paths,
        architecture=architecture,
        training=training_config,
        quality_target=QualityTarget(target_accuracy, target_macro_f1, True),
        smoke=SmokeConfig(1, 8, False),
        runs_dir=runs_dir,
        compatibility=CompatibilityConfig(**expected_compatibility),
    )
