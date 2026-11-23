"""Auditable training, checkpoint freezing, and final evaluation lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import subprocess
import uuid
from typing import Any

import numpy as np
import tensorflow as tf

from chihuahua_muffin.data.canonical import canonical_json_bytes, sha256_hex
from chihuahua_muffin.data.inventory import sha256_file

from .architecture import build_model, compile_model
from .config import DEFAULT_MODEL_CONFIG, ModelRunConfig, load_model_config
from .contract import DatasetContract, load_dataset_contract
from .device import DeviceSelection, environment_summary, select_device
from .metrics import (
    PANEL_CATEGORY_ORDER,
    classification_panel_png,
    classification_metrics,
    save_confusion_matrix,
    save_training_curves,
    select_classification_examples,
)
from .pipeline import balanced_smoke_subset, build_dataset


class RunError(RuntimeError):
    """Raised when a run cannot safely advance through its lifecycle."""


class BestEpochTracker(tf.keras.callbacks.Callback):
    """Track the absolute minimum val_loss independently of min_delta."""

    def __init__(self) -> None:
        super().__init__()
        self.best_epoch: int | None = None
        self.best_value = float("inf")

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        value = (logs or {}).get("val_loss")
        if value is not None and np.isfinite(value) and value < self.best_value:
            self.best_value = float(value)
            self.best_epoch = epoch + 1


def configure_reproducibility(seed: int, *, deterministic_ops: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_state(project_root: Path) -> dict[str, object]:
    def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            arguments,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    revision = command(["git", "rev-parse", "HEAD"])
    status = command(["git", "status", "--porcelain"])
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else "unknown",
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_bytes(path, canonical_json_bytes(value))


def _create_run_directory(root: Path, mode: str) -> tuple[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{mode}-{uuid.uuid4().hex[:8]}"
    path = root / run_id
    path.mkdir()
    return run_id, path


def _artifact_entry(path: Path, run_dir: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(run_dir).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _history_payload(history: dict[str, list[float]]) -> dict[str, list[float]]:
    return {
        key: [float(value) for value in values]
        for key, values in sorted(history.items())
    }


def _predict_probabilities(model: tf.keras.Model, dataset: tf.data.Dataset) -> list[float]:
    values = model.predict(dataset, verbose=0)
    return [float(value) for value in np.asarray(values).reshape(-1)]


def _evaluate_loss(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict[str, float]:
    result = model.evaluate(dataset, verbose=0, return_dict=True)
    return {str(key): float(value) for key, value in result.items()}


def _base_manifest(
    *,
    run_id: str,
    mode: str,
    config: ModelRunConfig,
    contract: DatasetContract,
    device: DeviceSelection,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "mode": mode,
        "status": "running",
        "created_at": _now(),
        "updated_at": _now(),
        "git": _git_state(project_root),
        "environment": environment_summary(),
        "device": device.as_dict(),
        "configuration": {
            "path": "config.toml",
            "sha256": sha256_hex(config.source_bytes),
            "seed": config.seed,
        },
        "dataset": {
            "dataset_version_id": contract.dataset_version_id,
            "manifest_sha256": contract.manifest_sha256,
            "manifest_file_sha256": contract.manifest_file_sha256,
            "index_sha256": contract.index_sha256,
            "counts_by_split_and_class": contract.counts_by_split_and_class,
            "assignments_modified": False,
        },
        "model": {
            "architecture": config.architecture.name,
            "input_shape": list(config.architecture.input_shape),
            "trainable_parameters": config.architecture.expected_trainable_parameters,
            "output_semantics": "P(muffin)",
        },
        "training": {},
        "checkpoint": None,
        "threshold": config.training.threshold,
        "metrics": None,
        "artifacts": {},
        "limitations": [
            *contract.limitations,
            "TensorFlow determinism does not imply bit-identical results across hardware.",
        ],
        "evaluation": {"test_consulted": False},
    }


def run_training(
    project_root: str | Path = ".",
    *,
    config_path: str | Path = DEFAULT_MODEL_CONFIG,
    smoke: bool,
    allow_cpu: bool,
) -> Path:
    """Run a smoke or full training while preserving a frozen checkpoint."""

    root = Path(project_root).resolve()
    config_source = Path(config_path)
    if not config_source.is_absolute():
        config_source = root / config_source
    config = load_model_config(config_source)
    configure_reproducibility(
        config.seed, deterministic_ops=config.training.deterministic_ops
    )
    device = select_device(require_gpu=not smoke, allow_cpu=allow_cpu)
    contract = load_dataset_contract(root, config, verify_source_hashes=True)
    mode = "smoke" if smoke else "train"
    run_id, run_dir = _create_run_directory(root / config.runs_dir, mode)
    config_artifact = run_dir / "config.toml"
    with config_artifact.open("xb") as stream:
        stream.write(config.source_bytes)
    manifest = _base_manifest(
        run_id=run_id,
        mode=mode,
        config=config,
        contract=contract,
        device=device,
        project_root=root,
    )
    manifest_path = run_dir / "run_manifest.json"
    try:
        _write_json(manifest_path, manifest)
        train_samples = contract.for_split("train")
        validation_samples = contract.for_split("validation")
        epochs = config.training.max_epochs
        if smoke:
            train_samples = balanced_smoke_subset(
                train_samples,
                samples_per_class=config.smoke.samples_per_class_per_split,
            )
            validation_samples = balanced_smoke_subset(
                validation_samples,
                samples_per_class=config.smoke.samples_per_class_per_split,
            )
            epochs = config.smoke.epochs
        train_bundle = build_dataset(
            train_samples,
            batch_size=config.training.batch_size,
            training=True,
            seed=config.seed,
        )
        validation_bundle = build_dataset(
            validation_samples,
            batch_size=config.training.batch_size,
            training=False,
            seed=config.seed,
        )
        strategy = tf.distribute.OneDeviceStrategy(device.selected)
        with strategy.scope():
            model = build_model(config)
            compile_model(model, config)
        summary_lines: list[str] = []
        model.summary(print_fn=summary_lines.append)
        summary_path = run_dir / "model_summary.txt"
        _atomic_bytes(
            summary_path,
            ("\n".join(summary_lines) + "\n").encode("utf-8"),
        )
        checkpoint_path = run_dir / "model.keras"
        tracker = BestEpochTracker()
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=config.training.checkpoint.monitor,
            mode=config.training.checkpoint.mode,
            save_best_only=config.training.checkpoint.save_best_only,
            verbose=0,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=config.training.early_stopping.monitor,
            patience=config.training.early_stopping.patience,
            min_delta=config.training.early_stopping.min_delta,
            restore_best_weights=config.training.early_stopping.restore_best_weights,
            verbose=0,
        )
        history_object = model.fit(
            train_bundle.dataset,
            validation_data=validation_bundle.dataset,
            epochs=epochs,
            callbacks=[checkpoint, early_stopping, tracker],
            shuffle=False,
            verbose=2,
        )
        if tracker.best_epoch is None or not checkpoint_path.is_file():
            raise RunError("training did not produce a valid best checkpoint")
        history = _history_payload(history_object.history)
        selected = tf.keras.models.load_model(checkpoint_path)
        validation_evaluation = _evaluate_loss(selected, validation_bundle.dataset)
        validation_probabilities = _predict_probabilities(
            selected, validation_bundle.dataset
        )
        validation_labels = [sample.class_id for sample in validation_samples]
        validation_metrics = classification_metrics(
            validation_labels,
            validation_probabilities,
            threshold=config.training.threshold,
        )
        validation_metrics["loss"] = validation_evaluation["loss"]
        quality_met = (
            float(validation_metrics["accuracy"])
            > config.quality_target.validation_accuracy
            and float(validation_metrics["macro_f1"])
            > config.quality_target.validation_macro_f1
        )
        metrics_payload: dict[str, Any] = {
            "schema_version": "1.0",
            "mode": mode,
            "test_consulted": False,
            "history": history,
            "validation": validation_metrics,
            "baseline": {
                "strategy": "always_chihuahua",
                "accuracy": config.quality_target.validation_accuracy,
                "macro_f1": config.quality_target.validation_macro_f1,
            },
            "quality_target": {
                "comparison": "strictly_greater",
                "met": quality_met if not smoke else None,
                "applicable": not smoke,
            },
        }
        metrics_path = run_dir / "metrics.json"
        _write_json(metrics_path, metrics_payload)
        curves_path = run_dir / "training_curves.png"
        save_training_curves(history, curves_path)
        stop_reason = (
            "early_stopping"
            if early_stopping.stopped_epoch > 0
            else "requested_epochs_completed"
        )
        manifest["status"] = "smoke_completed" if smoke else "trained_frozen"
        manifest["updated_at"] = _now()
        manifest["training"] = {
            "requested_epochs": epochs,
            "executed_epochs": len(history["loss"]),
            "stop_reason": stop_reason,
            "early_stopping": {
                "monitor": "val_loss",
                "patience": 3,
                "min_delta": 0.001,
                "restore_best_weights": True,
            },
            "batch_size": config.training.batch_size,
            "optimizer": "adam",
            "learning_rate": config.training.learning_rate,
            "loss": "binary_crossentropy",
            "train_sample_count": len(train_samples),
            "validation_sample_count": len(validation_samples),
            "train_sample_ids_sha256": sha256_hex(
                "\n".join(sorted(train_bundle.sample_ids)).encode("utf-8")
            ),
            "validation_sample_ids_sha256": sha256_hex(
                "\n".join(sorted(validation_bundle.sample_ids)).encode("utf-8")
            ),
            "smoke_cpu_authorized": bool(smoke and device.selected == "/CPU:0"),
            "history": history,
        }
        manifest["checkpoint"] = {
            **_artifact_entry(checkpoint_path, run_dir),
            "selection": "absolute_minimum_val_loss",
            "epoch": tracker.best_epoch,
            "val_loss": tracker.best_value,
            "independent_of_early_stopping_min_delta": True,
        }
        manifest["metrics"] = {
            "validation": validation_metrics,
            "quality_target_met": quality_met if not smoke else None,
            "quality_claim_applicable": not smoke,
        }
        manifest["artifacts"] = {
            "config": _artifact_entry(config_artifact, run_dir),
            "model": _artifact_entry(checkpoint_path, run_dir),
            "model_summary": _artifact_entry(summary_path, run_dir),
            "metrics": _artifact_entry(metrics_path, run_dir),
            "training_curves": _artifact_entry(curves_path, run_dir),
        }
        _write_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["updated_at"] = _now()
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        try:
            _write_json(manifest_path, manifest)
        except Exception as manifest_error:
            raise RunError(
                f"run failed ({error}); failure manifest also failed: {manifest_error}"
            ) from error
        raise
    return run_dir


def _load_canonical_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RunError(f"cannot read run manifest: {error}") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise RunError("run manifest is not canonical JSON")
    return value


def evaluate_run(
    run_id: str,
    project_root: str | Path = ".",
) -> Path:
    """Consult test exactly after verifying the frozen training state."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise RunError("run_id is unsafe")
    root = Path(project_root).resolve()
    run_dir = root / "artifacts/runs" / run_id
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RunError(f"run does not exist: {run_id}")
    manifest_path = run_dir / "run_manifest.json"
    manifest = _load_canonical_manifest(manifest_path)
    if manifest.get("run_id") != run_id or manifest.get("status") != "trained_frozen":
        raise RunError("only a trained_frozen run can be evaluated")
    if manifest.get("evaluation") != {"test_consulted": False}:
        raise RunError("run evaluation state is not pristine")
    config_path = run_dir / "config.toml"
    config = load_model_config(config_path)
    config_entry = manifest.get("artifacts", {}).get("config")
    model_entry = manifest.get("artifacts", {}).get("model")
    metrics_entry = manifest.get("artifacts", {}).get("metrics")
    if type(config_entry) is not dict or config_entry.get("sha256") != sha256_file(config_path):
        raise RunError("frozen configuration digest is invalid")
    model_path = run_dir / "model.keras"
    if type(model_entry) is not dict or model_entry.get("sha256") != sha256_file(model_path):
        raise RunError("frozen checkpoint digest is invalid")
    metrics_path = run_dir / "metrics.json"
    if type(metrics_entry) is not dict or metrics_entry.get("sha256") != sha256_file(metrics_path):
        raise RunError("pre-evaluation metrics digest is invalid")
    contract = load_dataset_contract(root, config, verify_source_hashes=True)
    if manifest.get("dataset", {}).get("manifest_sha256") != contract.manifest_sha256:
        raise RunError("dataset manifest changed between training and evaluation")
    configure_reproducibility(config.seed, deterministic_ops=True)
    device = select_device(require_gpu=False, allow_cpu=True)
    test_samples = contract.for_split("test")
    test_bundle = build_dataset(
        test_samples,
        batch_size=config.training.batch_size,
        training=False,
        seed=config.seed,
    )
    try:
        model = tf.keras.models.load_model(model_path)
        evaluation = _evaluate_loss(model, test_bundle.dataset)
        probabilities = _predict_probabilities(model, test_bundle.dataset)
        test_metrics = classification_metrics(
            [sample.class_id for sample in test_samples],
            probabilities,
            threshold=config.training.threshold,
        )
        test_metrics["loss"] = evaluation["loss"]
        previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        previous_digest = sha256_file(metrics_path)
        previous_metrics["test_consulted"] = True
        previous_metrics["test"] = test_metrics
        _write_json(metrics_path, previous_metrics)
        confusion_path = run_dir / "confusion_matrix.png"
        save_confusion_matrix(test_metrics["confusion_matrix"], confusion_path)  # type: ignore[arg-type]
        manifest["status"] = "evaluated"
        manifest["updated_at"] = _now()
        manifest["evaluation"] = {
            "test_consulted": True,
            "evaluated_at": _now(),
            "checkpoint_sha256": sha256_file(model_path),
            "threshold": config.training.threshold,
            "test_sample_count": len(test_samples),
            "test_sample_ids_sha256": sha256_hex(
                "\n".join(sorted(test_bundle.sample_ids)).encode("utf-8")
            ),
            "device": device.as_dict(),
            "metrics": test_metrics,
            "used_for_selection_or_tuning": False,
        }
        manifest["artifacts"]["pre_evaluation_metrics_sha256"] = previous_digest
        manifest["artifacts"]["metrics"] = _artifact_entry(metrics_path, run_dir)
        manifest["artifacts"]["confusion_matrix"] = _artifact_entry(
            confusion_path, run_dir
        )
        _write_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "evaluation_failed"
        manifest["updated_at"] = _now()
        manifest["evaluation"] = {
            "test_consulted": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "used_for_selection_or_tuning": False,
        }
        _write_json(manifest_path, manifest)
        raise
    return run_dir


def create_evaluation_panel(
    run_id: str,
    project_root: str | Path = ".",
    *,
    examples_per_category: int = 3,
) -> Path:
    """Create an audited VP/VN/FP/FN panel for an already evaluated run."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise RunError("run_id is unsafe")
    root = Path(project_root).resolve()
    run_dir = root / "artifacts/runs" / run_id
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RunError(f"run does not exist: {run_id}")
    manifest_path = run_dir / "run_manifest.json"
    manifest = _load_canonical_manifest(manifest_path)
    evaluation = manifest.get("evaluation")
    if (
        manifest.get("run_id") != run_id
        or manifest.get("status") != "evaluated"
        or type(evaluation) is not dict
        or evaluation.get("test_consulted") is not True
    ):
        raise RunError("only an evaluated run can produce the evaluation panel")

    panel_path = run_dir / "evaluation_panel.png"
    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict:
        raise RunError("run artifact registry is invalid")
    existing_entry = artifacts.get("evaluation_panel")
    replacing_registered_panel = False
    if existing_entry is not None:
        if (
            type(existing_entry) is not dict
            or existing_entry.get("path") != panel_path.name
            or not panel_path.is_file()
            or existing_entry.get("sha256") != sha256_file(panel_path)
        ):
            raise RunError("registered evaluation panel is invalid")
        existing_metadata = evaluation.get("example_panel")
        if (
            type(existing_metadata) is dict
            and existing_metadata.get("schema_version") == "1.0"
        ):
            return panel_path
        replacing_registered_panel = True
    if not replacing_registered_panel and (panel_path.exists() or panel_path.is_symlink()):
        raise RunError("unregistered evaluation panel already exists")

    config_path = run_dir / "config.toml"
    model_path = run_dir / "model.keras"
    metrics_path = run_dir / "metrics.json"
    for key, path, description in (
        ("config", config_path, "configuration"),
        ("model", model_path, "checkpoint"),
        ("metrics", metrics_path, "evaluation metrics"),
    ):
        entry = artifacts.get(key)
        if type(entry) is not dict or entry.get("sha256") != sha256_file(path):
            raise RunError(f"frozen {description} digest is invalid")

    config = load_model_config(config_path)
    contract = load_dataset_contract(root, config, verify_source_hashes=True)
    if manifest.get("dataset", {}).get("manifest_sha256") != contract.manifest_sha256:
        raise RunError("dataset manifest changed after evaluation")
    test_samples = contract.for_split("test")
    test_bundle = build_dataset(
        test_samples,
        batch_size=config.training.batch_size,
        training=False,
        seed=config.seed,
    )
    sample_ids_sha256 = sha256_hex(
        "\n".join(sorted(test_bundle.sample_ids)).encode("utf-8")
    )
    if (
        evaluation.get("test_sample_count") != len(test_samples)
        or evaluation.get("test_sample_ids_sha256") != sample_ids_sha256
        or evaluation.get("threshold") != config.training.threshold
    ):
        raise RunError("evaluated test identity or threshold changed")

    configure_reproducibility(config.seed, deterministic_ops=True)
    generation_device = select_device(require_gpu=False, allow_cpu=True)
    model = tf.keras.models.load_model(model_path)
    probabilities = _predict_probabilities(model, test_bundle.dataset)
    recomputed = classification_metrics(
        [sample.class_id for sample in test_samples],
        probabilities,
        threshold=config.training.threshold,
    )
    recorded_metrics = evaluation.get("metrics")
    if (
        type(recorded_metrics) is not dict
        or recomputed["confusion_matrix"] != recorded_metrics.get("confusion_matrix")
        or recomputed["sample_count"] != recorded_metrics.get("sample_count")
    ):
        raise RunError("recomputed classifications differ from the final evaluation")

    try:
        examples = select_classification_examples(
            test_samples,
            probabilities,
            threshold=config.training.threshold,
            examples_per_category=examples_per_category,
        )
    except ValueError as error:
        raise RunError(str(error)) from error
    _atomic_bytes(
        panel_path,
        classification_panel_png(
            examples, examples_per_category=examples_per_category
        ),
    )
    panel_metadata = {
        "schema_version": "1.0",
        "generated_at": _now(),
        "source_split": "test",
        "selection_policy": "sample_id_ascending_v1",
        "examples_per_category": examples_per_category,
        "input_representation": "spec_01_preprocessed_128x128_rgb",
        "probability_source": "recomputed_from_frozen_checkpoint_for_visualization",
        "classification_consistency_check": "confusion_matrix_matches_final_evaluation",
        "generation_device": generation_device.as_dict(),
        "used_for_selection_or_tuning": False,
        "categories": {
            category: [
                example.manifest_payload()
                for example in examples
                if example.category == category
            ]
            for category in PANEL_CATEGORY_ORDER
        },
    }
    evaluation["example_panel"] = panel_metadata
    artifacts["evaluation_panel"] = _artifact_entry(panel_path, run_dir)
    manifest["updated_at"] = _now()
    try:
        _write_json(manifest_path, manifest)
    except Exception:
        if not replacing_registered_panel:
            panel_path.unlink(missing_ok=True)
        raise
    return panel_path
