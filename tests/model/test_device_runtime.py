from pathlib import Path

import pytest

from chihuahua_muffin.model import device
from chihuahua_muffin.model.device import DeviceUnavailableError
from chihuahua_muffin.model.runtime import (
    BestEpochTracker,
    RunError,
    create_evaluation_panel,
    evaluate_run,
)


def test_gpu_requirement_rejects_absence_without_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        device.tf.config,
        "list_physical_devices",
        lambda device_type=None: [] if device_type == "GPU" else [],
    )
    monkeypatch.setattr(device, "_nvidia_smi", lambda: ("failed", "driver unavailable"))

    with pytest.raises(DeviceUnavailableError, match="no CPU fallback"):
        device.select_device(require_gpu=True, allow_cpu=False)


def test_best_epoch_tracker_uses_absolute_minimum_not_min_delta() -> None:
    tracker = BestEpochTracker()
    tracker.on_epoch_end(0, {"val_loss": 1.0})
    tracker.on_epoch_end(1, {"val_loss": 0.9995})

    assert tracker.best_epoch == 2
    assert tracker.best_value == 0.9995


def test_evaluation_rejects_unknown_run_before_consulting_test(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="does not exist"):
        evaluate_run("missing-run", tmp_path)


def test_evaluation_rejects_unsafe_run_id(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="unsafe"):
        evaluate_run("../escape", tmp_path)


def test_evaluation_panel_rejects_unknown_run(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="does not exist"):
        create_evaluation_panel("missing-run", tmp_path)


def test_evaluation_panel_rejects_unsafe_run_id(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="unsafe"):
        create_evaluation_panel("../escape", tmp_path)
