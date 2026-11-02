"""TensorFlow device diagnostics and explicit execution gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import subprocess

import keras
import tensorflow as tf


class DeviceUnavailableError(RuntimeError):
    """Raised when the requested execution device cannot be proven usable."""


@dataclass(frozen=True, slots=True)
class DeviceSelection:
    requested: str
    selected: str
    probe_tensor_device: str
    tensorflow_version: str
    keras_version: str
    tensorflow_built_with_cuda: bool
    tensorflow_build_info: dict[str, object]
    physical_devices: tuple[str, ...]
    gpu_devices: tuple[str, ...]
    nvidia_smi_status: str
    nvidia_smi_output: str

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["physical_devices"] = list(self.physical_devices)
        value["gpu_devices"] = list(self.gpu_devices)
        return value


def _nvidia_smi() -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return "unavailable", str(error)
    output = (result.stdout + result.stderr).strip()
    return ("success" if result.returncode == 0 else "failed"), output


def select_device(*, require_gpu: bool, allow_cpu: bool) -> DeviceSelection:
    """Select and execute a real probe without silent CPU fallback."""

    physical = tuple(
        f"{item.name}:{item.device_type}" for item in tf.config.list_physical_devices()
    )
    gpus = tuple(item.name for item in tf.config.list_physical_devices("GPU"))
    smi_status, smi_output = _nvidia_smi()
    if gpus:
        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
        selected = "/GPU:0"
        requested = "gpu"
    elif require_gpu or not allow_cpu:
        raise DeviceUnavailableError(
            "TensorFlow did not expose a usable GPU; full training has no CPU "
            f"fallback. nvidia-smi={smi_status}: {smi_output or 'no output'}"
        )
    else:
        selected = "/CPU:0"
        requested = "cpu_explicitly_authorized"

    try:
        with tf.device(selected):
            probe = tf.linalg.matmul(
                tf.constant([[1.0, 2.0]]), tf.constant([[1.0], [1.0]])
            )
        _ = probe.numpy()
    except Exception as error:
        raise DeviceUnavailableError(
            f"TensorFlow device probe failed on {selected}: {error}"
        ) from error
    actual = probe.device
    expected_marker = "GPU" if selected.startswith("/GPU") else "CPU"
    if expected_marker not in actual.upper():
        raise DeviceUnavailableError(
            f"device probe ran on {actual!r}, not requested {selected}"
        )
    build_info = {
        str(key): value for key, value in tf.sysconfig.get_build_info().items()
    }
    return DeviceSelection(
        requested=requested,
        selected=selected,
        probe_tensor_device=actual,
        tensorflow_version=tf.__version__,
        keras_version=keras.__version__,
        tensorflow_built_with_cuda=tf.test.is_built_with_cuda(),
        tensorflow_build_info=build_info,
        physical_devices=physical,
        gpu_devices=gpus,
        nvidia_smi_status=smi_status,
        nvidia_smi_output=smi_output,
    )


def environment_summary() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "tensorflow_built_with_cuda": tf.test.is_built_with_cuda(),
        "tensorflow_build_info": {
            str(key): value for key, value in tf.sysconfig.get_build_info().items()
        },
    }
