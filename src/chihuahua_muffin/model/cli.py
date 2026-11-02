"""Command-line harness for SPEC 02."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_MODEL_CONFIG, load_model_config
from .device import DeviceUnavailableError, environment_summary, select_device
from .runtime import RunError, evaluate_run, run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPEC 02 training harness")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=str(DEFAULT_MODEL_CONFIG))
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="run a minimal end-to-end training")
    smoke.add_argument("--allow-cpu", action="store_true")
    commands.add_parser("train", help="run full GPU-only training")
    evaluate = commands.add_parser("evaluate", help="evaluate a frozen run on test")
    evaluate.add_argument("--run-id", required=True)
    commands.add_parser("diagnose", help="print TensorFlow device diagnostics")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = Path(arguments.project_root).resolve()
    config_path = Path(arguments.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        if arguments.command == "smoke":
            path = run_training(
                root,
                config_path=config_path,
                smoke=True,
                allow_cpu=arguments.allow_cpu,
            )
            print(f"smoke_run={path.name}")
            print(f"run_directory={path}")
        elif arguments.command == "train":
            path = run_training(
                root,
                config_path=config_path,
                smoke=False,
                allow_cpu=False,
            )
            print(f"training_run={path.name}")
            print(f"run_directory={path}")
        elif arguments.command == "evaluate":
            path = evaluate_run(arguments.run_id, root)
            print(f"evaluated_run={path.name}")
            print(f"run_directory={path}")
        else:
            config = load_model_config(config_path)
            try:
                device = select_device(require_gpu=True, allow_cpu=False)
                device_payload = device.as_dict()
                status = "gpu_usable"
            except DeviceUnavailableError as error:
                device_payload = {"error": str(error)}
                status = "gpu_unavailable"
            print(
                json.dumps(
                    {
                        "status": status,
                        "configured_compatibility": {
                            field: getattr(config.compatibility, field)
                            for field in config.compatibility.__dataclass_fields__
                        },
                        "environment": environment_summary(),
                        "device": device_payload,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if status == "gpu_usable" else 2
    except (ValueError, RuntimeError, RunError, DeviceUnavailableError) as error:
        parser = _parser()
        parser.exit(2, f"ERRO: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
