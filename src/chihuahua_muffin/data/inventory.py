"""Deterministic image inventory and initial quality report for SPEC 01."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
import warnings

from PIL import Image, UnidentifiedImageError

from .canonical import canonical_json_bytes, make_sample_id


CLASS_IDS = {"chihuahua": 0, "muffin": 1}
PARTITION_NAMES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}
EXPECTED_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
EXPECTED_FORMAT_EXTENSIONS = {
    "JPEG": {".jpeg", ".jpg"},
    "PNG": {".png"},
}
GRAYSCALE_MODES = {"1", "L"}


class InventoryError(ValueError):
    """Raised when an inventory cannot be created safely."""


class AuditStatus(StrEnum):
    """Reconciliation state assigned to every discovered file."""

    ACCEPTED = "accepted"
    EXCLUDED = "excluded"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class InventoryRecord:
    """Deterministic observations and decisions for one source file."""

    sample_id: str
    relative_path: str
    sha256: str
    byte_size: int
    class_name: str | None
    class_id: int | None
    source_partition: str | None
    source_partition_name: str | None
    image_format: str | None
    width: int | None
    height: int | None
    mode: str | None
    channels: int | None
    is_animated: bool | None
    exif_orientation: int | None
    exif_transpose_required: bool
    color_conversion: str | None
    status: AuditStatus
    reasons: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        """Return only canonical-JSON-compatible values."""

        return {
            "sample_id": self.sample_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "source_partition": self.source_partition,
            "source_partition_name": self.source_partition_name,
            "format": self.image_format,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "channels": self.channels,
            "is_animated": self.is_animated,
            "exif_orientation": self.exif_orientation,
            "exif_transpose_required": self.exif_transpose_required,
            "color_conversion": self.color_conversion,
            "status": self.status.value,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ExactDuplicateCandidate:
    """Same-byte candidates reported without making a deduplication decision."""

    sha256: str
    sample_ids: tuple[str, ...]
    relative_paths: tuple[str, ...]
    class_names: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "sample_ids": list(self.sample_ids),
            "relative_paths": list(self.relative_paths),
            "class_names": list(self.class_names),
        }


@dataclass(frozen=True, slots=True)
class InventoryResult:
    """Complete, reconciled initial audit over one immutable source tree."""

    dataset_version_id: str
    records: tuple[InventoryRecord, ...]
    exact_duplicate_candidates: tuple[ExactDuplicateCandidate, ...]

    def inventory_payload(self) -> dict[str, object]:
        return {
            "artifact_type": "initial_inventory",
            "artifact_version": "1.0",
            "dataset_version_id": self.dataset_version_id,
            "records": [record.as_json() for record in self.records],
        }

    def quality_payload(self) -> dict[str, object]:
        status_counts = Counter(record.status.value for record in self.records)
        reason_counts = Counter(
            reason for record in self.records for reason in record.reasons
        )
        class_counts = Counter(
            record.class_name or "unidentified" for record in self.records
        )
        partition_counts = Counter(
            record.source_partition or "none" for record in self.records
        )
        format_counts = Counter(
            record.image_format or "unidentified" for record in self.records
        )
        mode_counts = Counter(record.mode or "unidentified" for record in self.records)
        return {
            "artifact_type": "initial_quality_report",
            "artifact_version": "1.0",
            "dataset_version_id": self.dataset_version_id,
            "total_discovered_files": len(self.records),
            "reconciled_files": sum(status_counts.values()),
            "counts_by_status": dict(sorted(status_counts.items())),
            "counts_by_reason": dict(sorted(reason_counts.items())),
            "counts_by_class": dict(sorted(class_counts.items())),
            "counts_by_source_partition": dict(sorted(partition_counts.items())),
            "counts_by_format": dict(sorted(format_counts.items())),
            "counts_by_mode": dict(sorted(mode_counts.items())),
            "exact_duplicate_candidate_group_count": len(
                self.exact_duplicate_candidates
            ),
            "exact_duplicate_candidates": [
                candidate.as_json() for candidate in self.exact_duplicate_candidates
            ],
            "originals_modified": False,
            "limitations": [
                "This is an initial inventory, not a released dataset manifest.",
                "Exact duplicate hashes are candidates only; no occurrence was removed.",
                "Perceptual duplicate detection and split assignment are not implemented.",
            ],
        }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file without loading it completely into memory."""

    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_dataset(root: str | Path, *, dataset_version_id: str) -> InventoryResult:
    """Inventory every discovered file in stable relative-path order."""

    source_root = Path(root)
    if source_root.is_symlink():
        raise InventoryError("dataset root must not be a symbolic link")
    if not source_root.exists() or not source_root.is_dir():
        raise InventoryError("dataset root must be an existing directory")

    discovered = sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_file() or path.is_symlink()
        ),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    records = tuple(
        _inspect_file(source_root, path, dataset_version_id=dataset_version_id)
        for path in discovered
    )
    return InventoryResult(
        dataset_version_id=dataset_version_id,
        records=records,
        exact_duplicate_candidates=_find_exact_duplicate_candidates(records),
    )


def write_initial_audit(result: InventoryResult, destination: str | Path) -> tuple[Path, Path]:
    """Write local non-final inventory and QC artifacts without overwriting."""

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise InventoryError(f"audit destination already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    inventory_path = target / "inventory.json"
    quality_path = target / "quality_report.json"
    inventory_path.write_bytes(canonical_json_bytes(result.inventory_payload()))
    quality_path.write_bytes(canonical_json_bytes(result.quality_payload()))
    return inventory_path, quality_path


def load_initial_inventory(
    path: str | Path,
    *,
    expected_dataset_version_id: str | None = None,
) -> InventoryResult:
    """Load and strictly validate an Increment 4 initial inventory artifact."""

    inventory_path = Path(path)
    try:
        document = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InventoryError(f"cannot read initial inventory: {error}") from error
    if type(document) is not dict or set(document) != {
        "artifact_type",
        "artifact_version",
        "dataset_version_id",
        "records",
    }:
        raise InventoryError("initial inventory artifact fields do not match schema")
    if document["artifact_type"] != "initial_inventory":
        raise InventoryError("unexpected initial inventory artifact_type")
    if document["artifact_version"] != "1.0":
        raise InventoryError("unsupported initial inventory artifact_version")
    dataset_version_id = _required_text(
        document["dataset_version_id"], "dataset_version_id"
    )
    if (
        expected_dataset_version_id is not None
        and dataset_version_id != expected_dataset_version_id
    ):
        raise InventoryError("initial inventory belongs to a different version")
    if type(document["records"]) is not list:
        raise InventoryError("initial inventory records must be a JSON list")

    records = tuple(
        _record_from_json(value, dataset_version_id=dataset_version_id)
        for value in document["records"]
    )
    relative_paths = [record.relative_path for record in records]
    sample_ids = [record.sample_id for record in records]
    if relative_paths != sorted(relative_paths):
        raise InventoryError("initial inventory records are not in canonical path order")
    if len(relative_paths) != len(set(relative_paths)):
        raise InventoryError("initial inventory contains duplicate relative paths")
    if len(sample_ids) != len(set(sample_ids)):
        raise InventoryError("initial inventory contains duplicate sample IDs")
    return InventoryResult(
        dataset_version_id=dataset_version_id,
        records=records,
        exact_duplicate_candidates=_find_exact_duplicate_candidates(records),
    )


def _inspect_file(
    root: Path,
    path: Path,
    *,
    dataset_version_id: str,
) -> InventoryRecord:
    relative_path = path.relative_to(root).as_posix()
    sample_id = make_sample_id(dataset_version_id, relative_path)
    class_name, class_reason = _infer_class(relative_path)
    partition, partition_name, partition_reason = _infer_partition(relative_path)
    reasons = [reason for reason in (class_reason, partition_reason) if reason]

    if path.is_symlink():
        return _record(
            sample_id=sample_id,
            relative_path=relative_path,
            status=AuditStatus.REVIEW,
            reasons=tuple([*reasons, "symbolic_link_not_inspected"]),
            class_name=class_name,
            source_partition=partition,
            source_partition_name=partition_name,
        )

    try:
        file_hash = sha256_file(path)
        byte_size = path.stat().st_size
    except OSError:
        return _record(
            sample_id=sample_id,
            relative_path=relative_path,
            status=AuditStatus.EXCLUDED,
            reasons=tuple([*reasons, "file_read_error"]),
            class_name=class_name,
            source_partition=partition,
            source_partition_name=partition_name,
        )

    extension = path.suffix.casefold()
    try:
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image_format = image.format
                width, height = image.size
                mode = image.mode
                channels = len(image.getbands())
                is_animated = bool(
                    getattr(image, "is_animated", False)
                    or getattr(image, "n_frames", 1) > 1
                )
                exif_orientation = image.getexif().get(274)
                if not is_animated:
                    image.load()
            if captured_warnings:
                reasons.append("decompression_bomb_warning")
    except Image.DecompressionBombError:
        return _record(
            sample_id=sample_id,
            relative_path=relative_path,
            sha256=file_hash,
            byte_size=byte_size,
            status=AuditStatus.EXCLUDED,
            reasons=tuple([*reasons, "decompression_bomb_error"]),
            class_name=class_name,
            source_partition=partition,
            source_partition_name=partition_name,
        )
    except (OSError, SyntaxError, UnidentifiedImageError, ValueError):
        reason = (
            "invalid_or_unreadable_image"
            if extension in EXPECTED_IMAGE_EXTENSIONS
            else "non_image_file"
        )
        return _record(
            sample_id=sample_id,
            relative_path=relative_path,
            sha256=file_hash,
            byte_size=byte_size,
            status=AuditStatus.EXCLUDED,
            reasons=tuple([*reasons, reason]),
            class_name=class_name,
            source_partition=partition,
            source_partition_name=partition_name,
        )

    if is_animated:
        reasons.append("animated_image")
        status = AuditStatus.EXCLUDED
    else:
        status = AuditStatus.ACCEPTED

    if image_format not in EXPECTED_FORMAT_EXTENSIONS:
        reasons.append("unexpected_image_format")
    elif extension not in EXPECTED_FORMAT_EXTENSIONS[image_format]:
        reasons.append("format_extension_mismatch")

    color_conversion: str | None
    if mode == "RGB":
        color_conversion = "preserve_rgb"
    elif mode == "RGBA":
        color_conversion = "rgba_on_white"
    elif mode in GRAYSCALE_MODES:
        color_conversion = "grayscale_replicate"
    else:
        color_conversion = None
        reasons.append("unsupported_color_mode")

    if exif_orientation is not None and (
        type(exif_orientation) is not int or exif_orientation not in range(1, 9)
    ):
        reasons.append("invalid_exif_orientation")
    if status is not AuditStatus.EXCLUDED and reasons:
        status = AuditStatus.REVIEW

    return _record(
        sample_id=sample_id,
        relative_path=relative_path,
        sha256=file_hash,
        byte_size=byte_size,
        class_name=class_name,
        source_partition=partition,
        source_partition_name=partition_name,
        image_format=image_format,
        width=width,
        height=height,
        mode=mode,
        channels=channels,
        is_animated=is_animated,
        exif_orientation=(
            exif_orientation if type(exif_orientation) is int else None
        ),
        exif_transpose_required=(
            type(exif_orientation) is int and exif_orientation in range(2, 9)
        ),
        color_conversion=color_conversion,
        status=status,
        reasons=tuple(sorted(set(reasons))),
    )


def _record(
    *,
    sample_id: str,
    relative_path: str,
    status: AuditStatus,
    reasons: tuple[str, ...],
    sha256: str = "",
    byte_size: int = 0,
    class_name: str | None = None,
    source_partition: str | None = None,
    source_partition_name: str | None = None,
    image_format: str | None = None,
    width: int | None = None,
    height: int | None = None,
    mode: str | None = None,
    channels: int | None = None,
    is_animated: bool | None = None,
    exif_orientation: int | None = None,
    exif_transpose_required: bool = False,
    color_conversion: str | None = None,
) -> InventoryRecord:
    return InventoryRecord(
        sample_id=sample_id,
        relative_path=relative_path,
        sha256=sha256,
        byte_size=byte_size,
        class_name=class_name,
        class_id=CLASS_IDS.get(class_name) if class_name else None,
        source_partition=source_partition,
        source_partition_name=source_partition_name,
        image_format=image_format,
        width=width,
        height=height,
        mode=mode,
        channels=channels,
        is_animated=is_animated,
        exif_orientation=exif_orientation,
        exif_transpose_required=exif_transpose_required,
        color_conversion=color_conversion,
        status=status,
        reasons=tuple(sorted(set(reasons))),
    )


def _infer_class(relative_path: str) -> tuple[str | None, str | None]:
    matches = {
        part.casefold()
        for part in Path(relative_path).parts[:-1]
        if part.casefold() in CLASS_IDS
    }
    if len(matches) == 1:
        return matches.pop(), None
    if not matches:
        return None, "class_not_identified"
    return None, "ambiguous_class_path"


def _infer_partition(
    relative_path: str,
) -> tuple[str | None, str | None, str | None]:
    matches = [
        (part, PARTITION_NAMES[part.casefold()])
        for part in Path(relative_path).parts[:-1]
        if part.casefold() in PARTITION_NAMES
    ]
    canonical = {value for _, value in matches}
    if len(canonical) == 1:
        original = next(part for part, value in matches if value in canonical)
        return canonical.pop(), original, None
    if not canonical:
        return None, None, None
    return None, None, "ambiguous_source_partition"


def _find_exact_duplicate_candidates(
    records: tuple[InventoryRecord, ...],
) -> tuple[ExactDuplicateCandidate, ...]:
    by_hash: dict[str, list[InventoryRecord]] = defaultdict(list)
    for record in records:
        if record.sha256 and record.image_format is not None:
            by_hash[record.sha256].append(record)
    candidates: list[ExactDuplicateCandidate] = []
    for file_hash, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda record: record.sample_id)
        candidates.append(
            ExactDuplicateCandidate(
                sha256=file_hash,
                sample_ids=tuple(record.sample_id for record in ordered),
                relative_paths=tuple(record.relative_path for record in ordered),
                class_names=tuple(
                    sorted(
                        {
                            record.class_name or "unidentified"
                            for record in ordered
                        }
                    )
                ),
            )
        )
    return tuple(candidates)


_INVENTORY_RECORD_FIELDS = {
    "sample_id",
    "relative_path",
    "sha256",
    "byte_size",
    "class_name",
    "class_id",
    "source_partition",
    "source_partition_name",
    "format",
    "width",
    "height",
    "mode",
    "channels",
    "is_animated",
    "exif_orientation",
    "exif_transpose_required",
    "color_conversion",
    "status",
    "reasons",
}


def _record_from_json(
    value: object,
    *,
    dataset_version_id: str,
) -> InventoryRecord:
    if type(value) is not dict or set(value) != _INVENTORY_RECORD_FIELDS:
        raise InventoryError("initial inventory record fields do not match schema")
    sample_id = _required_text(value["sample_id"], "sample_id")
    relative_path = _required_text(value["relative_path"], "relative_path")
    if sample_id != make_sample_id(dataset_version_id, relative_path):
        raise InventoryError("initial inventory sample_id does not match its path")
    file_hash = _optional_text(value["sha256"], "sha256", empty_allowed=True)
    if file_hash and (
        len(file_hash) != 64
        or any(character not in "0123456789abcdef" for character in file_hash)
    ):
        raise InventoryError("initial inventory SHA-256 is invalid")
    byte_size = _required_integer(value["byte_size"], "byte_size", minimum=0)
    class_name = _optional_text(value["class_name"], "class_name")
    class_id = _optional_integer(value["class_id"], "class_id", minimum=0)
    if class_name is None:
        if class_id is not None:
            raise InventoryError("class_id must be null when class_name is null")
    elif class_name not in CLASS_IDS or class_id != CLASS_IDS[class_name]:
        raise InventoryError("class_name and class_id are inconsistent")
    inferred_class, _ = _infer_class(relative_path)
    if class_name != inferred_class:
        raise InventoryError("class_name is inconsistent with relative_path")
    source_partition = _optional_text(
        value["source_partition"], "source_partition"
    )
    if source_partition not in {None, "train", "validation", "test"}:
        raise InventoryError("source_partition is unknown")
    source_partition_name = _optional_text(
        value["source_partition_name"], "source_partition_name"
    )
    inferred_partition, inferred_partition_name, _ = _infer_partition(relative_path)
    if (
        source_partition != inferred_partition
        or source_partition_name != inferred_partition_name
    ):
        raise InventoryError("source partition is inconsistent with relative_path")
    image_format = _optional_text(value["format"], "format")
    width = _optional_integer(value["width"], "width", minimum=1)
    height = _optional_integer(value["height"], "height", minimum=1)
    mode = _optional_text(value["mode"], "mode")
    channels = _optional_integer(value["channels"], "channels", minimum=1)
    is_animated = _optional_boolean(value["is_animated"], "is_animated")
    exif_orientation = _optional_integer(
        value["exif_orientation"], "exif_orientation", minimum=1
    )
    if exif_orientation is not None and exif_orientation > 8:
        raise InventoryError("exif_orientation must be between 1 and 8")
    exif_transpose_required = _required_boolean(
        value["exif_transpose_required"], "exif_transpose_required"
    )
    color_conversion = _optional_text(
        value["color_conversion"], "color_conversion"
    )
    try:
        status = AuditStatus(value["status"])
    except (TypeError, ValueError) as error:
        raise InventoryError("initial inventory status is unknown") from error
    reasons_value = value["reasons"]
    if type(reasons_value) is not list or any(
        type(reason) is not str or not reason for reason in reasons_value
    ):
        raise InventoryError("initial inventory reasons must be a list of strings")
    reasons = tuple(reasons_value)
    if reasons != tuple(sorted(set(reasons))):
        raise InventoryError("initial inventory reasons are not canonical")
    return InventoryRecord(
        sample_id=sample_id,
        relative_path=relative_path,
        sha256=file_hash or "",
        byte_size=byte_size,
        class_name=class_name,
        class_id=class_id,
        source_partition=source_partition,
        source_partition_name=source_partition_name,
        image_format=image_format,
        width=width,
        height=height,
        mode=mode,
        channels=channels,
        is_animated=is_animated,
        exif_orientation=exif_orientation,
        exif_transpose_required=exif_transpose_required,
        color_conversion=color_conversion,
        status=status,
        reasons=reasons,
    )


def _required_text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise InventoryError(f"{field} must be a non-empty string")
    return value


def _optional_text(
    value: object,
    field: str,
    *,
    empty_allowed: bool = False,
) -> str | None:
    if value is None:
        return None
    if type(value) is not str or (not value and not empty_allowed):
        raise InventoryError(f"{field} must be null or a valid string")
    return value


def _required_integer(value: object, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise InventoryError(f"{field} must be an integer >= {minimum}")
    return value


def _optional_integer(
    value: object,
    field: str,
    *,
    minimum: int,
) -> int | None:
    if value is None:
        return None
    return _required_integer(value, field, minimum=minimum)


def _required_boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise InventoryError(f"{field} must be a boolean")
    return value


def _optional_boolean(value: object, field: str) -> bool | None:
    if value is None:
        return None
    return _required_boolean(value, field)
