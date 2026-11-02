"""Read-only consumer for the immutable SPEC 01 dataset contract."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
from typing import Any

from chihuahua_muffin.data.canonical import (
    INDEX_CSV_HEADER,
    canonical_index_csv_bytes,
    canonical_json_bytes,
    manifest_sha256,
    normalize_relative_path,
    sha256_hex,
)
from chihuahua_muffin.data.inventory import sha256_file

from .config import ModelRunConfig


class DatasetContractError(ValueError):
    """Raised before model work when the SPEC 01 contract is incompatible."""


@dataclass(frozen=True, slots=True)
class DatasetSample:
    sample_id: str
    relative_path: str
    path: Path
    class_name: str
    class_id: int
    source_partition: str
    split: str
    sha256: str


@dataclass(frozen=True, slots=True)
class DatasetContract:
    dataset_version_id: str
    manifest_sha256: str
    manifest_file_sha256: str
    index_sha256: str
    raw_root: Path
    samples: tuple[DatasetSample, ...]
    counts_by_split_and_class: dict[str, dict[str, int]]
    limitations: tuple[str, ...]

    def for_split(self, split: str) -> tuple[DatasetSample, ...]:
        if split not in {"train", "validation", "test"}:
            raise DatasetContractError(f"unknown split: {split}")
        return tuple(sample for sample in self.samples if sample.split == split)


REQUIRED_PREPROCESSING = {
    "application_stage": "read_time_by_spec_02_tf_data",
    "channels": 3,
    "color_mode": "RGB",
    "exclude_animated": True,
    "fix_exif_orientation": True,
    "grayscale_to_rgb": "replicate",
    "height": 128,
    "interpolation": "bilinear",
    "materialize_resized_images": False,
    "padding_color": "white",
    "padding_normalized": [1.0, 1.0, 1.0],
    "padding_rgb": [255, 255, 255],
    "pixel_max": 1.0,
    "pixel_min": 0.0,
    "resize": "pad",
    "rgba_background": "white",
    "width": 128,
}


def _canonical_object(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetContractError(f"cannot read {description}: {error}") from error
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise DatasetContractError(f"{description} is not canonical JSON")
    return value, raw


def _parse_index(raw: bytes) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(StringIO(text, newline=""))
        if tuple(reader.fieldnames or ()) != INDEX_CSV_HEADER:
            raise DatasetContractError("dataset index header is incompatible")
        rows = [dict(row) for row in reader]
    except (UnicodeError, csv.Error) as error:
        raise DatasetContractError(f"cannot parse dataset index: {error}") from error
    if canonical_index_csv_bytes(rows) != raw:
        raise DatasetContractError("dataset index is not canonical CSV")
    return rows


def _safe_project_path(project_root: Path, relative: Path, description: str) -> Path:
    target = project_root / relative
    try:
        target.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise DatasetContractError(f"{description} escapes project root") from error
    return target


def load_dataset_contract(
    project_root: str | Path,
    config: ModelRunConfig,
    *,
    verify_source_hashes: bool = True,
    enforce_approved_digests: bool = True,
) -> DatasetContract:
    """Validate the published contract and consume its existing assignments."""

    root = Path(project_root).resolve()
    manifest_path = _safe_project_path(root, config.dataset.manifest, "manifest")
    index_path = _safe_project_path(root, config.dataset.index, "index")
    quality_path = _safe_project_path(
        root, config.dataset.quality_report, "quality report"
    )
    manifest, manifest_bytes = _canonical_object(manifest_path, "dataset manifest")
    quality, quality_bytes = _canonical_object(quality_path, "quality report")

    if manifest.get("schema_version") != "1.0":
        raise DatasetContractError("dataset manifest schema_version is incompatible")
    self_digest = manifest.get("manifest_sha256")
    if type(self_digest) is not str or self_digest != manifest_sha256(
        manifest, digest_field="manifest_sha256"
    ):
        raise DatasetContractError("dataset manifest self-digest is invalid")
    if enforce_approved_digests and self_digest != config.dataset.manifest_sha256:
        raise DatasetContractError("dataset manifest differs from the approved contract")
    if manifest.get("class_mapping") != {"chihuahua": 0, "muffin": 1}:
        raise DatasetContractError("class mapping is incompatible")
    if manifest.get("positive_class") != "muffin":
        raise DatasetContractError("positive class must be muffin")
    if manifest.get("preprocessing") != REQUIRED_PREPROCESSING:
        raise DatasetContractError("preprocessing contract is incompatible")

    split_contract = manifest.get("split")
    if type(split_contract) is not dict or any(
        split_contract.get(field) != expected
        for field, expected in {
            "scenario": "A",
            "seed": 42,
            "algorithm": "stratified_sha256_rank_v1",
            "validation_source": "train",
            "test_original_preserved": True,
        }.items()
    ):
        raise DatasetContractError("split contract is incompatible")
    expected_counts = split_contract.get("counts_by_split_and_class")
    if type(expected_counts) is not dict or set(expected_counts) != {
        "train",
        "validation",
        "test",
    }:
        raise DatasetContractError("split counts are missing or incompatible")

    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict:
        raise DatasetContractError("manifest artifact references are missing")
    index_reference = artifacts.get("index")
    quality_reference = artifacts.get("quality_report")
    try:
        index_bytes = index_path.read_bytes()
    except OSError as error:
        raise DatasetContractError(f"cannot read dataset index: {error}") from error
    index_digest = sha256_hex(index_bytes)
    if type(index_reference) is not dict or index_reference.get("sha256") != index_digest:
        raise DatasetContractError("dataset index digest does not match manifest")
    if enforce_approved_digests and index_digest != config.dataset.index_sha256:
        raise DatasetContractError("dataset index differs from the approved contract")
    if type(quality_reference) is not dict or quality_reference.get("sha256") != sha256_hex(quality_bytes):
        raise DatasetContractError("quality report digest does not match manifest")

    rows = _parse_index(index_bytes)
    if index_reference.get("row_count") != len(rows):
        raise DatasetContractError("dataset index row count does not match manifest")
    dataset = manifest.get("dataset")
    if type(dataset) is not dict or type(dataset.get("raw_root")) is not str:
        raise DatasetContractError("raw dataset root is missing from manifest")
    raw_relative = Path(dataset["raw_root"])
    raw_root = _safe_project_path(root, raw_relative, "raw dataset root")
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise DatasetContractError("raw dataset root is unavailable")

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    samples: list[DatasetSample] = []
    resolved_raw = raw_root.resolve()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen_ids:
            raise DatasetContractError(f"repeated sample_id: {sample_id}")
        seen_ids.add(sample_id)
        if row["status"] != "accepted":
            continue
        if row["class_name"] not in {"chihuahua", "muffin"}:
            raise DatasetContractError(f"invalid class for sample {sample_id}")
        expected_class_id = 0 if row["class_name"] == "chihuahua" else 1
        if row["class_id"] != str(expected_class_id):
            raise DatasetContractError(f"invalid class_id for sample {sample_id}")
        if row["split"] not in {"train", "validation", "test"}:
            raise DatasetContractError(f"invalid split for sample {sample_id}")
        relative_path = normalize_relative_path(row["relative_path"])
        if relative_path != row["relative_path"]:
            raise DatasetContractError(f"noncanonical path for sample {sample_id}")
        source = raw_root / relative_path
        try:
            source.resolve().relative_to(resolved_raw)
        except ValueError as error:
            raise DatasetContractError(f"unsafe path for sample {sample_id}") from error
        if source.is_symlink() or not source.is_file():
            raise DatasetContractError(f"source file is unavailable for sample {sample_id}")
        content_hash = row["sha256"]
        if content_hash in seen_hashes:
            raise DatasetContractError("accepted SHA-256 appears more than once")
        seen_hashes.add(content_hash)
        if verify_source_hashes and sha256_file(source) != content_hash:
            raise DatasetContractError(f"source digest differs for sample {sample_id}")
        samples.append(
            DatasetSample(
                sample_id=sample_id,
                relative_path=relative_path,
                path=source,
                class_name=row["class_name"],
                class_id=expected_class_id,
                source_partition=row["source_partition"],
                split=row["split"],
                sha256=content_hash,
            )
        )

    actual = Counter((sample.split, sample.class_name) for sample in samples)
    normalized_counts = {
        split: {
            class_name: actual[(split, class_name)]
            for class_name in ("chihuahua", "muffin")
        }
        for split in ("train", "validation", "test")
    }
    if normalized_counts != expected_counts:
        raise DatasetContractError("index assignments differ from manifest split counts")
    counts = manifest.get("counts")
    if type(counts) is not dict or counts.get("accepted_images") != len(samples):
        raise DatasetContractError("accepted image count differs from manifest")
    if quality.get("accepted_counts_by_split_and_class") != normalized_counts:
        raise DatasetContractError("quality report split counts differ from index")

    limitations = manifest.get("limitations")
    if type(limitations) is not list or any(type(item) is not str for item in limitations):
        raise DatasetContractError("manifest limitations are invalid")
    return DatasetContract(
        dataset_version_id=str(manifest.get("dataset_version_id")),
        manifest_sha256=self_digest,
        manifest_file_sha256=sha256_hex(manifest_bytes),
        index_sha256=index_digest,
        raw_root=raw_root,
        samples=tuple(samples),
        counts_by_split_and_class=normalized_counts,
        limitations=tuple(limitations),
    )
