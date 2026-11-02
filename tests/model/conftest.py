"""Synthetic SPEC 02 contract fixtures independent of the real dataset."""

from dataclasses import replace
from pathlib import Path

from PIL import Image
import pytest

from chihuahua_muffin.data.canonical import (
    canonical_index_csv_bytes,
    canonical_json_bytes,
    manifest_sha256,
    sha256_hex,
)
from chihuahua_muffin.data.inventory import sha256_file
from chihuahua_muffin.model.config import load_model_config
from chihuahua_muffin.model.contract import REQUIRED_PREPROCESSING


@pytest.fixture
def synthetic_model_contract(tmp_path: Path):
    root = tmp_path / "project"
    raw = root / "data/raw/synthetic"
    rows = []
    sample_index = 0
    counts = {}
    for split in ("train", "validation", "test"):
        counts[split] = {"chihuahua": 0, "muffin": 0}
        for class_name, class_id in (("chihuahua", 0), ("muffin", 1)):
            relative = f"source/{split}-{class_name}.png"
            path = raw / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB", (7 + sample_index, 5), (class_id * 180, sample_index, 10)
            ).save(path)
            sample_id = f"{sample_index:064x}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "relative_path": relative,
                    "class_name": class_name,
                    "class_id": class_id,
                    "source_partition": "train" if split != "test" else "test",
                    "split": split,
                    "status": "accepted",
                    "reason": "",
                    "sha256": sha256_file(path),
                    "duplicate_group_id": "",
                    "perceptual_method": "",
                    "perceptual_evidence": "",
                    "format": "PNG",
                    "width": 7 + sample_index,
                    "height": 5,
                    "channels": 3,
                    "is_animated": False,
                    "exif_orientation_applied": False,
                    "color_conversion": "preserve_rgb",
                }
            )
            counts[split][class_name] += 1
            sample_index += 1

    prepared = root / "data/prepared"
    prepared.mkdir(parents=True)
    index_bytes = canonical_index_csv_bytes(rows)
    (prepared / "index.csv").write_bytes(index_bytes)
    quality = {
        "artifact_type": "final_quality_report",
        "schema_version": "1.0",
        "accepted_counts_by_split_and_class": counts,
    }
    quality_bytes = canonical_json_bytes(quality)
    (prepared / "quality_report.json").write_bytes(quality_bytes)
    manifest = {
        "artifact_type": "dataset_manifest",
        "schema_version": "1.0",
        "manifest_sha256": "",
        "dataset_version_id": "synthetic:v1",
        "dataset": {"raw_root": "data/raw/synthetic"},
        "class_mapping": {"chihuahua": 0, "muffin": 1},
        "positive_class": "muffin",
        "preprocessing": REQUIRED_PREPROCESSING,
        "split": {
            "scenario": "A",
            "seed": 42,
            "algorithm": "stratified_sha256_rank_v1",
            "validation_source": "train",
            "test_original_preserved": True,
            "counts_by_split_and_class": counts,
        },
        "counts": {"accepted_images": 6},
        "artifacts": {
            "index": {
                "path": "index.csv",
                "sha256": sha256_hex(index_bytes),
                "row_count": 6,
            },
            "quality_report": {
                "path": "quality_report.json",
                "sha256": sha256_hex(quality_bytes),
            },
        },
        "limitations": ["synthetic fixture"],
    }
    manifest["manifest_sha256"] = manifest_sha256(
        manifest, digest_field="manifest_sha256"
    )
    (prepared / "dataset_manifest.json").write_bytes(canonical_json_bytes(manifest))
    config = load_model_config()
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            manifest_sha256=manifest["manifest_sha256"],
            index_sha256=sha256_hex(index_bytes),
        ),
    )
    return root, config, manifest
