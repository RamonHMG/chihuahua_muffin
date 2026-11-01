import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from chihuahua_muffin.data.inventory import (
    AuditStatus,
    InventoryError,
    inventory_dataset,
    sha256_file,
    write_initial_audit,
)


VERSION_ID = "kaggle:2733586:v2"


def _save(
    path: Path,
    *,
    mode: str = "RGB",
    size: tuple[int, int] = (7, 5),
    color: object = (10, 20, 30),
    image_format: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path, format=image_format)
    return path


def test_inventory_is_deterministic_and_records_source_metadata(tmp_path: Path) -> None:
    second = _save(tmp_path / "train/muffin/z.png", mode="RGBA", color=(1, 2, 3, 4))
    first = _save(tmp_path / "test/chihuahua/a.jpg")

    one = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)
    two = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)

    assert one == two
    assert [record.relative_path for record in one.records] == [
        "test/chihuahua/a.jpg",
        "train/muffin/z.png",
    ]
    jpeg, png = one.records
    assert jpeg.class_name == "chihuahua"
    assert jpeg.class_id == 0
    assert jpeg.source_partition == "test"
    assert jpeg.image_format == "JPEG"
    assert (jpeg.width, jpeg.height, jpeg.channels) == (7, 5, 3)
    assert jpeg.sha256 == sha256_file(first)
    assert jpeg.status is AuditStatus.ACCEPTED
    assert png.class_name == "muffin"
    assert png.class_id == 1
    assert png.source_partition == "train"
    assert png.image_format == "PNG"
    assert png.mode == "RGBA"
    assert png.channels == 4
    assert png.color_conversion == "rgba_on_white"
    assert png.sha256 == sha256_file(second)


def test_inventory_reconciles_invalid_and_non_image_files(tmp_path: Path) -> None:
    invalid = tmp_path / "train/chihuahua/broken.jpg"
    invalid.parent.mkdir(parents=True)
    invalid.write_bytes(b"not a jpeg")
    note = tmp_path / "README.txt"
    note.write_text("metadata", encoding="utf-8")

    result = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)
    by_path = {record.relative_path: record for record in result.records}

    assert by_path["train/chihuahua/broken.jpg"].status is AuditStatus.EXCLUDED
    assert "invalid_or_unreadable_image" in by_path[
        "train/chihuahua/broken.jpg"
    ].reasons
    assert by_path["README.txt"].status is AuditStatus.EXCLUDED
    assert "non_image_file" in by_path["README.txt"].reasons
    report = result.quality_payload()
    assert report["total_discovered_files"] == 2
    assert report["reconciled_files"] == 2
    assert report["counts_by_status"] == {"excluded": 2}


@pytest.mark.parametrize(
    ("mode", "color", "conversion"),
    [
        ("RGB", (1, 2, 3), "preserve_rgb"),
        ("RGBA", (1, 2, 3, 4), "rgba_on_white"),
        ("L", 40, "grayscale_replicate"),
    ],
)
def test_supported_color_modes_are_recorded(
    tmp_path: Path, mode: str, color: object, conversion: str
) -> None:
    _save(tmp_path / f"train/chihuahua/{mode}.png", mode=mode, color=color)

    record = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID).records[0]

    assert record.status is AuditStatus.ACCEPTED
    assert record.mode == mode
    assert record.color_conversion == conversion


def test_animation_is_excluded_with_reason(tmp_path: Path) -> None:
    path = tmp_path / "train/muffin/animated.gif"
    path.parent.mkdir(parents=True)
    frames = [Image.new("RGB", (3, 3), color) for color in ("red", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=20)

    record = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID).records[0]

    assert record.is_animated is True
    assert record.status is AuditStatus.EXCLUDED
    assert "animated_image" in record.reasons


def test_decompression_bomb_error_is_explicitly_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save(tmp_path / "train/chihuahua/image.png")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    record = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID).records[0]

    assert record.status is AuditStatus.EXCLUDED
    assert "decompression_bomb_error" in record.reasons


def test_unexpected_format_and_extension_mismatch_require_review(tmp_path: Path) -> None:
    _save(tmp_path / "train/chihuahua/image.bmp", image_format="BMP")
    _save(tmp_path / "train/muffin/not-really-jpeg.jpg", image_format="PNG")

    result = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)
    by_path = {record.relative_path: record for record in result.records}

    assert by_path["train/chihuahua/image.bmp"].status is AuditStatus.REVIEW
    assert "unexpected_image_format" in by_path["train/chihuahua/image.bmp"].reasons
    mismatch = by_path["train/muffin/not-really-jpeg.jpg"]
    assert mismatch.status is AuditStatus.REVIEW
    assert "format_extension_mismatch" in mismatch.reasons


def test_exif_orientation_is_recorded_for_later_correction(tmp_path: Path) -> None:
    path = tmp_path / "test/chihuahua/oriented.jpg"
    path.parent.mkdir(parents=True)
    image = Image.new("RGB", (4, 2), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)

    record = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID).records[0]

    assert record.exif_orientation == 6
    assert record.exif_transpose_required is True
    assert (record.width, record.height) == (4, 2)


def test_exact_hashes_are_candidates_without_automatic_exclusion(tmp_path: Path) -> None:
    original = _save(tmp_path / "train/chihuahua/a.png")
    duplicate = tmp_path / "test/chihuahua/b.png"
    duplicate.parent.mkdir(parents=True)
    shutil.copyfile(original, duplicate)

    result = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)

    assert all(record.status is AuditStatus.ACCEPTED for record in result.records)
    assert len(result.exact_duplicate_candidates) == 1
    candidate = result.exact_duplicate_candidates[0]
    assert candidate.sha256 == sha256_file(original)
    assert candidate.relative_paths == (
        "train/chihuahua/a.png",
        "test/chihuahua/b.png",
    )


def test_unknown_class_is_not_silently_accepted(tmp_path: Path) -> None:
    _save(tmp_path / "train/other/image.png")

    record = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID).records[0]

    assert record.status is AuditStatus.REVIEW
    assert record.class_name is None
    assert "class_not_identified" in record.reasons


def test_inventory_and_preprocessing_do_not_modify_original_bytes(tmp_path: Path) -> None:
    path = _save(tmp_path / "train/muffin/image.png", mode="RGBA", color=(4, 5, 6, 7))
    before = path.read_bytes()

    inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)

    assert path.read_bytes() == before


def test_initial_audit_writes_canonical_local_artifacts_without_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _save(source / "train/chihuahua/image.png")
    result = inventory_dataset(source, dataset_version_id=VERSION_ID)
    destination = tmp_path / "audit"

    inventory_path, quality_path = write_initial_audit(result, destination)

    assert json.loads(inventory_path.read_text(encoding="utf-8"))[
        "artifact_type"
    ] == "initial_inventory"
    assert json.loads(quality_path.read_text(encoding="utf-8"))[
        "artifact_type"
    ] == "initial_quality_report"
    assert not inventory_path.read_bytes().endswith(b"\n")
    with pytest.raises(InventoryError, match="already exists"):
        write_initial_audit(result, destination)


def test_inventory_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(InventoryError, match="existing directory"):
        inventory_dataset(tmp_path / "missing", dataset_version_id=VERSION_ID)
