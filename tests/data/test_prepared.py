import csv
from io import StringIO
import json
from pathlib import Path

from PIL import Image
import pytest

from chihuahua_muffin.data.canonical import (
    canonical_index_csv_bytes,
    canonical_json_bytes,
    manifest_sha256,
    sha256_hex,
)
from chihuahua_muffin.data.deduplication import (
    analyze_exact_duplicates,
    assess_validation_feasibility,
    write_exact_audit,
)
from chihuahua_muffin.data.inventory import inventory_dataset, sha256_file, write_initial_audit
from chihuahua_muffin.data.prepared import (
    PreparationError,
    PreparationLayout,
    build_prepared_artifacts,
    load_pipeline_inputs,
    prepare_dataset,
    publish_prepared_artifacts,
    validate_prepared_dataset,
)
from chihuahua_muffin.data.review import canonical_human_decisions_bytes
from chihuahua_muffin.data.splitting import create_split_assignments, write_split_audit


VERSION_ID = "kaggle:2733586:v2"
PROJECT_ROOT = Path(__file__).parents[2]
EXAMPLE_CONFIG = PROJECT_ROOT / "config/example.toml"


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color).save(path)


def _synthetic_pipeline(tmp_path: Path):
    root = tmp_path / "project"
    layout = PreparationLayout.default(root)
    raw = layout.raw_root
    for class_index, (class_name, train_count, test_count) in enumerate(
        (("chihuahua", 7, 3), ("muffin", 5, 2))
    ):
        for index in range(train_count):
            _image(
                raw / f"train/{class_name}/{index:02d}.png",
                (class_index * 100, index, 10),
            )
        for index in range(test_count):
            _image(
                raw / f"test/{class_name}/{index:02d}.png",
                (class_index * 100, index, 20),
            )
    marker = raw / ".complete/bundle.complete"
    marker.parent.mkdir(parents=True)
    marker.write_text("synthetic marker", encoding="utf-8")

    inventory = inventory_dataset(raw, dataset_version_id=VERSION_ID)
    write_initial_audit(inventory, layout.initial_inventory.parent)
    exact = analyze_exact_duplicates(inventory)
    exact_assessment = assess_validation_feasibility(exact)
    write_exact_audit(
        exact,
        exact_assessment,
        (),
        layout.exact_deduplication.parent,
    )
    split = create_split_assignments(
        inventory,
        exact,
        known_unreviewed_perceptual_candidate_pair_count=0,
    )
    write_split_audit(split, layout.split_assignments.parent)

    perceptual_dir = layout.perceptual_audit.parent
    perceptual_dir.mkdir(parents=True)
    perceptual = {
        "artifact_type": "perceptual_duplicate_audit",
        "artifact_version": "1.0",
        "dataset_version_id": VERSION_ID,
        "images_analyzed": len(split.assignments),
        "candidate_pair_count": 0,
        "reviewed_candidate_count": 0,
        "label_conflict_candidate_count": 0,
        "method": {
            "name": "phash",
            "library": "ImageHash",
            "library_version": "4.3.2",
            "hash_size": 8,
            "highfreq_factor": 4,
            "hash_bits": 64,
            "distance": "hamming",
            "purpose": "candidate_triage_only",
        },
        "candidate_screening": {
            "maximum_hamming_distance": 12,
            "is_duplicate_threshold": False,
            "automatic_removal": False,
            "requires_human_review": True,
        },
    }
    layout.perceptual_audit.write_bytes(canonical_json_bytes(perceptual))
    layout.perceptual_human_decisions.write_bytes(
        canonical_human_decisions_bytes(VERSION_ID, ())
    )
    layout.perceptual_validation_feasibility.write_bytes(
        canonical_json_bytes({"artifact_type": "synthetic_feasibility"})
    )
    layout.perceptual_review_panel.write_text(
        "<!doctype html><title>Synthetic review</title>", encoding="utf-8"
    )

    config = EXAMPLE_CONFIG.read_text(encoding="utf-8")
    config = config.replace(
        "acquisition_file_count = 5918",
        f"acquisition_file_count = {len(inventory.records)}",
    )
    layout.config_path.parent.mkdir(parents=True)
    layout.config_path.write_text(config, encoding="utf-8")

    digests = {
        name: sha256_file(path)
        for name, path in layout.source_artifacts().items()
    }
    counts = {
        "train": {"chihuahua": 6, "muffin": 4},
        "validation": {"chihuahua": 1, "muffin": 1},
        "test": {"chihuahua": 3, "muffin": 2},
    }
    return layout, digests, counts


def _prepare(tmp_path: Path):
    layout, digests, counts = _synthetic_pipeline(tmp_path)
    result = prepare_dataset(
        layout,
        approved_source_digests=digests,
        expected_split_counts=counts,
        expected_perceptual_candidate_count=0,
    )
    return layout, digests, counts, result


def _rewrite_manifest(layout: PreparationLayout, mutate) -> None:
    path = layout.prepared_dir / "dataset_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["manifest_sha256"] = manifest_sha256(
        manifest, digest_field="manifest_sha256"
    )
    path.write_bytes(canonical_json_bytes(manifest))


def _parsed_index(layout: PreparationLayout) -> list[dict[str, object]]:
    content = (layout.prepared_dir / "index.csv").read_text(encoding="utf-8")
    result = []
    for row in csv.DictReader(StringIO(content, newline="")):
        parsed: dict[str, object] = dict(row)
        for field in ("class_id", "width", "height", "channels"):
            if row[field]:
                parsed[field] = int(row[field])
        for field in ("is_animated", "exif_orientation_applied"):
            if row[field]:
                parsed[field] = row[field] == "true"
        result.append(parsed)
    return result


def _replace_index(layout: PreparationLayout, rows: list[dict[str, object]]) -> None:
    content = canonical_index_csv_bytes(rows)
    (layout.prepared_dir / "index.csv").write_bytes(content)

    def update(manifest: dict[str, object]) -> None:
        manifest["artifacts"]["index"]["sha256"] = sha256_hex(content)  # type: ignore[index]
        manifest["artifacts"]["index"]["row_count"] = len(rows)  # type: ignore[index]

    _rewrite_manifest(layout, update)


def test_complete_pipeline_generates_and_validates_all_final_artifacts(
    tmp_path: Path,
) -> None:
    layout, _, counts, result = _prepare(tmp_path)

    assert result.publication.status == "created"
    assert {path.name for path in layout.prepared_dir.iterdir()} == {
        "index.csv",
        "dataset_manifest.json",
        "quality_report.json",
    }
    assert result.validation.discovered_file_count == 18
    assert result.validation.accepted_image_count == 17
    assert result.validation.counts_by_split_and_class == counts
    manifest = json.loads(
        (layout.prepared_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    quality = json.loads(
        (layout.prepared_dir / "quality_report.json").read_text(encoding="utf-8")
    )
    assert manifest["license"]["status"] == "ambiguous"
    assert manifest["license"]["confirmed"] is False
    assert manifest["split"]["counts_by_split_and_class"] == counts
    assert quality["counts_by_status"] == {"accepted": 17, "excluded": 1}
    assert quality["excluded_occurrences"][0]["reason"] == (
        "class_not_identified;non_image_file"
    )


def test_reexecution_is_deterministic_and_recognizes_identical_output(
    tmp_path: Path,
) -> None:
    layout, digests, counts, first = _prepare(tmp_path)
    before = {
        path.name: path.read_bytes() for path in layout.prepared_dir.iterdir()
    }

    second = prepare_dataset(
        layout,
        approved_source_digests=digests,
        expected_split_counts=counts,
        expected_perceptual_candidate_count=0,
    )

    assert first.publication.status == "created"
    assert second.publication.status == "up_to_date"
    assert {path.name: path.read_bytes() for path in layout.prepared_dir.iterdir()} == before


def test_validator_detects_changed_final_artifact_digest(tmp_path: Path) -> None:
    layout, digests, counts, _ = _prepare(tmp_path)
    index = layout.prepared_dir / "index.csv"
    index.write_bytes(index.read_bytes() + b"\n")

    with pytest.raises(PreparationError, match="digest|canonical"):
        validate_prepared_dataset(
            layout,
            approved_source_digests=digests,
            expected_split_counts=counts,
            expected_perceptual_candidate_count=0,
        )


def test_validator_rejects_unsafe_index_path_even_with_updated_digests(
    tmp_path: Path,
) -> None:
    layout, digests, counts, _ = _prepare(tmp_path)
    rows = _parsed_index(layout)
    rows[0]["relative_path"] = "../escape.png"
    _replace_index(layout, rows)

    with pytest.raises(PreparationError, match="unsafe path"):
        validate_prepared_dataset(
            layout,
            approved_source_digests=digests,
            expected_split_counts=counts,
            expected_perceptual_candidate_count=0,
        )


def test_validator_rejects_incompatible_manifest_schema(tmp_path: Path) -> None:
    layout, digests, counts, _ = _prepare(tmp_path)
    _rewrite_manifest(layout, lambda manifest: manifest.update(schema_version="2.0"))

    with pytest.raises(PreparationError, match="schema_version"):
        validate_prepared_dataset(
            layout,
            approved_source_digests=digests,
            expected_split_counts=counts,
            expected_perceptual_candidate_count=0,
        )


def test_validator_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    layout, digests, counts, _ = _prepare(tmp_path)
    index_path = layout.prepared_dir / "index.csv"
    lines = index_path.read_bytes().splitlines(keepends=True)
    first_id = lines[1].split(b",", 1)[0]
    lines[2] = first_id + b"," + lines[2].split(b",", 1)[1]
    content = b"".join(lines)
    index_path.write_bytes(content)
    _rewrite_manifest(
        layout,
        lambda manifest: manifest["artifacts"]["index"].update(  # type: ignore[index]
            sha256=sha256_hex(content)
        ),
    )

    with pytest.raises(PreparationError, match="duplicate sample_id|canonical contract"):
        validate_prepared_dataset(
            layout,
            approved_source_digests=digests,
            expected_split_counts=counts,
            expected_perceptual_candidate_count=0,
        )


def test_validator_rejects_class_or_count_divergence(tmp_path: Path) -> None:
    layout, digests, counts, _ = _prepare(tmp_path)
    rows = _parsed_index(layout)
    accepted_muffin = next(
        row for row in rows if row["status"] == "accepted" and row["class_name"] == "muffin"
    )
    accepted_muffin["class_name"] = "chihuahua"
    accepted_muffin["class_id"] = 0
    _replace_index(layout, rows)

    with pytest.raises(PreparationError, match="counts|class"):
        validate_prepared_dataset(
            layout,
            approved_source_digests=digests,
            expected_split_counts=counts,
            expected_perceptual_candidate_count=0,
        )


def test_validator_detects_modified_original(tmp_path: Path) -> None:
    layout, digests, counts, _ = _prepare(tmp_path)
    source = next(layout.raw_root.rglob("*.png"))
    source.write_bytes(source.read_bytes() + b"changed")

    with pytest.raises(PreparationError, match="SHA-256"):
        validate_prepared_dataset(
            layout,
            approved_source_digests=digests,
            expected_split_counts=counts,
            expected_perceptual_candidate_count=0,
        )


def test_failed_publication_leaves_no_partial_final_directory(tmp_path: Path) -> None:
    layout, digests, counts = _synthetic_pipeline(tmp_path)
    inputs = load_pipeline_inputs(
        layout,
        approved_source_digests=digests,
        expected_split_counts=counts,
        expected_perceptual_candidate_count=0,
    )
    artifacts = build_prepared_artifacts(inputs, expected_split_counts=counts)
    calls = 0

    def failing_writer(path: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        path.write_bytes(content)

    with pytest.raises(PreparationError, match="publication failed"):
        publish_prepared_artifacts(
            artifacts,
            layout.prepared_dir,
            writer=failing_writer,
        )

    assert not layout.prepared_dir.exists()
    assert not list(layout.prepared_dir.parent.glob(".prepared.tmp-*"))


def test_divergent_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    layout, digests, counts, _ = _prepare(tmp_path)
    manifest_path = layout.prepared_dir / "dataset_manifest.json"
    manifest_path.write_bytes(b"divergent")

    with pytest.raises(PreparationError, match="will not be overwritten"):
        prepare_dataset(
            layout,
            approved_source_digests=digests,
            expected_split_counts=counts,
            expected_perceptual_candidate_count=0,
        )
    assert manifest_path.read_bytes() == b"divergent"


def test_validator_is_read_only(tmp_path: Path) -> None:
    layout, digests, counts, _ = _prepare(tmp_path)
    paths = [
        *[path for path in layout.prepared_dir.iterdir()],
        *[path for path in layout.raw_root.rglob("*") if path.is_file()],
    ]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}

    validate_prepared_dataset(
        layout,
        approved_source_digests=digests,
        expected_split_counts=counts,
        expected_perceptual_candidate_count=0,
    )

    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before


def test_changed_approved_assignment_artifact_blocks_generation(tmp_path: Path) -> None:
    layout, digests, counts = _synthetic_pipeline(tmp_path)
    layout.split_assignments.write_bytes(layout.split_assignments.read_bytes() + b"\n")

    with pytest.raises(PreparationError, match="approved source artifact diverged"):
        prepare_dataset(
            layout,
            approved_source_digests=digests,
            expected_split_counts=counts,
            expected_perceptual_candidate_count=0,
        )
