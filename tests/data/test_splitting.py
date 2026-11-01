from dataclasses import replace
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from chihuahua_muffin.data.deduplication import analyze_exact_duplicates
from chihuahua_muffin.data.inventory import inventory_dataset
from chihuahua_muffin.data.splitting import (
    DatasetSplit,
    SplitError,
    create_split_assignments,
    write_split_audit,
)


VERSION_ID = "kaggle:2733586:v2"


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (6, 5), color).save(path)
    return path


def _dataset(
    root: Path,
    *,
    train_counts: tuple[int, int] = (10, 10),
    test_counts: tuple[int, int] = (3, 3),
):
    for class_index, class_name in enumerate(("chihuahua", "muffin")):
        for index in range(train_counts[class_index]):
            _image(
                root / f"train/{class_name}/{index:03d}.png",
                (class_index * 100, index, 11),
            )
        for index in range(test_counts[class_index]):
            _image(
                root / f"test/{class_name}/{index:03d}.png",
                (class_index * 100, index, 22),
            )
    inventory = inventory_dataset(root, dataset_version_id=VERSION_ID)
    return inventory, analyze_exact_duplicates(inventory)


def test_split_is_stratified_and_preserves_metadata_and_source_rules(
    tmp_path: Path,
) -> None:
    inventory, exact = _dataset(
        tmp_path,
        train_counts=(13, 7),
        test_counts=(4, 2),
    )

    result = create_split_assignments(inventory, exact)
    assignments = {item.sample_id: item for item in result.assignments}
    accepted = {
        record.sample_id: record
        for record in inventory.records
        if record.image_format is not None
    }

    assert result.counts_by_split_and_class == {
        "train": {"chihuahua": 11, "muffin": 6},
        "validation": {"chihuahua": 2, "muffin": 1},
        "test": {"chihuahua": 4, "muffin": 2},
    }
    assert len(assignments) == len(result.assignments) == len(accepted)
    assert all(
        item.source_partition == "train"
        for item in result.assignments
        if item.split is DatasetSplit.VALIDATION
    )
    for sample_id, assignment in assignments.items():
        record = accepted[sample_id]
        assert (
            assignment.relative_path,
            assignment.sha256,
            assignment.class_name,
            assignment.class_id,
        ) == (
            record.relative_path,
            record.sha256,
            record.class_name,
            record.class_id,
        )


def test_original_test_ids_are_preserved_integrally(tmp_path: Path) -> None:
    inventory, exact = _dataset(tmp_path)

    result = create_split_assignments(inventory, exact)

    original_test_ids = {
        record.sample_id
        for record in inventory.records
        if record.source_partition == "test"
    }
    assigned_test_ids = {
        item.sample_id
        for item in result.assignments
        if item.split is DatasetSplit.TEST
    }
    assert assigned_test_ids == original_test_ids
    assert all(
        item.source_partition == "test"
        for item in result.assignments
        if item.split is DatasetSplit.TEST
    )


def test_split_is_reproducible_and_independent_of_inventory_order(
    tmp_path: Path,
) -> None:
    inventory, exact = _dataset(tmp_path, train_counts=(17, 11))
    reordered_inventory = replace(
        inventory,
        records=tuple(reversed(inventory.records)),
    )
    reordered_exact = analyze_exact_duplicates(reordered_inventory)

    first = create_split_assignments(inventory, exact)
    second = create_split_assignments(inventory, exact)
    reordered = create_split_assignments(reordered_inventory, reordered_exact)

    assert first == second == reordered
    assert first.assignments_payload() == reordered.assignments_payload()


def test_exact_duplicate_exclusions_are_respected_without_moving_files(
    tmp_path: Path,
) -> None:
    inventory, _ = _dataset(tmp_path, train_counts=(4, 4), test_counts=(2, 2))
    source = tmp_path / "train/chihuahua/000.png"
    duplicate = tmp_path / "test/chihuahua/copied.png"
    shutil.copyfile(source, duplicate)
    before = {path: path.read_bytes() for path in (source, duplicate)}
    inventory = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)
    exact = analyze_exact_duplicates(inventory)

    result = create_split_assignments(inventory, exact)

    assigned_ids = {item.sample_id for item in result.assignments}
    excluded_ids = {
        outcome.sample_id
        for group in exact.groups
        for outcome in group.outcomes
        if outcome.action.value == "exclude"
    }
    assert assigned_ids.isdisjoint(excluded_ids)
    assert result.exact_excluded_sample_count == 1
    assert {path: path.read_bytes() for path in (source, duplicate)} == before


def test_exact_policy_cannot_remove_an_original_test_id(tmp_path: Path) -> None:
    _dataset(tmp_path, train_counts=(4, 4), test_counts=(2, 2))
    shutil.copyfile(
        tmp_path / "test/chihuahua/000.png",
        tmp_path / "test/chihuahua/001.png",
    )
    inventory = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)
    exact = analyze_exact_duplicates(inventory)

    with pytest.raises(SplitError, match="remove IDs from the original test"):
        create_split_assignments(inventory, exact)


@pytest.mark.parametrize(
    ("train_counts", "test_counts", "partition"),
    [
        ((2, 0), (1, 1), "train"),
        ((1, 2), (1, 1), "train"),
        ((2, 2), (1, 0), "test"),
    ],
)
def test_empty_or_insufficient_classes_are_rejected(
    tmp_path: Path,
    train_counts: tuple[int, int],
    test_counts: tuple[int, int],
    partition: str,
) -> None:
    inventory, exact = _dataset(
        tmp_path,
        train_counts=train_counts,
        test_counts=test_counts,
    )

    with pytest.raises(SplitError, match=f"source {partition}"):
        create_split_assignments(inventory, exact)


def test_blocking_exact_review_prevents_split(tmp_path: Path) -> None:
    original = _image(tmp_path / "train/chihuahua/conflict.png", (1, 2, 3))
    copied = tmp_path / "test/muffin/conflict.png"
    copied.parent.mkdir(parents=True)
    shutil.copyfile(original, copied)
    for partition in ("train", "test"):
        _image(tmp_path / f"{partition}/chihuahua/other.png", (10, 20, 30))
        _image(tmp_path / f"{partition}/muffin/other.png", (30, 20, 10))
    inventory = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)
    exact = analyze_exact_duplicates(inventory)

    with pytest.raises(SplitError, match="blocking reviews"):
        create_split_assignments(inventory, exact)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": 7}, "seed"),
        ({"validation_fraction": 0.2}, "validation_fraction"),
        (
            {"known_unreviewed_perceptual_candidate_pair_count": -1},
            "candidate count",
        ),
    ],
)
def test_only_approved_split_configuration_is_accepted(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    inventory, exact = _dataset(tmp_path)

    with pytest.raises(SplitError, match=message):
        create_split_assignments(inventory, exact, **kwargs)  # type: ignore[arg-type]


def test_split_audit_is_canonical_deterministic_and_explicitly_nonfinal(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    inventory, exact = _dataset(source)
    originals = {
        path: path.read_bytes() for path in source.rglob("*.png")
    }
    result = create_split_assignments(
        inventory,
        exact,
        known_unreviewed_perceptual_candidate_pair_count=719,
    )

    first_assignments, first_report = write_split_audit(result, tmp_path / "first")
    second_assignments, second_report = write_split_audit(result, tmp_path / "second")
    report = json.loads(first_report.read_text(encoding="utf-8"))

    assert first_assignments.read_bytes() == second_assignments.read_bytes()
    assert first_report.read_bytes() == second_report.read_bytes()
    assert report["scenario"] == "A"
    assert report["source_test_ids_preserved"] is True
    assert report["final_manifest_created"] is False
    assert report["perceptual_policy"] == {
        "candidate_pairs_are_confirmed_duplicates": False,
        "candidate_pairs_used_for_assignment": False,
        "candidate_pairs_used_for_grouping": False,
        "known_unreviewed_candidate_pair_count": 719,
        "manual_review_required_for_didactic_experiment": False,
    }
    assert not first_assignments.read_bytes().endswith(b"\n")
    assert {path: path.read_bytes() for path in source.rglob("*.png")} == originals
    with pytest.raises(SplitError, match="already exists"):
        write_split_audit(result, tmp_path / "first")


def test_inventory_and_exact_result_must_match(tmp_path: Path) -> None:
    inventory, exact = _dataset(tmp_path)
    mismatched = replace(exact, dataset_version_id="other-version")

    with pytest.raises(SplitError, match="different versions"):
        create_split_assignments(inventory, mismatched)
