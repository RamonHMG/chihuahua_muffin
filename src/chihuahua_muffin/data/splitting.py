"""Deterministic train/validation/test assignments for SPEC 01."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

from .canonical import canonical_json_bytes, sha256_hex
from .deduplication import ExactDeduplicationResult, SampleAction
from .inventory import AuditStatus, CLASS_IDS, InventoryRecord, InventoryResult


APPROVED_SEED = 42
APPROVED_VALIDATION_FRACTION = 0.15
_SPLIT_KEY_DOMAIN = b"chihuahua-muffin:validation-split:v1\x00"


class SplitError(ValueError):
    """Raised when approved split assignments cannot be created safely."""


class DatasetSplit(StrEnum):
    """Final logical subsets owned by SPEC 01."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Preserved source metadata plus one deterministic logical split."""

    sample_id: str
    relative_path: str
    sha256: str
    class_name: str
    class_id: int
    source_partition: str
    split: DatasetSplit

    def as_json(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "source_partition": self.source_partition,
            "split": self.split.value,
        }


@dataclass(frozen=True, slots=True)
class DatasetSplitResult:
    """Approved Scenario A assignments, before the final Increment 8 manifest."""

    dataset_version_id: str
    seed: int
    validation_fraction_of_source_train: float
    assignments: tuple[SplitAssignment, ...]
    exact_duplicate_group_count: int
    exact_excluded_sample_count: int
    known_unreviewed_perceptual_candidate_pair_count: int

    @property
    def counts_by_split_and_class(self) -> dict[str, dict[str, int]]:
        counts = Counter(
            (assignment.split.value, assignment.class_name)
            for assignment in self.assignments
        )
        return {
            split.value: {
                class_name: counts[(split.value, class_name)]
                for class_name in CLASS_IDS
            }
            for split in DatasetSplit
        }

    @property
    def counts_by_source_partition_and_class(self) -> dict[str, dict[str, int]]:
        counts = Counter(
            (assignment.source_partition, assignment.class_name)
            for assignment in self.assignments
        )
        return {
            partition: {
                class_name: counts[(partition, class_name)]
                for class_name in CLASS_IDS
            }
            for partition in ("train", "test")
        }

    def assignments_payload(self) -> dict[str, object]:
        return {
            "artifact_type": "dataset_split_assignments",
            "artifact_version": "1.0",
            "artifact_status": "nonfinal_increment_7_record",
            "dataset_version_id": self.dataset_version_id,
            "assignments": [assignment.as_json() for assignment in self.assignments],
        }

    def report_payload(self, *, assignments_sha256: str) -> dict[str, object]:
        split_counts = self.counts_by_split_and_class
        source_counts = self.counts_by_source_partition_and_class
        source_train_total = sum(source_counts["train"].values())
        validation_total = sum(split_counts[DatasetSplit.VALIDATION.value].values())
        train_total = sum(split_counts[DatasetSplit.TRAIN.value].values())
        return {
            "artifact_type": "dataset_split_report",
            "artifact_version": "1.0",
            "artifact_status": "nonfinal_increment_7_record",
            "dataset_version_id": self.dataset_version_id,
            "scenario": "A",
            "algorithm": "stratified_sha256_rank_v1",
            "seed": self.seed,
            "requested_validation_fraction_of_source_train": (
                self.validation_fraction_of_source_train
            ),
            "effective_train_fraction_of_source_train": (
                train_total / source_train_total
            ),
            "effective_validation_fraction_of_source_train": (
                validation_total / source_train_total
            ),
            "class_mapping": dict(CLASS_IDS),
            "counts_by_split_and_class": split_counts,
            "counts_by_source_partition_and_class": source_counts,
            "assignment_count": len(self.assignments),
            "assignments_sha256": assignments_sha256,
            "source_train_policy": "85_percent_train_15_percent_validation",
            "source_test_policy": "preserved_in_full_without_redistribution",
            "source_test_ids_preserved": True,
            "validation_uses_only_source_train": True,
            "sample_ids_are_disjoint_across_splits": True,
            "exact_duplicate_group_count": self.exact_duplicate_group_count,
            "exact_excluded_sample_count": self.exact_excluded_sample_count,
            "perceptual_policy": {
                "known_unreviewed_candidate_pair_count": (
                    self.known_unreviewed_perceptual_candidate_pair_count
                ),
                "candidate_pairs_are_confirmed_duplicates": False,
                "candidate_pairs_used_for_grouping": False,
                "candidate_pairs_used_for_assignment": False,
                "manual_review_required_for_didactic_experiment": False,
            },
            "final_manifest_created": False,
            "originals_modified": False,
            "limitations": [
                "Unreviewed pHash candidates may cross train, validation, and test.",
                "Possible overlap limits interpretation of generalization metrics.",
                "Perceptual similarity is not evidence of confirmed duplication.",
                "This report is an Increment 7 input to the future final manifest.",
            ],
        }


def create_split_assignments(
    inventory: InventoryResult,
    exact_result: ExactDeduplicationResult,
    *,
    validation_fraction: float = APPROVED_VALIDATION_FRACTION,
    seed: int = APPROVED_SEED,
    known_unreviewed_perceptual_candidate_pair_count: int = 0,
) -> DatasetSplitResult:
    """Create approved Scenario A assignments without modifying source files."""

    _validate_configuration(
        validation_fraction=validation_fraction,
        seed=seed,
        perceptual_candidate_count=known_unreviewed_perceptual_candidate_pair_count,
    )
    if inventory.dataset_version_id != exact_result.dataset_version_id:
        raise SplitError("inventory and exact result belong to different versions")
    if exact_result.training_release_blocked:
        raise SplitError("exact deduplication has unresolved blocking reviews")

    eligible = _eligible_records(inventory, exact_result)
    source_train = tuple(
        record for record in eligible if record.source_partition == "train"
    )
    source_test = tuple(
        record for record in eligible if record.source_partition == "test"
    )
    original_test_ids = {
        record.sample_id
        for record in inventory.records
        if record.status is AuditStatus.ACCEPTED
        and record.image_format is not None
        and record.source_partition == "test"
    }
    if {record.sample_id for record in source_test} != original_test_ids:
        raise SplitError("exact policy would remove IDs from the original test")
    unexpected_partitions = sorted(
        {
            record.source_partition or "none"
            for record in eligible
            if record.source_partition not in {"train", "test"}
        }
    )
    if unexpected_partitions:
        raise SplitError(
            f"eligible records have unsupported source partitions: {unexpected_partitions}"
        )

    train_by_class = _records_by_class(source_train, partition="train", minimum=2)
    _records_by_class(source_test, partition="test", minimum=1)

    validation_ids: set[str] = set()
    for class_name in CLASS_IDS:
        ordered = sorted(
            train_by_class[class_name],
            key=lambda record: _selection_key(record.sample_id, seed=seed),
        )
        validation_count = _validation_count(len(ordered), validation_fraction)
        validation_ids.update(
            record.sample_id for record in ordered[:validation_count]
        )

    assignments = []
    for record in eligible:
        if record.source_partition == "test":
            split = DatasetSplit.TEST
        elif record.sample_id in validation_ids:
            split = DatasetSplit.VALIDATION
        else:
            split = DatasetSplit.TRAIN
        assignments.append(_assignment(record, split=split))

    result = DatasetSplitResult(
        dataset_version_id=inventory.dataset_version_id,
        seed=seed,
        validation_fraction_of_source_train=validation_fraction,
        assignments=tuple(sorted(assignments, key=lambda item: item.sample_id)),
        exact_duplicate_group_count=len(exact_result.groups),
        exact_excluded_sample_count=exact_result.excluded_sample_count,
        known_unreviewed_perceptual_candidate_pair_count=(
            known_unreviewed_perceptual_candidate_pair_count
        ),
    )
    _validate_result(result, source_train=source_train, source_test=source_test)
    return result


def write_split_audit(
    result: DatasetSplitResult,
    destination: str | Path,
) -> tuple[Path, Path]:
    """Write non-final local assignment and report artifacts without overwrite."""

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise SplitError(f"split audit destination already exists: {target}")
    assignments_bytes = canonical_json_bytes(result.assignments_payload())
    report_bytes = canonical_json_bytes(
        result.report_payload(assignments_sha256=sha256_hex(assignments_bytes))
    )
    target.mkdir(parents=True, exist_ok=False)
    assignments_path = target / "split_assignments.json"
    report_path = target / "split_report.json"
    assignments_path.write_bytes(assignments_bytes)
    report_path.write_bytes(report_bytes)
    return assignments_path, report_path


def _validate_configuration(
    *, validation_fraction: float, seed: int, perceptual_candidate_count: int
) -> None:
    if type(validation_fraction) is not float or validation_fraction != 0.15:
        raise SplitError("validation_fraction must be the approved value 0.15")
    if type(seed) is not int or seed != APPROVED_SEED:
        raise SplitError("seed must be the approved value 42")
    if type(perceptual_candidate_count) is not int or perceptual_candidate_count < 0:
        raise SplitError("perceptual candidate count must be a non-negative integer")


def _eligible_records(
    inventory: InventoryResult,
    exact_result: ExactDeduplicationResult,
) -> tuple[InventoryRecord, ...]:
    accepted = tuple(
        record
        for record in inventory.records
        if record.status is AuditStatus.ACCEPTED and record.image_format is not None
    )
    accepted_by_id: dict[str, InventoryRecord] = {}
    for record in accepted:
        if record.sample_id in accepted_by_id:
            raise SplitError(f"duplicate accepted sample_id: {record.sample_id}")
        accepted_by_id[record.sample_id] = record

    actions: dict[str, SampleAction] = {}
    for group in exact_result.groups:
        for outcome in group.outcomes:
            if outcome.sample_id not in accepted_by_id:
                raise SplitError(
                    f"exact outcome references unknown sample_id: {outcome.sample_id}"
                )
            if outcome.sample_id in actions:
                raise SplitError(
                    f"sample appears in multiple exact groups: {outcome.sample_id}"
                )
            actions[outcome.sample_id] = outcome.action

    eligible = tuple(
        record
        for record in accepted
        if actions.get(record.sample_id, SampleAction.KEEP) is SampleAction.KEEP
    )
    actual_counts = _source_counts(eligible)
    if actual_counts != exact_result.eligible_counts:
        raise SplitError("exact eligible counts do not match the inventory records")
    return eligible


def _records_by_class(
    records: tuple[InventoryRecord, ...], *, partition: str, minimum: int
) -> dict[str, tuple[InventoryRecord, ...]]:
    grouped: dict[str, list[InventoryRecord]] = defaultdict(list)
    for record in records:
        if record.class_name not in CLASS_IDS:
            raise SplitError(f"eligible sample has invalid class: {record.sample_id}")
        if record.class_id != CLASS_IDS[record.class_name]:
            raise SplitError(f"eligible sample has inconsistent class ID: {record.sample_id}")
        if not record.relative_path or not record.sha256:
            raise SplitError(f"eligible sample lacks path or SHA-256: {record.sample_id}")
        grouped[record.class_name].append(record)

    insufficient = {
        class_name: len(grouped[class_name])
        for class_name in CLASS_IDS
        if len(grouped[class_name]) < minimum
    }
    if insufficient:
        raise SplitError(
            f"source {partition} has empty or insufficient classes: {insufficient}"
        )
    return {
        class_name: tuple(grouped[class_name])
        for class_name in CLASS_IDS
    }


def _selection_key(sample_id: str, *, seed: int) -> tuple[bytes, str]:
    encoded = sample_id.encode("utf-8")
    digest = sha256()
    digest.update(_SPLIT_KEY_DOMAIN)
    digest.update(seed.to_bytes(8, byteorder="big", signed=False))
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)
    return digest.digest(), sample_id


def _validation_count(count: int, fraction: float) -> int:
    estimate = int(count * fraction + 0.5)
    return min(max(estimate, 1), count - 1)


def _assignment(record: InventoryRecord, *, split: DatasetSplit) -> SplitAssignment:
    if (
        record.class_name not in CLASS_IDS
        or record.class_id is None
        or record.source_partition not in {"train", "test"}
    ):
        raise SplitError(f"invalid eligible record: {record.sample_id}")
    return SplitAssignment(
        sample_id=record.sample_id,
        relative_path=record.relative_path,
        sha256=record.sha256,
        class_name=record.class_name,
        class_id=record.class_id,
        source_partition=record.source_partition,
        split=split,
    )


def _validate_result(
    result: DatasetSplitResult,
    *,
    source_train: tuple[InventoryRecord, ...],
    source_test: tuple[InventoryRecord, ...],
) -> None:
    assignment_ids = [assignment.sample_id for assignment in result.assignments]
    if len(assignment_ids) != len(set(assignment_ids)):
        raise SplitError("a sample_id was assigned more than once")
    source_ids = {record.sample_id for record in (*source_train, *source_test)}
    if set(assignment_ids) != source_ids:
        raise SplitError("split assignments do not preserve all eligible sample IDs")

    original_test_ids = {record.sample_id for record in source_test}
    assigned_test_ids = {
        assignment.sample_id
        for assignment in result.assignments
        if assignment.split is DatasetSplit.TEST
    }
    if assigned_test_ids != original_test_ids:
        raise SplitError("original test sample IDs were not preserved exactly")
    if any(
        assignment.source_partition != "train"
        for assignment in result.assignments
        if assignment.split is DatasetSplit.VALIDATION
    ):
        raise SplitError("validation contains a sample outside source train")


def _source_counts(
    records: tuple[InventoryRecord, ...],
) -> dict[str, dict[str, int]]:
    counts = Counter(
        (record.source_partition or "none", record.class_name or "unidentified")
        for record in records
    )
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for (partition, class_name), count in sorted(counts.items()):
        nested[partition][class_name] = count
    return {
        partition: dict(sorted(class_counts.items()))
        for partition, class_counts in sorted(nested.items())
    }
