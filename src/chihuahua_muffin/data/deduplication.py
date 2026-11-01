"""Exact SHA-256 deduplication and preliminary validation analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .canonical import CanonicalizationError, canonical_json_bytes
from .inventory import AuditStatus, InventoryRecord, InventoryResult
from .review import (
    HumanReviewDecision,
    ReviewDecisionState,
    ReviewError,
    ReviewOccurrenceType,
    canonical_human_decisions_bytes,
)


class DeduplicationError(ValueError):
    """Raised when exact-deduplication inputs or decisions are inconsistent."""


class SampleAction(StrEnum):
    """Logical action only; source files are never modified or removed."""

    KEEP = "keep"
    EXCLUDE = "exclude"
    BLOCK = "block"


class ExactOccurrenceType(StrEnum):
    """Classification of one repeated SHA-256 group."""

    SAME_LABEL = "same_label"
    LABEL_CONFLICT = "label_conflict"


@dataclass(frozen=True, slots=True)
class SampleOutcome:
    """Logical treatment of one occurrence in an exact duplicate group."""

    sample_id: str
    relative_path: str
    class_name: str
    source_partition: str | None
    action: SampleAction
    reason: str

    def as_json(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "relative_path": self.relative_path,
            "class_name": self.class_name,
            "source_partition": self.source_partition,
            "action": self.action.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ExactDuplicateGroup:
    """Deterministic treatment of all accepted images sharing one SHA-256."""

    group_id: str
    sha256: str
    occurrence_type: ExactOccurrenceType
    class_names: tuple[str, ...]
    source_partitions: tuple[str, ...]
    crosses_source_partitions: bool
    outcomes: tuple[SampleOutcome, ...]
    human_decision: ReviewDecisionState | None
    human_reviewer: str | None
    blocking: bool

    def as_json(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "sha256": self.sha256,
            "occurrence_type": self.occurrence_type.value,
            "class_names": list(self.class_names),
            "source_partitions": list(self.source_partitions),
            "crosses_source_partitions": self.crosses_source_partitions,
            "outcomes": [outcome.as_json() for outcome in self.outcomes],
            "human_decision": (
                self.human_decision.value if self.human_decision is not None else None
            ),
            "human_reviewer": self.human_reviewer,
            "blocking": self.blocking,
        }


@dataclass(frozen=True, slots=True)
class ExactDeduplicationResult:
    """Non-destructive exact-deduplication decisions over an inventory."""

    dataset_version_id: str
    groups: tuple[ExactDuplicateGroup, ...]
    input_accepted_counts: dict[str, dict[str, int]]
    eligible_counts: dict[str, dict[str, int]]
    excluded_counts: dict[str, dict[str, int]]
    blocked_counts: dict[str, dict[str, int]]
    blocking_review_ids: tuple[str, ...]

    @property
    def training_release_blocked(self) -> bool:
        return bool(self.blocking_review_ids)

    @property
    def excluded_sample_count(self) -> int:
        return _nested_total(self.excluded_counts)

    @property
    def blocked_sample_count(self) -> int:
        return _nested_total(self.blocked_counts)

    def payload(self) -> dict[str, object]:
        same_label_groups = sum(
            group.occurrence_type is ExactOccurrenceType.SAME_LABEL
            for group in self.groups
        )
        conflict_groups = len(self.groups) - same_label_groups
        return {
            "artifact_type": "exact_deduplication_audit",
            "artifact_version": "1.0",
            "dataset_version_id": self.dataset_version_id,
            "method": "sha256_original_bytes",
            "group_count": len(self.groups),
            "same_label_group_count": same_label_groups,
            "label_conflict_group_count": conflict_groups,
            "cross_partition_group_count": sum(
                group.crosses_source_partitions for group in self.groups
            ),
            "groups": [group.as_json() for group in self.groups],
            "input_accepted_counts": self.input_accepted_counts,
            "eligible_counts": self.eligible_counts,
            "excluded_counts": self.excluded_counts,
            "blocked_counts": self.blocked_counts,
            "excluded_sample_count": self.excluded_sample_count,
            "blocked_sample_count": self.blocked_sample_count,
            "blocking_review_ids": list(self.blocking_review_ids),
            "training_release_blocked": self.training_release_blocked,
            "originals_modified": False,
            "limitations": [
                "Only byte-identical SHA-256 groups are analyzed.",
                "Perceptual duplicate detection is not implemented.",
                "No source file, label, partition, or final split is modified.",
            ],
        }


@dataclass(frozen=True, slots=True)
class ValidationFeasibilityAssessment:
    """Preliminary counts only; this object never assigns validation samples."""

    dataset_version_id: str
    assessment: str
    final_scenario_decision: str
    source_train_preserved: bool
    source_test_preserved: bool
    candidate_validation_fraction_of_source_train: float
    seed_reserved_for_future_split: int
    eligible_source_train_counts: dict[str, int]
    eligible_source_test_counts: dict[str, int]
    estimated_candidate_validation_counts: dict[str, int]
    estimated_remaining_train_counts: dict[str, int]
    exact_duplicate_group_count: int
    cross_partition_exact_group_count: int
    exact_excluded_sample_count: int
    exact_blocked_sample_count: int
    risks: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return {
            "artifact_type": "preliminary_validation_feasibility",
            "artifact_version": "1.0",
            "dataset_version_id": self.dataset_version_id,
            "assessment": self.assessment,
            "final_scenario_decision": self.final_scenario_decision,
            "source_train_preserved": self.source_train_preserved,
            "source_test_preserved": self.source_test_preserved,
            "candidate_validation_fraction_of_source_train": (
                self.candidate_validation_fraction_of_source_train
            ),
            "seed_reserved_for_future_split": self.seed_reserved_for_future_split,
            "eligible_source_train_counts": self.eligible_source_train_counts,
            "eligible_source_test_counts": self.eligible_source_test_counts,
            "estimated_candidate_validation_counts": (
                self.estimated_candidate_validation_counts
            ),
            "estimated_remaining_train_counts": self.estimated_remaining_train_counts,
            "exact_duplicate_group_count": self.exact_duplicate_group_count,
            "cross_partition_exact_group_count": (
                self.cross_partition_exact_group_count
            ),
            "exact_excluded_sample_count": self.exact_excluded_sample_count,
            "exact_blocked_sample_count": self.exact_blocked_sample_count,
            "split_created": False,
            "risks": list(self.risks),
        }


def analyze_exact_duplicates(
    inventory: InventoryResult,
    *,
    human_decisions: tuple[HumanReviewDecision, ...] = (),
) -> ExactDeduplicationResult:
    """Group accepted images by SHA-256 and assign logical outcomes."""

    decisions_by_id: dict[str, HumanReviewDecision] = {}
    for decision in human_decisions:
        if decision.review_id in decisions_by_id:
            raise DeduplicationError(f"duplicate review_id: {decision.review_id}")
        decisions_by_id[decision.review_id] = decision

    accepted = tuple(
        record
        for record in inventory.records
        if record.status is AuditStatus.ACCEPTED and record.image_format is not None
    )
    by_hash: dict[str, list[InventoryRecord]] = defaultdict(list)
    for record in accepted:
        if not record.sha256:
            raise DeduplicationError(
                f"accepted sample has no SHA-256: {record.sample_id}"
            )
        by_hash[record.sha256].append(record)

    groups: list[ExactDuplicateGroup] = []
    consumed_decisions: set[str] = set()
    for file_hash, records in sorted(by_hash.items()):
        if len(records) < 2:
            continue
        ordered = tuple(sorted(records, key=_preservation_order))
        group_id = f"exact-sha256:{file_hash}"
        class_names = tuple(sorted({record.class_name or "unidentified" for record in ordered}))
        partitions = tuple(
            sorted({record.source_partition or "none" for record in ordered})
        )
        occurrence_type = (
            ExactOccurrenceType.SAME_LABEL
            if len(class_names) == 1
            else ExactOccurrenceType.LABEL_CONFLICT
        )
        decision = decisions_by_id.get(group_id)
        if decision is not None:
            consumed_decisions.add(group_id)
        if occurrence_type is ExactOccurrenceType.SAME_LABEL:
            if decision is not None:
                raise DeduplicationError(
                    "same-label exact duplicates use the approved automatic policy"
                )
            outcomes = _automatic_same_label_outcomes(ordered)
            human_state = None
            reviewer = None
        else:
            outcomes = _conflict_outcomes(group_id, ordered, decision)
            human_state = decision.decision if decision is not None else None
            reviewer = decision.reviewer if decision is not None else None
        groups.append(
            ExactDuplicateGroup(
                group_id=group_id,
                sha256=file_hash,
                occurrence_type=occurrence_type,
                class_names=class_names,
                source_partitions=partitions,
                crosses_source_partitions=len(partitions) > 1,
                outcomes=outcomes,
                human_decision=human_state,
                human_reviewer=reviewer,
                blocking=any(
                    outcome.action is SampleAction.BLOCK for outcome in outcomes
                ),
            )
        )

    unused_decisions = sorted(set(decisions_by_id) - consumed_decisions)
    if unused_decisions:
        raise DeduplicationError(
            f"human decisions do not match exact conflict groups: {unused_decisions}"
        )

    action_by_sample = {
        outcome.sample_id: outcome.action
        for group in groups
        for outcome in group.outcomes
    }
    eligible = tuple(
        record
        for record in accepted
        if action_by_sample.get(record.sample_id, SampleAction.KEEP)
        is SampleAction.KEEP
    )
    excluded = tuple(
        record
        for record in accepted
        if action_by_sample.get(record.sample_id) is SampleAction.EXCLUDE
    )
    blocked = tuple(
        record
        for record in accepted
        if action_by_sample.get(record.sample_id) is SampleAction.BLOCK
    )
    blocking_ids = tuple(sorted(group.group_id for group in groups if group.blocking))
    return ExactDeduplicationResult(
        dataset_version_id=inventory.dataset_version_id,
        groups=tuple(groups),
        input_accepted_counts=_counts_by_partition_and_class(accepted),
        eligible_counts=_counts_by_partition_and_class(eligible),
        excluded_counts=_counts_by_partition_and_class(excluded),
        blocked_counts=_counts_by_partition_and_class(blocked),
        blocking_review_ids=blocking_ids,
    )


def assess_validation_feasibility(
    result: ExactDeduplicationResult,
    *,
    candidate_fraction: float = 0.15,
    seed: int = 42,
) -> ValidationFeasibilityAssessment:
    """Estimate counts without choosing or moving a single sample."""

    if type(candidate_fraction) is not float or not 0.0 < candidate_fraction < 1.0:
        raise DeduplicationError("candidate_fraction must be a float between 0 and 1")
    if type(seed) is not int or seed != 42:
        raise DeduplicationError("seed must be the approved value 42")
    train_counts = dict(result.eligible_counts.get("train", {}))
    test_counts = dict(result.eligible_counts.get("test", {}))
    expected_classes = {"chihuahua", "muffin"}
    has_class_support = all(train_counts.get(name, 0) >= 2 for name in expected_classes)
    estimated_validation = {
        name: _estimated_validation_count(count, candidate_fraction)
        for name, count in sorted(train_counts.items())
    }
    remaining_train = {
        name: count - estimated_validation[name]
        for name, count in sorted(train_counts.items())
    }
    risks = [
        "Perceptual duplicate analysis is pending and may change group boundaries.",
        "Train/test independence is not established by exact hashes alone.",
        "No validation membership is assigned before the Increment 7 decision.",
    ]
    if result.training_release_blocked:
        assessment = "blocked_pending_exact_human_review"
        risks.append("Blocking exact-duplicate reviews remain unresolved.")
    elif not has_class_support:
        assessment = "insufficient_class_support_after_exact_deduplication"
        risks.append("At least one class lacks enough eligible source-train samples.")
    else:
        assessment = "exact_stage_supports_candidate_validation_analysis"
    return ValidationFeasibilityAssessment(
        dataset_version_id=result.dataset_version_id,
        assessment=assessment,
        final_scenario_decision="deferred_until_perceptual_audit",
        source_train_preserved=True,
        source_test_preserved=True,
        candidate_validation_fraction_of_source_train=candidate_fraction,
        seed_reserved_for_future_split=seed,
        eligible_source_train_counts=dict(sorted(train_counts.items())),
        eligible_source_test_counts=dict(sorted(test_counts.items())),
        estimated_candidate_validation_counts=estimated_validation,
        estimated_remaining_train_counts=remaining_train,
        exact_duplicate_group_count=len(result.groups),
        cross_partition_exact_group_count=sum(
            group.crosses_source_partitions for group in result.groups
        ),
        exact_excluded_sample_count=result.excluded_sample_count,
        exact_blocked_sample_count=result.blocked_sample_count,
        risks=tuple(risks),
    )


def write_exact_audit(
    result: ExactDeduplicationResult,
    assessment: ValidationFeasibilityAssessment,
    decisions: tuple[HumanReviewDecision, ...],
    destination: str | Path,
) -> tuple[Path, Path, Path]:
    """Write three non-final local artifacts without overwriting a destination."""

    if assessment.dataset_version_id != result.dataset_version_id:
        raise DeduplicationError("assessment belongs to a different dataset version")
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise DeduplicationError(f"exact audit destination already exists: {target}")
    deduplication_bytes = canonical_json_bytes(result.payload())
    feasibility_bytes = canonical_json_bytes(assessment.payload())
    try:
        decisions_bytes = canonical_human_decisions_bytes(
            result.dataset_version_id, decisions
        )
    except (ReviewError, CanonicalizationError) as error:
        raise DeduplicationError(f"cannot serialize human decisions: {error}") from error
    target.mkdir(parents=True, exist_ok=False)
    deduplication_path = target / "exact_deduplication.json"
    decisions_path = target / "human_decisions.json"
    feasibility_path = target / "validation_feasibility.json"
    deduplication_path.write_bytes(deduplication_bytes)
    decisions_path.write_bytes(decisions_bytes)
    feasibility_path.write_bytes(feasibility_bytes)
    return deduplication_path, decisions_path, feasibility_path


def _automatic_same_label_outcomes(
    records: tuple[InventoryRecord, ...],
) -> tuple[SampleOutcome, ...]:
    preserved = records[0]
    outcomes = [
        _outcome(
            preserved,
            SampleAction.KEEP,
            "deterministic_exact_duplicate_representative",
        )
    ]
    for record in records[1:]:
        reason = (
            "exact_duplicate_of_preserved_test"
            if preserved.source_partition == "test"
            and record.source_partition != "test"
            else "exact_duplicate_redundant_same_label"
        )
        outcomes.append(_outcome(record, SampleAction.EXCLUDE, reason))
    return tuple(sorted(outcomes, key=lambda outcome: outcome.sample_id))


def _conflict_outcomes(
    group_id: str,
    records: tuple[InventoryRecord, ...],
    decision: HumanReviewDecision | None,
) -> tuple[SampleOutcome, ...]:
    target_ids = tuple(sorted(record.sample_id for record in records))
    if decision is None:
        return tuple(
            _outcome(record, SampleAction.BLOCK, "label_conflict_requires_human_review")
            for record in records
        )
    if decision.review_id != group_id:
        raise DeduplicationError("human decision review_id does not match group")
    if decision.occurrence_type is not ReviewOccurrenceType.EXACT_LABEL_CONFLICT:
        raise DeduplicationError("human decision occurrence_type does not match group")
    if decision.target_sample_ids != target_ids:
        raise DeduplicationError("human decision targets do not match group samples")

    if decision.decision is ReviewDecisionState.UNCERTAIN:
        return tuple(
            _outcome(record, SampleAction.BLOCK, "human_review_uncertain")
            for record in records
        )
    if decision.decision is ReviewDecisionState.NOT_DUPLICATE:
        return tuple(
            _outcome(
                record,
                SampleAction.BLOCK,
                "not_duplicate_decision_conflicts_with_identical_sha256",
            )
            for record in records
        )
    if not decision.has_resolved_disposition:
        return tuple(
            _outcome(
                record,
                SampleAction.BLOCK,
                "confirmed_duplicate_requires_sample_disposition",
            )
            for record in records
        )
    retained = set(decision.retained_sample_ids)
    excluded = set(decision.excluded_sample_ids)
    outcomes = []
    for record in records:
        if record.sample_id in retained:
            outcomes.append(
                _outcome(record, SampleAction.KEEP, "human_review_retained")
            )
        elif record.sample_id in excluded:
            outcomes.append(
                _outcome(record, SampleAction.EXCLUDE, "human_review_excluded")
            )
        else:
            raise DeduplicationError("human decision disposition is incomplete")
    return tuple(sorted(outcomes, key=lambda outcome: outcome.sample_id))


def _outcome(
    record: InventoryRecord,
    action: SampleAction,
    reason: str,
) -> SampleOutcome:
    if record.class_name is None:
        raise DeduplicationError("accepted duplicate sample has no class")
    return SampleOutcome(
        sample_id=record.sample_id,
        relative_path=record.relative_path,
        class_name=record.class_name,
        source_partition=record.source_partition,
        action=action,
        reason=reason,
    )


def _preservation_order(record: InventoryRecord) -> tuple[int, str, str]:
    partition_priority = {"test": 0, "train": 1, "validation": 2, None: 3}
    return (
        partition_priority.get(record.source_partition, 4),
        record.relative_path,
        record.sample_id,
    )


def _counts_by_partition_and_class(
    records: tuple[InventoryRecord, ...],
) -> dict[str, dict[str, int]]:
    counts = Counter(
        (record.source_partition or "none", record.class_name or "unidentified")
        for record in records
    )
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for (partition, class_name), count in sorted(counts.items()):
        nested[partition][class_name] = count
    return {partition: dict(sorted(values.items())) for partition, values in sorted(nested.items())}


def _nested_total(values: dict[str, dict[str, int]]) -> int:
    return sum(count for partition in values.values() for count in partition.values())


def _estimated_validation_count(count: int, fraction: float) -> int:
    if count < 2:
        return 0
    estimate = int(count * fraction + 0.5)
    return min(max(estimate, 1), count - 1)
