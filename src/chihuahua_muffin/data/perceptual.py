"""Perceptual duplicate triage for SPEC 01 Increment 6.

pHash distance is evidence for review, never an automatic duplicate decision.
This module does not modify source images, labels, partitions, or final splits.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import StrEnum
from html import escape
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import imagehash
import numpy as np
from PIL import Image, ImageOps

from .canonical import canonical_json_bytes, canonical_json_sha256
from .deduplication import (
    ExactDeduplicationResult,
    SampleAction,
)
from .inventory import (
    AuditStatus,
    GRAYSCALE_MODES,
    InventoryRecord,
    InventoryResult,
    sha256_file,
)
from .review import (
    HumanReviewDecision,
    ReviewDecisionState,
    ReviewError,
    ReviewOccurrenceType,
    canonical_human_decisions_bytes,
)


PHASH_HASH_SIZE = 8
PHASH_HIGHFREQ_FACTOR = 4
PHASH_BITS = PHASH_HASH_SIZE * PHASH_HASH_SIZE
DEFAULT_BLOCK_SIZE = 256
MAX_REVIEW_CANDIDATES = 100_000

_POPCOUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


class PerceptualAuditError(ValueError):
    """Raised when perceptual-audit inputs or artifacts are inconsistent."""


class ComparisonScope(StrEnum):
    """Relationship between the original partitions of a compared pair."""

    WITHIN_TRAIN = "within_train"
    TRAIN_TEST = "train_test"
    WITHIN_TEST = "within_test"
    INVOLVING_OTHER = "involving_other_partition"


@dataclass(frozen=True, slots=True)
class PerceptualHashRecord:
    """One exact-stage-eligible image and its candidate 64-bit pHash."""

    sample_id: str
    relative_path: str
    sha256: str
    class_name: str
    source_partition: str | None
    width: int
    height: int
    phash: str

    def as_json(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "class_name": self.class_name,
            "source_partition": self.source_partition,
            "width": self.width,
            "height": self.height,
            "phash": self.phash,
        }


@dataclass(frozen=True, slots=True)
class PerceptualCandidatePair:
    """A low-distance pair queued for review without an automatic outcome."""

    review_id: str
    distance: int
    comparison_scope: ComparisonScope
    label_conflict: bool
    left: PerceptualHashRecord
    right: PerceptualHashRecord
    human_decision: HumanReviewDecision | None = None

    @property
    def occurrence_type(self) -> ReviewOccurrenceType:
        if self.label_conflict:
            return ReviewOccurrenceType.PERCEPTUAL_LABEL_CONFLICT
        return ReviewOccurrenceType.PERCEPTUAL_SAME_LABEL

    @property
    def target_sample_ids(self) -> tuple[str, str]:
        values = tuple(sorted((self.left.sample_id, self.right.sample_id)))
        return values[0], values[1]

    def as_json(self) -> dict[str, object]:
        return {
            "review_id": self.review_id,
            "distance": self.distance,
            "comparison_scope": self.comparison_scope.value,
            "label_conflict": self.label_conflict,
            "occurrence_type": self.occurrence_type.value,
            "left": self.left.as_json(),
            "right": self.right.as_json(),
            "human_decision": (
                self.human_decision.as_json()
                if self.human_decision is not None
                else None
            ),
            "automatic_duplicate_confirmation": False,
            "automatic_exclusion": False,
        }


@dataclass(frozen=True, slots=True)
class PerceptualAuditResult:
    """Deterministic all-pairs distance profile and bounded review queue."""

    dataset_version_id: str
    records: tuple[PerceptualHashRecord, ...]
    pair_comparison_count: int
    distance_histogram: tuple[int, ...]
    distance_histograms_by_scope: dict[ComparisonScope, tuple[int, ...]]
    same_label_distance_histogram: tuple[int, ...]
    conflicting_label_distance_histogram: tuple[int, ...]
    nearest_neighbor_distance_histogram: tuple[int, ...]
    unpaired_image_count: int
    candidate_max_distance: int
    candidates: tuple[PerceptualCandidatePair, ...]
    block_size: int
    exact_stage_excluded_count: int
    exact_stage_blocked_count: int

    @property
    def images_analyzed(self) -> int:
        return len(self.records)

    @property
    def within_train_candidate_count(self) -> int:
        return sum(
            pair.comparison_scope is ComparisonScope.WITHIN_TRAIN
            for pair in self.candidates
        )

    @property
    def train_test_candidate_count(self) -> int:
        return sum(
            pair.comparison_scope is ComparisonScope.TRAIN_TEST
            for pair in self.candidates
        )

    @property
    def label_conflict_candidate_count(self) -> int:
        return sum(pair.label_conflict for pair in self.candidates)

    @property
    def reviewed_candidate_count(self) -> int:
        return sum(pair.human_decision is not None for pair in self.candidates)

    def payload(self) -> dict[str, object]:
        decision_counts = Counter(
            pair.human_decision.decision.value
            for pair in self.candidates
            if pair.human_decision is not None
        )
        return {
            "artifact_type": "perceptual_duplicate_audit",
            "artifact_version": "1.0",
            "dataset_version_id": self.dataset_version_id,
            "method": {
                "name": "phash",
                "purpose": "candidate_triage_only",
                "library": "ImageHash",
                "library_version": version("ImageHash"),
                "hash_size": PHASH_HASH_SIZE,
                "highfreq_factor": PHASH_HIGHFREQ_FACTOR,
                "hash_bits": PHASH_BITS,
                "distance": "hamming",
                "image_normalization": [
                    "read_only_source",
                    "exif_transpose",
                    "RGB_preserved",
                    "grayscale_replicated_to_RGB",
                    "RGBA_composited_on_white",
                    "ImageHash_phash_internal_luminance_and_DCT",
                ],
            },
            "runtime_versions": {
                "ImageHash": version("ImageHash"),
                "NumPy": version("numpy"),
                "Pillow": version("Pillow"),
                "PyWavelets": version("PyWavelets"),
                "SciPy": version("scipy"),
            },
            "comparison_strategy": {
                "name": "blocked_vectorized_upper_triangle",
                "time_complexity": "O(n^2)",
                "working_memory": "O(block_size^2)",
                "block_size": self.block_size,
                "rationale": (
                    "The complete upper triangle is practical for 5,917 images "
                    "and provides the full empirical distance distribution."
                ),
            },
            "images_analyzed": self.images_analyzed,
            "pair_comparison_count": self.pair_comparison_count,
            "source_sha256_verified_before_hashing": True,
            "distance_distribution": _histogram_payload(self.distance_histogram),
            "distance_distributions_by_scope": {
                scope.value: _histogram_payload(
                    self.distance_histograms_by_scope[scope]
                )
                for scope in ComparisonScope
            },
            "same_label_distance_distribution": _histogram_payload(
                self.same_label_distance_histogram
            ),
            "conflicting_label_distance_distribution": _histogram_payload(
                self.conflicting_label_distance_histogram
            ),
            "nearest_neighbor_distance_distribution": _histogram_payload(
                self.nearest_neighbor_distance_histogram
            ),
            "unpaired_image_count": self.unpaired_image_count,
            "candidate_screening": {
                "maximum_hamming_distance": self.candidate_max_distance,
                "is_duplicate_threshold": False,
                "requires_human_review": True,
                "automatic_removal": False,
            },
            "candidate_pair_count": len(self.candidates),
            "within_train_candidate_count": self.within_train_candidate_count,
            "train_test_candidate_count": self.train_test_candidate_count,
            "label_conflict_candidate_count": (
                self.label_conflict_candidate_count
            ),
            "reviewed_candidate_count": self.reviewed_candidate_count,
            "human_decision_counts": dict(sorted(decision_counts.items())),
            "records": [record.as_json() for record in self.records],
            "candidates": [pair.as_json() for pair in self.candidates],
            "exact_stage_excluded_count": self.exact_stage_excluded_count,
            "exact_stage_blocked_count": self.exact_stage_blocked_count,
            "originals_modified": False,
            "labels_modified": False,
            "partitions_modified": False,
            "split_created": False,
            "limitations": [
                "pHash uses luminance structure and can miss crops or major edits.",
                "Low Hamming distance can join different but similarly composed images.",
                "High Hamming distance does not prove that photographs are independent.",
                "Pairwise candidates are not duplicate groups until human review.",
                "No candidate is confirmed, excluded, or relabeled automatically.",
            ],
        }


@dataclass(frozen=True, slots=True)
class PerceptualValidationAssessment:
    """Feasibility evidence only; it never selects a scenario or creates a split."""

    dataset_version_id: str
    assessment: str
    scenario_decision: str
    software_recommendation: str
    candidate_validation_fraction_of_source_train: float
    seed_reserved_for_future_split: int
    eligible_source_train_counts: dict[str, int]
    eligible_source_test_counts: dict[str, int]
    estimated_candidate_validation_counts: dict[str, int]
    estimated_remaining_train_counts: dict[str, int]
    perceptual_candidate_pair_count: int
    within_train_candidate_count: int
    train_test_candidate_count: int
    label_conflict_candidate_count: int
    reviewed_candidate_count: int
    unresolved_candidate_count: int
    confirmed_duplicate_pair_count: int
    risks: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return {
            "artifact_type": "perceptual_validation_feasibility",
            "artifact_version": "1.0",
            "dataset_version_id": self.dataset_version_id,
            "assessment": self.assessment,
            "scenario_decision": self.scenario_decision,
            "software_recommendation": self.software_recommendation,
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
            "perceptual_candidate_pair_count": self.perceptual_candidate_pair_count,
            "within_train_candidate_count": self.within_train_candidate_count,
            "train_test_candidate_count": self.train_test_candidate_count,
            "label_conflict_candidate_count": self.label_conflict_candidate_count,
            "reviewed_candidate_count": self.reviewed_candidate_count,
            "unresolved_candidate_count": self.unresolved_candidate_count,
            "confirmed_duplicate_pair_count": self.confirmed_duplicate_pair_count,
            "split_created": False,
            "scenario_selected": False,
            "risks": list(self.risks),
        }


def analyze_perceptual_duplicates(
    source_root: str | Path,
    inventory: InventoryResult,
    exact_result: ExactDeduplicationResult,
    *,
    candidate_max_distance: int,
    human_decisions: tuple[HumanReviewDecision, ...] = (),
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> PerceptualAuditResult:
    """Hash exact-stage-eligible images and profile every unique pair."""

    records = compute_perceptual_hashes(source_root, inventory, exact_result)
    return analyze_perceptual_hashes(
        records,
        dataset_version_id=inventory.dataset_version_id,
        candidate_max_distance=candidate_max_distance,
        human_decisions=human_decisions,
        block_size=block_size,
        exact_stage_excluded_count=exact_result.excluded_sample_count,
        exact_stage_blocked_count=exact_result.blocked_sample_count,
    )


def compute_perceptual_hashes(
    source_root: str | Path,
    inventory: InventoryResult,
    exact_result: ExactDeduplicationResult,
) -> tuple[PerceptualHashRecord, ...]:
    """Compute pHash after deterministic read-only normalization."""

    if exact_result.dataset_version_id != inventory.dataset_version_id:
        raise PerceptualAuditError(
            "exact result belongs to a different dataset version"
        )
    root = Path(source_root)
    if root.is_symlink() or not root.is_dir():
        raise PerceptualAuditError(
            "source_root must be an existing non-symbolic-link directory"
        )
    resolved_root = root.resolve()
    exact_actions = {
        outcome.sample_id: outcome.action
        for group in exact_result.groups
        for outcome in group.outcomes
    }
    records: list[PerceptualHashRecord] = []
    for record in sorted(inventory.records, key=lambda item: item.sample_id):
        if record.status is not AuditStatus.ACCEPTED:
            continue
        if exact_actions.get(record.sample_id, SampleAction.KEEP) is not SampleAction.KEEP:
            continue
        _validate_eligible_record(record)
        path = _source_path(resolved_root, record.relative_path)
        if path.is_symlink() or not path.is_file():
            raise PerceptualAuditError(
                f"eligible source is missing or symbolic: {record.relative_path}"
            )
        if sha256_file(path) != record.sha256:
            raise PerceptualAuditError(
                f"source changed after inventory: {record.relative_path}"
            )
        normalized = _normalized_image_for_phash(path, record)
        try:
            perceptual_hash = imagehash.phash(
                normalized,
                hash_size=PHASH_HASH_SIZE,
                highfreq_factor=PHASH_HIGHFREQ_FACTOR,
            )
        finally:
            normalized.close()
        records.append(
            PerceptualHashRecord(
                sample_id=record.sample_id,
                relative_path=record.relative_path,
                sha256=record.sha256,
                class_name=record.class_name or "",
                source_partition=record.source_partition,
                width=record.width or 0,
                height=record.height or 0,
                phash=str(perceptual_hash),
            )
        )
    return tuple(records)


def analyze_perceptual_hashes(
    records: tuple[PerceptualHashRecord, ...],
    *,
    dataset_version_id: str,
    candidate_max_distance: int,
    human_decisions: tuple[HumanReviewDecision, ...] = (),
    block_size: int = DEFAULT_BLOCK_SIZE,
    exact_stage_excluded_count: int = 0,
    exact_stage_blocked_count: int = 0,
) -> PerceptualAuditResult:
    """Profile precomputed hashes; useful for deterministic tests and reruns."""

    if type(dataset_version_id) is not str or not dataset_version_id.strip():
        raise PerceptualAuditError("dataset_version_id must be a non-empty string")
    if type(candidate_max_distance) is not int or not 0 <= candidate_max_distance <= 64:
        raise PerceptualAuditError(
            "candidate_max_distance must be an integer between 0 and 64"
        )
    if type(block_size) is not int or block_size < 1:
        raise PerceptualAuditError("block_size must be a positive integer")
    ordered = tuple(sorted(records, key=lambda record: record.sample_id))
    sample_ids = [record.sample_id for record in ordered]
    if len(sample_ids) != len(set(sample_ids)):
        raise PerceptualAuditError("perceptual records must have unique sample IDs")
    for record in ordered:
        if len(record.phash) != PHASH_BITS // 4:
            raise PerceptualAuditError("pHash must contain exactly 64 bits")
        try:
            int(record.phash, 16)
        except ValueError as error:
            raise PerceptualAuditError("pHash must be lowercase hexadecimal") from error
        if record.phash != record.phash.lower():
            raise PerceptualAuditError("pHash must be lowercase hexadecimal")

    profile = _profile_all_pairs(
        ordered,
        candidate_max_distance=candidate_max_distance,
        block_size=block_size,
    )
    candidates = tuple(
        _candidate_pair(ordered[left], ordered[right], distance)
        for distance, left, right in profile.candidate_coordinates
    )
    decision_map = _validate_perceptual_decisions(candidates, human_decisions)
    candidates = tuple(
        replace(pair, human_decision=decision_map.get(pair.review_id))
        for pair in candidates
    )
    return PerceptualAuditResult(
        dataset_version_id=dataset_version_id,
        records=ordered,
        pair_comparison_count=profile.pair_comparison_count,
        distance_histogram=profile.distance_histogram,
        distance_histograms_by_scope=profile.distance_histograms_by_scope,
        same_label_distance_histogram=profile.same_label_distance_histogram,
        conflicting_label_distance_histogram=(
            profile.conflicting_label_distance_histogram
        ),
        nearest_neighbor_distance_histogram=(
            profile.nearest_neighbor_distance_histogram
        ),
        unpaired_image_count=profile.unpaired_image_count,
        candidate_max_distance=candidate_max_distance,
        candidates=candidates,
        block_size=block_size,
        exact_stage_excluded_count=exact_stage_excluded_count,
        exact_stage_blocked_count=exact_stage_blocked_count,
    )


def assess_perceptual_validation_feasibility(
    exact_result: ExactDeduplicationResult,
    result: PerceptualAuditResult,
    *,
    candidate_fraction: float = 0.15,
    seed: int = 42,
) -> PerceptualValidationAssessment:
    """Summarize feasibility without assigning samples or choosing A/B."""

    if exact_result.dataset_version_id != result.dataset_version_id:
        raise PerceptualAuditError("audit stages belong to different dataset versions")
    if type(candidate_fraction) is not float or not 0.0 < candidate_fraction < 1.0:
        raise PerceptualAuditError("candidate_fraction must be a float between 0 and 1")
    if type(seed) is not int or seed != 42:
        raise PerceptualAuditError("seed must be the approved value 42")
    train_counts = dict(sorted(exact_result.eligible_counts.get("train", {}).items()))
    test_counts = dict(sorted(exact_result.eligible_counts.get("test", {}).items()))
    validation_counts = {
        name: _estimated_validation_count(count, candidate_fraction)
        for name, count in train_counts.items()
    }
    remaining_counts = {
        name: count - validation_counts[name]
        for name, count in train_counts.items()
    }
    confirmed = sum(
        pair.human_decision is not None
        and pair.human_decision.decision is ReviewDecisionState.CONFIRMED_DUPLICATE
        for pair in result.candidates
    )
    unresolved = sum(pair.human_decision is None for pair in result.candidates)
    class_support = all(train_counts.get(name, 0) >= 2 for name in ("chihuahua", "muffin"))
    risks = [
        "Candidate pairs require visual review; pHash distance is not proof.",
        "Confirmed pairs must become indivisible groups before any future split.",
        "No train/validation membership is assigned in Increment 6.",
        "The original test partition must remain reserved from model selection.",
    ]
    if exact_result.training_release_blocked:
        assessment = "blocked_by_exact_human_review"
        risks.append("Exact-duplicate blocking reviews remain unresolved.")
    elif not class_support:
        assessment = "insufficient_train_class_support"
        risks.append("At least one class lacks enough eligible train samples.")
    elif unresolved:
        assessment = "quantitatively_feasible_pending_perceptual_review"
        risks.append("Perceptual candidate decisions remain unresolved.")
    else:
        assessment = "quantitatively_feasible_after_recorded_reviews"
    return PerceptualValidationAssessment(
        dataset_version_id=result.dataset_version_id,
        assessment=assessment,
        scenario_decision="awaiting_human_decision",
        software_recommendation="not_selected_automatically",
        candidate_validation_fraction_of_source_train=candidate_fraction,
        seed_reserved_for_future_split=seed,
        eligible_source_train_counts=train_counts,
        eligible_source_test_counts=test_counts,
        estimated_candidate_validation_counts=validation_counts,
        estimated_remaining_train_counts=remaining_counts,
        perceptual_candidate_pair_count=len(result.candidates),
        within_train_candidate_count=result.within_train_candidate_count,
        train_test_candidate_count=result.train_test_candidate_count,
        label_conflict_candidate_count=result.label_conflict_candidate_count,
        reviewed_candidate_count=result.reviewed_candidate_count,
        unresolved_candidate_count=unresolved,
        confirmed_duplicate_pair_count=confirmed,
        risks=tuple(risks),
    )


def write_perceptual_audit(
    result: PerceptualAuditResult,
    assessment: PerceptualValidationAssessment,
    decisions: tuple[HumanReviewDecision, ...],
    source_root: str | Path,
    destination: str | Path,
    *,
    panel_pair_limit: int = 300,
) -> tuple[Path, Path, Path, Path]:
    """Write local non-final JSON artifacts and a deterministic review panel."""

    if assessment.dataset_version_id != result.dataset_version_id:
        raise PerceptualAuditError("assessment belongs to a different dataset version")
    decision_map = _validate_perceptual_decisions(result.candidates, decisions)
    embedded_ids = {
        pair.review_id
        for pair in result.candidates
        if pair.human_decision is not None
    }
    if set(decision_map) != embedded_ids:
        raise PerceptualAuditError(
            "result must be analyzed with the same human decisions being written"
        )
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise PerceptualAuditError(
            f"perceptual audit destination already exists: {target}"
        )
    if type(panel_pair_limit) is not int or panel_pair_limit < 1:
        raise PerceptualAuditError("panel_pair_limit must be a positive integer")
    audit_bytes = canonical_json_bytes(result.payload())
    assessment_bytes = canonical_json_bytes(assessment.payload())
    try:
        decisions_bytes = canonical_human_decisions_bytes(
            result.dataset_version_id, decisions
        )
    except ReviewError as error:
        raise PerceptualAuditError(f"cannot serialize human decisions: {error}") from error
    panel_path = target / "review_panel.html"
    panel_bytes = render_review_panel(
        result,
        source_root,
        panel_path,
        max_pairs=panel_pair_limit,
    )
    target.mkdir(parents=True, exist_ok=False)
    audit_path = target / "perceptual_audit.json"
    decisions_path = target / "perceptual_human_decisions.json"
    assessment_path = target / "validation_feasibility.json"
    audit_path.write_bytes(audit_bytes)
    decisions_path.write_bytes(decisions_bytes)
    assessment_path.write_bytes(assessment_bytes)
    panel_path.write_bytes(panel_bytes)
    return audit_path, decisions_path, assessment_path, panel_path


def render_review_panel(
    result: PerceptualAuditResult,
    source_root: str | Path,
    panel_path: str | Path,
    *,
    max_pairs: int = 300,
) -> bytes:
    """Render a local side-by-side panel with a JSON decision export."""

    if type(max_pairs) is not int or max_pairs < 1:
        raise PerceptualAuditError("max_pairs must be a positive integer")
    selected = _select_panel_pairs(result.candidates, max_pairs)
    output_path = Path(panel_path)
    root = Path(source_root).resolve()
    pair_payloads = []
    for pair in selected:
        pair_payload = pair.as_json()
        pair_payload["left"]["image_uri"] = _relative_image_uri(
            root, pair.left.relative_path, output_path.parent
        )
        pair_payload["right"]["image_uri"] = _relative_image_uri(
            root, pair.right.relative_path, output_path.parent
        )
        pair_payloads.append(pair_payload)
    panel_data = {
        "artifact_type": "perceptual_review_panel_data",
        "artifact_version": "1.0",
        "dataset_version_id": result.dataset_version_id,
        "candidate_max_distance": result.candidate_max_distance,
        "total_candidate_pair_count": len(result.candidates),
        "displayed_pair_count": len(selected),
        "pairs": pair_payloads,
    }
    serialized = json.dumps(
        panel_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    html = _PANEL_TEMPLATE.replace("__PANEL_DATA__", serialized)
    return html.encode("utf-8")


@dataclass(frozen=True, slots=True)
class _PairwiseProfile:
    pair_comparison_count: int
    distance_histogram: tuple[int, ...]
    distance_histograms_by_scope: dict[ComparisonScope, tuple[int, ...]]
    same_label_distance_histogram: tuple[int, ...]
    conflicting_label_distance_histogram: tuple[int, ...]
    nearest_neighbor_distance_histogram: tuple[int, ...]
    unpaired_image_count: int
    candidate_coordinates: tuple[tuple[int, int, int], ...]


def _profile_all_pairs(
    records: tuple[PerceptualHashRecord, ...],
    *,
    candidate_max_distance: int,
    block_size: int,
) -> _PairwiseProfile:
    count = len(records)
    hashes = np.array([int(record.phash, 16) for record in records], dtype=np.uint64)
    partitions = np.array(
        [record.source_partition or "other" for record in records], dtype="U10"
    )
    classes = np.array([record.class_name for record in records], dtype="U16")
    histogram = np.zeros(PHASH_BITS + 1, dtype=np.int64)
    scope_histograms = {
        scope: np.zeros(PHASH_BITS + 1, dtype=np.int64)
        for scope in ComparisonScope
    }
    same_label_histogram = np.zeros(PHASH_BITS + 1, dtype=np.int64)
    conflict_histogram = np.zeros(PHASH_BITS + 1, dtype=np.int64)
    nearest = np.full(count, PHASH_BITS + 1, dtype=np.uint8)
    candidate_coordinates: list[tuple[int, int, int]] = []

    for left_start in range(0, count, block_size):
        left_stop = min(left_start + block_size, count)
        for right_start in range(left_start, count, block_size):
            right_stop = min(right_start + block_size, count)
            xor = np.bitwise_xor(
                hashes[left_start:left_stop, None],
                hashes[None, right_start:right_stop],
            )
            byte_view = xor.view(np.uint8).reshape((*xor.shape, 8))
            distances = _POPCOUNT[byte_view].sum(axis=2, dtype=np.uint8)
            if left_start == right_start:
                valid = np.triu(np.ones(distances.shape, dtype=bool), k=1)
            else:
                valid = np.ones(distances.shape, dtype=bool)
            valid_distances = distances[valid]
            histogram += np.bincount(
                valid_distances, minlength=PHASH_BITS + 1
            )
            relative_left, relative_right = np.nonzero(valid)
            absolute_left = left_start + relative_left
            absolute_right = right_start + relative_right
            left_partitions = partitions[absolute_left]
            right_partitions = partitions[absolute_right]
            for scope in ComparisonScope:
                scope_mask = _scope_mask(scope, left_partitions, right_partitions)
                scope_histograms[scope] += np.bincount(
                    valid_distances[scope_mask], minlength=PHASH_BITS + 1
                )
            same_label = classes[absolute_left] == classes[absolute_right]
            same_label_histogram += np.bincount(
                valid_distances[same_label], minlength=PHASH_BITS + 1
            )
            conflict_histogram += np.bincount(
                valid_distances[~same_label], minlength=PHASH_BITS + 1
            )

            working = np.where(valid, distances, PHASH_BITS + 1)
            if left_start == right_start:
                block_minimum = np.minimum(working.min(axis=0), working.min(axis=1))
                nearest[left_start:left_stop] = np.minimum(
                    nearest[left_start:left_stop], block_minimum
                )
            else:
                nearest[left_start:left_stop] = np.minimum(
                    nearest[left_start:left_stop], working.min(axis=1)
                )
                nearest[right_start:right_stop] = np.minimum(
                    nearest[right_start:right_stop], working.min(axis=0)
                )

            candidate_mask = valid_distances <= candidate_max_distance
            for distance, left_index, right_index in zip(
                valid_distances[candidate_mask],
                absolute_left[candidate_mask],
                absolute_right[candidate_mask],
                strict=True,
            ):
                candidate_coordinates.append(
                    (int(distance), int(left_index), int(right_index))
                )
                if len(candidate_coordinates) > MAX_REVIEW_CANDIDATES:
                    raise PerceptualAuditError(
                        "candidate screening produced more than 100,000 pairs; "
                        "lower the review ceiling and keep the full histogram"
                    )

    expected_pairs = count * (count - 1) // 2
    if int(histogram.sum()) != expected_pairs:
        raise PerceptualAuditError("pairwise distance profile is incomplete")
    if sum(int(values.sum()) for values in scope_histograms.values()) != expected_pairs:
        raise PerceptualAuditError("partition-scope profile is incomplete")
    paired = nearest <= PHASH_BITS
    nearest_histogram = np.bincount(
        nearest[paired], minlength=PHASH_BITS + 1
    )
    candidate_coordinates.sort()
    return _PairwiseProfile(
        pair_comparison_count=expected_pairs,
        distance_histogram=tuple(int(value) for value in histogram),
        distance_histograms_by_scope={
            scope: tuple(int(value) for value in scope_histograms[scope])
            for scope in ComparisonScope
        },
        same_label_distance_histogram=tuple(
            int(value) for value in same_label_histogram
        ),
        conflicting_label_distance_histogram=tuple(
            int(value) for value in conflict_histogram
        ),
        nearest_neighbor_distance_histogram=tuple(
            int(value) for value in nearest_histogram
        ),
        unpaired_image_count=int((~paired).sum()),
        candidate_coordinates=tuple(candidate_coordinates),
    )


def _validate_eligible_record(record: InventoryRecord) -> None:
    if not record.sha256:
        raise PerceptualAuditError(f"accepted sample has no SHA-256: {record.sample_id}")
    if record.class_name not in {"chihuahua", "muffin"}:
        raise PerceptualAuditError(f"accepted sample has invalid class: {record.sample_id}")
    if record.width is None or record.height is None:
        raise PerceptualAuditError(f"accepted sample has no dimensions: {record.sample_id}")
    if record.mode not in {"RGB", "RGBA", *GRAYSCALE_MODES}:
        raise PerceptualAuditError(f"accepted sample has unsupported mode: {record.sample_id}")


def _source_path(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise PerceptualAuditError("inventory contains an unsafe relative path")
    unresolved = root.joinpath(*relative.parts)
    if unresolved.is_symlink():
        raise PerceptualAuditError("inventory path resolves through a symbolic link")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise PerceptualAuditError("inventory path escapes source_root") from error
    return path


def _normalized_image_for_phash(
    path: Path, record: InventoryRecord
) -> Image.Image:
    try:
        with Image.open(path) as opened:
            if bool(
                getattr(opened, "is_animated", False)
                or getattr(opened, "n_frames", 1) > 1
            ):
                raise PerceptualAuditError(
                    f"animated image reached perceptual audit: {record.relative_path}"
                )
            opened.load()
            oriented = ImageOps.exif_transpose(opened)
            if oriented.mode == "RGB":
                normalized = oriented.copy()
            elif oriented.mode in GRAYSCALE_MODES:
                normalized = oriented.convert("RGB")
            elif oriented.mode == "RGBA":
                background = Image.new("RGB", oriented.size, (255, 255, 255))
                background.paste(oriented, mask=oriented.getchannel("A"))
                normalized = background
            else:
                raise PerceptualAuditError(
                    f"unsupported mode reached perceptual audit: {record.relative_path}"
                )
            if oriented is not opened:
                oriented.close()
            return normalized
    except PerceptualAuditError:
        raise
    except (OSError, SyntaxError, ValueError) as error:
        raise PerceptualAuditError(
            f"cannot decode eligible image: {record.relative_path}"
        ) from error


def _candidate_pair(
    left: PerceptualHashRecord,
    right: PerceptualHashRecord,
    distance: int,
) -> PerceptualCandidatePair:
    if right.sample_id < left.sample_id:
        left, right = right, left
    review_digest = canonical_json_sha256(
        {
            "method": "ImageHash.phash:8:4",
            "sample_ids": [left.sample_id, right.sample_id],
        }
    )
    return PerceptualCandidatePair(
        review_id=f"perceptual-phash64:{review_digest}",
        distance=distance,
        comparison_scope=_comparison_scope(
            left.source_partition, right.source_partition
        ),
        label_conflict=left.class_name != right.class_name,
        left=left,
        right=right,
    )


def _validate_perceptual_decisions(
    candidates: tuple[PerceptualCandidatePair, ...],
    decisions: tuple[HumanReviewDecision, ...],
) -> dict[str, HumanReviewDecision]:
    candidates_by_id = {pair.review_id: pair for pair in candidates}
    decisions_by_id: dict[str, HumanReviewDecision] = {}
    for decision in decisions:
        if decision.review_id in decisions_by_id:
            raise PerceptualAuditError(f"duplicate review_id: {decision.review_id}")
        pair = candidates_by_id.get(decision.review_id)
        if pair is None:
            raise PerceptualAuditError(
                "human decision does not match a screened perceptual pair"
            )
        if decision.target_sample_ids != pair.target_sample_ids:
            raise PerceptualAuditError(
                "human decision targets do not match the perceptual pair"
            )
        if decision.occurrence_type is not pair.occurrence_type:
            raise PerceptualAuditError(
                "human decision occurrence_type does not match the perceptual pair"
            )
        decisions_by_id[decision.review_id] = decision
    return decisions_by_id


def _scope_mask(
    scope: ComparisonScope,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    if scope is ComparisonScope.WITHIN_TRAIN:
        return (left == "train") & (right == "train")
    if scope is ComparisonScope.TRAIN_TEST:
        return ((left == "train") & (right == "test")) | (
            (left == "test") & (right == "train")
        )
    if scope is ComparisonScope.WITHIN_TEST:
        return (left == "test") & (right == "test")
    return ~(
        ((left == "train") & (right == "train"))
        | ((left == "test") & (right == "test"))
        | ((left == "train") & (right == "test"))
        | ((left == "test") & (right == "train"))
    )


def _comparison_scope(
    left_partition: str | None, right_partition: str | None
) -> ComparisonScope:
    values = {left_partition, right_partition}
    if values == {"train"}:
        return ComparisonScope.WITHIN_TRAIN
    if values == {"test"}:
        return ComparisonScope.WITHIN_TEST
    if values == {"train", "test"}:
        return ComparisonScope.TRAIN_TEST
    return ComparisonScope.INVOLVING_OTHER


def _histogram_payload(values: tuple[int, ...]) -> list[dict[str, int]]:
    return [
        {"distance": distance, "count": count}
        for distance, count in enumerate(values)
    ]


def _estimated_validation_count(count: int, fraction: float) -> int:
    if count < 2:
        return 0
    estimate = int(count * fraction + 0.5)
    return min(max(estimate, 1), count - 1)


def _select_panel_pairs(
    candidates: tuple[PerceptualCandidatePair, ...], max_pairs: int
) -> tuple[PerceptualCandidatePair, ...]:
    if len(candidates) <= max_pairs:
        return candidates
    selected: dict[str, PerceptualCandidatePair] = {}
    buckets = (
        sorted(
            (pair for pair in candidates if pair.label_conflict),
            key=lambda pair: (pair.distance, pair.review_id),
        ),
        sorted(
            (
                pair
                for pair in candidates
                if pair.comparison_scope is ComparisonScope.TRAIN_TEST
            ),
            key=lambda pair: (pair.distance, pair.review_id),
        ),
        sorted(candidates, key=lambda pair: (pair.distance, pair.review_id)),
        sorted(candidates, key=lambda pair: (-pair.distance, pair.review_id)),
    )
    positions = [0] * len(buckets)
    while len(selected) < max_pairs:
        changed = False
        for index, bucket in enumerate(buckets):
            while positions[index] < len(bucket):
                pair = bucket[positions[index]]
                positions[index] += 1
                if pair.review_id not in selected:
                    selected[pair.review_id] = pair
                    changed = True
                    break
            if len(selected) == max_pairs:
                break
        if not changed:
            break
    return tuple(sorted(selected.values(), key=lambda pair: (pair.distance, pair.review_id)))


def _relative_image_uri(
    root: Path, relative_path: str, panel_parent: Path
) -> str:
    image_path = _source_path(root, relative_path)
    relative = os.path.relpath(image_path, panel_parent.resolve())
    return quote(Path(relative).as_posix(), safe="/.")


_PANEL_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisão perceptual — Chihuahua vs. Muffin</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}body{max-width:1500px;margin:auto;padding:1rem}header{position:sticky;top:0;background:Canvas;padding:.8rem;border-bottom:1px solid GrayText;z-index:2}.pair{border:1px solid GrayText;border-radius:.6rem;padding:1rem;margin:1rem 0}.images{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.sample img{display:block;width:100%;height:340px;object-fit:contain;background:#eee}.meta{font-family:ui-monospace,monospace;overflow-wrap:anywhere;font-size:.82rem}.decision{display:grid;gap:.5rem;margin-top:1rem}.warning{color:#b45309}@media(max-width:800px){.images{grid-template-columns:1fr}.sample img{height:280px}}
</style>
</head>
<body>
<header>
<h1>Revisão perceptual de duplicatas</h1>
<p class="warning">pHash é somente triagem. Proximidade não confirma duplicação e não autoriza exclusão.</p>
<label>Revisor <input id="reviewer" autocomplete="name"></label>
<label>Data <input id="decision-date" type="date"></label>
<button id="export">Exportar decisões JSON</button>
<span id="summary"></span>
</header>
<main id="pairs"></main>
<script id="review-data" type="application/json">__PANEL_DATA__</script>
<script>
const data=JSON.parse(document.getElementById('review-data').textContent);
const main=document.getElementById('pairs');
document.getElementById('summary').textContent=`${data.displayed_pair_count} de ${data.total_candidate_pair_count} pares exibidos; teto de triagem ${data.candidate_max_distance}.`;
function sampleView(sample){const box=document.createElement('section');box.className='sample';const img=document.createElement('img');img.src=sample.image_uri;img.loading='lazy';img.alt=sample.relative_path;const meta=document.createElement('p');meta.className='meta';meta.textContent=`${sample.relative_path}\nID: ${sample.sample_id}\nclasse=${sample.class_name} partição=${sample.source_partition}\npHash=${sample.phash} ${sample.width}×${sample.height}`;box.append(img,meta);return box}
for(const pair of data.pairs){const card=document.createElement('article');card.className='pair';card.dataset.reviewId=pair.review_id;card.dataset.payload=JSON.stringify(pair);const title=document.createElement('h2');title.textContent=`distância ${pair.distance} · ${pair.comparison_scope}${pair.label_conflict?' · CONFLITO DE RÓTULO':''}`;const images=document.createElement('div');images.className='images';images.append(sampleView(pair.left),sampleView(pair.right));const form=document.createElement('div');form.className='decision';form.innerHTML=`<label>Decisão <select class="state"><option value="">pendente</option><option value="confirmed_duplicate">duplicata confirmada</option><option value="not_duplicate">não duplicata</option><option value="uncertain">incerto</option></select></label><label>Preservar, se duplicata confirmada <select class="retain"><option value="">sem disposição nesta etapa</option><option value="left">imagem esquerda</option><option value="right">imagem direita</option></select></label><label>Justificativa <textarea class="justification" rows="3"></textarea></label>`;if(pair.human_decision){form.querySelector('.state').value=pair.human_decision.decision;form.querySelector('.justification').value=pair.human_decision.justification;const retained=pair.human_decision.retained_sample_ids;if(retained.includes(pair.left.sample_id))form.querySelector('.retain').value='left';if(retained.includes(pair.right.sample_id))form.querySelector('.retain').value='right'}card.append(title,images,form);main.append(card)}
document.getElementById('export').addEventListener('click',()=>{const reviewer=document.getElementById('reviewer').value.trim();const decisionDate=document.getElementById('decision-date').value;const decisions=[];for(const card of document.querySelectorAll('.pair')){const pair=JSON.parse(card.dataset.payload);const state=card.querySelector('.state').value;if(!state)continue;const justification=card.querySelector('.justification').value.trim();if(!reviewer||!decisionDate||!justification){alert('Preencha revisor, data e justificativa para todas as decisões.');return}const targets=[pair.left.sample_id,pair.right.sample_id].sort();let retained=[],excluded=[];if(state==='not_duplicate')retained=targets;if(state==='confirmed_duplicate'){const keep=card.querySelector('.retain').value;if(keep){const kept=keep==='left'?pair.left.sample_id:pair.right.sample_id;retained=[kept];excluded=targets.filter(value=>value!==kept)}}decisions.push({review_id:pair.review_id,target_sample_ids:targets,occurrence_type:pair.occurrence_type,decision:state,justification,reviewer,decision_date:decisionDate,retained_sample_ids:retained.sort(),excluded_sample_ids:excluded.sort()})}decisions.sort((a,b)=>a.review_id.localeCompare(b.review_id));const payload={artifact_type:'human_review_decisions',artifact_version:'1.0',dataset_version_id:data.dataset_version_id,decisions};const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='perceptual_review_submission.json';link.click();URL.revokeObjectURL(link.href)})
</script>
</body>
</html>
"""
