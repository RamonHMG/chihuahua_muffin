"""Final SPEC 01 artifact generation and independent validation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from importlib.metadata import version as package_version
from io import StringIO
import json
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Callable

from .canonical import (
    INDEX_CSV_HEADER,
    CanonicalizationError,
    canonical_index_csv_bytes,
    canonical_json_bytes,
    make_sample_id,
    manifest_sha256,
    normalize_relative_path,
    sha256_hex,
)
from .config import ValidationMode, load_config
from .deduplication import (
    ExactDeduplicationResult,
    SampleAction,
    analyze_exact_duplicates,
)
from .inventory import (
    AuditStatus,
    CLASS_IDS,
    InventoryRecord,
    InventoryResult,
    load_initial_inventory,
    sha256_file,
)
from .provenance import authorization_from_config, provenance_from_config
from .records import ProjectConfig
from .review import load_human_decisions
from .splitting import (
    DatasetSplit,
    SplitAssignment,
    SplitAssignmentsArtifact,
    load_split_assignments,
)


FINAL_ARTIFACT_NAMES = (
    "index.csv",
    "dataset_manifest.json",
    "quality_report.json",
)
REAL_EXPECTED_SPLIT_COUNTS = {
    "train": {"chihuahua": 2175, "muffin": 1848},
    "validation": {"chihuahua": 384, "muffin": 326},
    "test": {"chihuahua": 640, "muffin": 544},
}
REAL_PERCEPTUAL_CANDIDATE_COUNT = 719
REAL_APPROVED_SOURCE_DIGESTS = {
    "initial_inventory": "6994b3d3d4226d958f9cc34a3e54b076356d368a7ac9b3e76608200380dbf4f3",
    "initial_quality_report": "72b8d34d1f56b87ce8cf1320ade72abff74575b9af9fff43a0eddea24fa84aae",
    "exact_deduplication": "cc9eebf742ad0d7069a24d1a7eeaef6f5610b1e01516cee7756270553bc62ad6",
    "exact_human_decisions": "aaa627f05b4907a0679fa1cd5e09f79b986841060ce329fccf2eb9d813494dce",
    "exact_validation_feasibility": "e53bcedb6cc313f4b33f9c912788f770b34ffe56149513267c2f09f91a4593e3",
    "perceptual_audit": "78a5da381d6561cce8faf71bb3ed401bbeed8c999b35a1789454108be6a1ee21",
    "perceptual_human_decisions": "aaa627f05b4907a0679fa1cd5e09f79b986841060ce329fccf2eb9d813494dce",
    "perceptual_validation_feasibility": "3e9546d23845d092d49c27ea526b4798cd2e6349a3f345d489b41aa394989084",
    "perceptual_review_panel": "d44c6bacc33546c0fa474d1c5c462f941549c5fe9b5812289257465b032c0e98",
    "split_assignments": "e37e955f8c3da8d74e56d4a26511de0411e0e9c6f3efffbfd549e5ab46ef727e",
    "split_report": "70ef339bbbf77d552205e4a41f89d52a7c3c35f42d6dddb0454ec8b90a3352c3",
}


class PreparationError(RuntimeError):
    """Raised when final data artifacts cannot be generated or validated."""


@dataclass(frozen=True, slots=True)
class PreparationLayout:
    """Explicit paths for every approved pipeline input and final output."""

    project_root: Path
    config_path: Path
    raw_root: Path
    initial_inventory: Path
    initial_quality_report: Path
    exact_deduplication: Path
    exact_human_decisions: Path
    exact_validation_feasibility: Path
    perceptual_audit: Path
    perceptual_human_decisions: Path
    perceptual_validation_feasibility: Path
    perceptual_review_panel: Path
    split_assignments: Path
    split_report: Path
    prepared_dir: Path

    @classmethod
    def default(cls, project_root: str | Path = ".") -> PreparationLayout:
        root = Path(project_root).resolve()
        audit = root / "data/audit"
        return cls(
            project_root=root,
            config_path=root / "config/example.toml",
            raw_root=root / "data/raw/kaggle-2733586-v2",
            initial_inventory=audit / "kaggle-2733586-v2-initial/inventory.json",
            initial_quality_report=(
                audit / "kaggle-2733586-v2-initial/quality_report.json"
            ),
            exact_deduplication=(
                audit / "kaggle-2733586-v2-exact/exact_deduplication.json"
            ),
            exact_human_decisions=(
                audit / "kaggle-2733586-v2-exact/human_decisions.json"
            ),
            exact_validation_feasibility=(
                audit / "kaggle-2733586-v2-exact/validation_feasibility.json"
            ),
            perceptual_audit=(
                audit / "kaggle-2733586-v2-perceptual/perceptual_audit.json"
            ),
            perceptual_human_decisions=(
                audit
                / "kaggle-2733586-v2-perceptual/perceptual_human_decisions.json"
            ),
            perceptual_validation_feasibility=(
                audit / "kaggle-2733586-v2-perceptual/validation_feasibility.json"
            ),
            perceptual_review_panel=(
                audit / "kaggle-2733586-v2-perceptual/review_panel.html"
            ),
            split_assignments=(
                audit / "kaggle-2733586-v2-splits/split_assignments.json"
            ),
            split_report=audit / "kaggle-2733586-v2-splits/split_report.json",
            prepared_dir=root / "data/prepared",
        )

    def source_artifacts(self) -> dict[str, Path]:
        return {
            "initial_inventory": self.initial_inventory,
            "initial_quality_report": self.initial_quality_report,
            "exact_deduplication": self.exact_deduplication,
            "exact_human_decisions": self.exact_human_decisions,
            "exact_validation_feasibility": self.exact_validation_feasibility,
            "perceptual_audit": self.perceptual_audit,
            "perceptual_human_decisions": self.perceptual_human_decisions,
            "perceptual_validation_feasibility": (
                self.perceptual_validation_feasibility
            ),
            "perceptual_review_panel": self.perceptual_review_panel,
            "split_assignments": self.split_assignments,
            "split_report": self.split_report,
        }


@dataclass(frozen=True, slots=True)
class PipelineInputs:
    config: ProjectConfig
    inventory: InventoryResult
    exact_result: ExactDeduplicationResult
    split_artifact: SplitAssignmentsArtifact
    split_report: dict[str, object]
    perceptual_summary: dict[str, object]
    source_artifacts: dict[str, dict[str, str]]
    raw_root: Path
    raw_root_logical: str


@dataclass(frozen=True, slots=True)
class PreparedArtifacts:
    """Complete final bytes, all built before publication begins."""

    index_bytes: bytes
    quality_report_bytes: bytes
    manifest_bytes: bytes
    manifest: dict[str, object]
    counts_by_split_and_class: dict[str, dict[str, int]]

    def files(self) -> dict[str, bytes]:
        return {
            "index.csv": self.index_bytes,
            "dataset_manifest.json": self.manifest_bytes,
            "quality_report.json": self.quality_report_bytes,
        }


@dataclass(frozen=True, slots=True)
class PublicationResult:
    status: str
    destination: Path


@dataclass(frozen=True, slots=True)
class ValidationResult:
    dataset_version_id: str
    discovered_file_count: int
    accepted_image_count: int
    counts_by_split_and_class: dict[str, dict[str, int]]
    index_sha256: str
    quality_report_sha256: str
    manifest_sha256: str
    manifest_file_sha256: str


@dataclass(frozen=True, slots=True)
class PreparationRunResult:
    publication: PublicationResult
    validation: ValidationResult


def prepare_dataset(
    layout: PreparationLayout,
    *,
    approved_source_digests: dict[str, str],
    expected_split_counts: dict[str, dict[str, int]],
    expected_perceptual_candidate_count: int,
) -> PreparationRunResult:
    """Integrate approved inputs, publish atomically, then validate read-only."""

    inputs = load_pipeline_inputs(
        layout,
        approved_source_digests=approved_source_digests,
        expected_split_counts=expected_split_counts,
        expected_perceptual_candidate_count=expected_perceptual_candidate_count,
    )
    artifacts = build_prepared_artifacts(
        inputs,
        expected_split_counts=expected_split_counts,
    )
    publication = publish_prepared_artifacts(artifacts, layout.prepared_dir)
    validation = validate_prepared_dataset(
        layout,
        approved_source_digests=approved_source_digests,
        expected_split_counts=expected_split_counts,
        expected_perceptual_candidate_count=expected_perceptual_candidate_count,
    )
    return PreparationRunResult(publication=publication, validation=validation)


def prepare_default_dataset(project_root: str | Path = ".") -> PreparationRunResult:
    return prepare_dataset(
        PreparationLayout.default(project_root),
        approved_source_digests=REAL_APPROVED_SOURCE_DIGESTS,
        expected_split_counts=REAL_EXPECTED_SPLIT_COUNTS,
        expected_perceptual_candidate_count=REAL_PERCEPTUAL_CANDIDATE_COUNT,
    )


def load_pipeline_inputs(
    layout: PreparationLayout,
    *,
    approved_source_digests: dict[str, str],
    expected_split_counts: dict[str, dict[str, int]],
    expected_perceptual_candidate_count: int,
) -> PipelineInputs:
    """Load approved artifacts and reject any divergence before generation."""

    actual_digests = _verify_approved_source_artifacts(
        layout, approved_source_digests
    )
    config = load_config(layout.config_path, mode=ValidationMode.ACQUISITION)
    provenance = provenance_from_config(config.dataset)
    authorization = authorization_from_config(config.dataset)
    provenance.require_acquisition_ready(authorization)
    dataset_version_id = provenance.immutable_version_id_for(authorization)

    inventory = load_initial_inventory(
        layout.initial_inventory,
        expected_dataset_version_id=dataset_version_id,
    )
    if len(inventory.records) != config.dataset.acquisition_file_count:
        raise PreparationError(
            "inventory file count differs from the recorded acquisition"
        )
    expected_initial_quality = canonical_json_bytes(inventory.quality_payload())
    if layout.initial_quality_report.read_bytes() != expected_initial_quality:
        raise PreparationError(
            "initial quality report does not match the approved inventory"
        )

    exact_decisions = load_human_decisions(
        layout.exact_human_decisions,
        expected_dataset_version_id=dataset_version_id,
    )
    exact_result = analyze_exact_duplicates(
        inventory,
        human_decisions=exact_decisions,
    )
    if exact_result.training_release_blocked:
        raise PreparationError("unresolved exact-duplicate reviews block release")
    if layout.exact_deduplication.read_bytes() != canonical_json_bytes(
        exact_result.payload()
    ):
        raise PreparationError(
            "exact deduplication artifact does not match inventory and decisions"
        )

    split_artifact = load_split_assignments(
        layout.split_assignments,
        expected_dataset_version_id=dataset_version_id,
    )
    split_report = _load_canonical_json_object(
        layout.split_report,
        description="split report",
    )
    _validate_split_inputs(
        inventory,
        exact_result,
        split_artifact,
        split_report,
        assignments_sha256=actual_digests["split_assignments"],
        expected_counts=expected_split_counts,
        expected_perceptual_candidate_count=expected_perceptual_candidate_count,
    )

    perceptual_document = _load_canonical_json_object(
        layout.perceptual_audit,
        description="perceptual audit",
    )
    perceptual_decisions = load_human_decisions(
        layout.perceptual_human_decisions,
        expected_dataset_version_id=dataset_version_id,
    )
    perceptual_summary = _validate_perceptual_inputs(
        perceptual_document,
        dataset_version_id=dataset_version_id,
        accepted_image_count=len(split_artifact.assignments),
        decision_count=len(perceptual_decisions),
        expected_candidate_count=expected_perceptual_candidate_count,
    )

    source_artifacts = {
        name: {
            "path": _logical_path(path, layout.project_root),
            "sha256": actual_digests[name],
        }
        for name, path in sorted(layout.source_artifacts().items())
    }
    return PipelineInputs(
        config=config,
        inventory=inventory,
        exact_result=exact_result,
        split_artifact=split_artifact,
        split_report=split_report,
        perceptual_summary=perceptual_summary,
        source_artifacts=source_artifacts,
        raw_root=layout.raw_root,
        raw_root_logical=_logical_path(layout.raw_root, layout.project_root),
    )


def _verify_approved_source_artifacts(
    layout: PreparationLayout,
    approved_source_digests: dict[str, str],
) -> dict[str, str]:
    paths = layout.source_artifacts()
    if set(approved_source_digests) != set(paths):
        raise PreparationError("approved source digest set does not match layout")
    actual: dict[str, str] = {}
    for name, path in sorted(paths.items()):
        if path.is_symlink() or not path.is_file():
            raise PreparationError(f"approved source artifact is missing: {path}")
        digest = sha256_file(path)
        if digest != approved_source_digests[name]:
            raise PreparationError(
                f"approved source artifact diverged: {name}; "
                f"expected {approved_source_digests[name]}, received {digest}"
            )
        actual[name] = digest
    return actual


def _load_canonical_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot read {description}: {error}") from error
    if type(value) is not dict:
        raise PreparationError(f"{description} must be a JSON object")
    if canonical_json_bytes(value) != raw:
        raise PreparationError(f"{description} is not canonical JSON")
    return value


def _validate_split_inputs(
    inventory: InventoryResult,
    exact_result: ExactDeduplicationResult,
    split_artifact: SplitAssignmentsArtifact,
    report: dict[str, object],
    *,
    assignments_sha256: str,
    expected_counts: dict[str, dict[str, int]],
    expected_perceptual_candidate_count: int,
) -> None:
    required_report_values = {
        "artifact_type": "dataset_split_report",
        "artifact_version": "1.0",
        "artifact_status": "nonfinal_increment_7_record",
        "dataset_version_id": inventory.dataset_version_id,
        "scenario": "A",
        "algorithm": "stratified_sha256_rank_v1",
        "seed": 42,
        "assignments_sha256": assignments_sha256,
        "source_test_ids_preserved": True,
        "validation_uses_only_source_train": True,
        "sample_ids_are_disjoint_across_splits": True,
        "originals_modified": False,
        "final_manifest_created": False,
    }
    for field, expected in required_report_values.items():
        if report.get(field) != expected:
            raise PreparationError(f"split report field {field!r} diverged")
    if report.get("counts_by_split_and_class") != expected_counts:
        raise PreparationError("split report counts differ from approved counts")

    policy = report.get("perceptual_policy")
    if type(policy) is not dict or policy != {
        "known_unreviewed_candidate_pair_count": expected_perceptual_candidate_count,
        "candidate_pairs_are_confirmed_duplicates": False,
        "candidate_pairs_used_for_grouping": False,
        "candidate_pairs_used_for_assignment": False,
        "manual_review_required_for_didactic_experiment": False,
    }:
        raise PreparationError("split report perceptual policy diverged")

    actions: dict[str, SampleAction] = {}
    for group in exact_result.groups:
        for outcome in group.outcomes:
            actions[outcome.sample_id] = outcome.action
    eligible = {
        record.sample_id: record
        for record in inventory.records
        if record.status is AuditStatus.ACCEPTED
        and record.image_format is not None
        and actions.get(record.sample_id, SampleAction.KEEP) is SampleAction.KEEP
    }
    assignments = {
        assignment.sample_id: assignment
        for assignment in split_artifact.assignments
    }
    if len(assignments) != len(split_artifact.assignments):
        raise PreparationError("split assignments contain repeated IDs")
    if set(assignments) != set(eligible):
        raise PreparationError("split assignments do not match eligible inventory IDs")
    for sample_id, assignment in assignments.items():
        record = eligible[sample_id]
        if (
            assignment.relative_path != record.relative_path
            or assignment.sha256 != record.sha256
            or assignment.class_name != record.class_name
            or assignment.class_id != record.class_id
            or assignment.source_partition != record.source_partition
        ):
            raise PreparationError(
                f"split assignment metadata diverges for sample {sample_id}"
            )
    actual_counts = _assignment_counts(split_artifact.assignments)
    if actual_counts != expected_counts:
        raise PreparationError("assignment counts differ from approved counts")
    original_test_ids = {
        record.sample_id
        for record in inventory.records
        if record.status is AuditStatus.ACCEPTED
        and record.image_format is not None
        and record.source_partition == "test"
    }
    assigned_test_ids = {
        assignment.sample_id
        for assignment in split_artifact.assignments
        if assignment.split is DatasetSplit.TEST
    }
    if assigned_test_ids != original_test_ids:
        raise PreparationError("original test IDs were not preserved exactly")


def _validate_perceptual_inputs(
    document: dict[str, object],
    *,
    dataset_version_id: str,
    accepted_image_count: int,
    decision_count: int,
    expected_candidate_count: int,
) -> dict[str, object]:
    if document.get("artifact_type") != "perceptual_duplicate_audit":
        raise PreparationError("unexpected perceptual audit artifact_type")
    if document.get("artifact_version") != "1.0":
        raise PreparationError("unsupported perceptual audit version")
    if document.get("dataset_version_id") != dataset_version_id:
        raise PreparationError("perceptual audit belongs to a different version")
    if document.get("images_analyzed") != accepted_image_count:
        raise PreparationError("perceptual audit image count diverged")
    if document.get("candidate_pair_count") != expected_candidate_count:
        raise PreparationError("perceptual candidate count diverged")
    if document.get("reviewed_candidate_count") != decision_count or decision_count != 0:
        raise PreparationError("perceptual decisions differ from approved empty review")
    method = document.get("method")
    screening = document.get("candidate_screening")
    if type(method) is not dict or any(
        method.get(field) != expected
        for field, expected in {
            "name": "phash",
            "library": "ImageHash",
            "library_version": "4.3.2",
            "hash_size": 8,
            "highfreq_factor": 4,
            "hash_bits": 64,
            "distance": "hamming",
            "purpose": "candidate_triage_only",
        }.items()
    ):
        raise PreparationError("perceptual method metadata diverged")
    if type(screening) is not dict or any(
        screening.get(field) != expected
        for field, expected in {
            "maximum_hamming_distance": 12,
            "is_duplicate_threshold": False,
            "automatic_removal": False,
            "requires_human_review": True,
        }.items()
    ):
        raise PreparationError("perceptual screening policy diverged")
    return {
        "method": "phash",
        "library": "ImageHash",
        "library_version": "4.3.2",
        "hash_size": 8,
        "highfreq_factor": 4,
        "hash_bits": 64,
        "distance": "hamming",
        "screening_maximum_distance": 12,
        "candidate_pair_count": expected_candidate_count,
        "reviewed_candidate_count": 0,
        "label_conflict_candidate_count": document.get(
            "label_conflict_candidate_count"
        ),
        "candidates_are_confirmed_duplicates": False,
        "used_for_exclusion": False,
        "used_for_grouping": False,
        "used_for_split_assignment": False,
    }


def _logical_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_prepared_artifacts(
    inputs: PipelineInputs,
    *,
    expected_split_counts: dict[str, dict[str, int]],
) -> PreparedArtifacts:
    """Build all final bytes after validating the source tree and assignments."""

    _verify_source_tree(inputs.raw_root, inputs.inventory)
    rows = _build_index_rows(
        inputs.inventory,
        inputs.exact_result,
        inputs.split_artifact,
    )
    index_bytes = canonical_index_csv_bytes(rows)
    quality_report = _build_quality_report(
        rows,
        dataset_version_id=inputs.inventory.dataset_version_id,
        exact_result=inputs.exact_result,
        perceptual_summary=inputs.perceptual_summary,
    )
    quality_bytes = canonical_json_bytes(quality_report)
    counts = _accepted_counts_by_split_and_class(rows)
    if counts != expected_split_counts:
        raise PreparationError("final index counts differ from approved counts")
    manifest = _build_manifest(
        inputs,
        index_sha256=sha256_hex(index_bytes),
        quality_report_sha256=sha256_hex(quality_bytes),
        index_row_count=len(rows),
        counts_by_split_and_class=counts,
        quality_report=quality_report,
    )
    manifest["manifest_sha256"] = manifest_sha256(
        manifest,
        digest_field="manifest_sha256",
    )
    manifest_bytes = canonical_json_bytes(manifest)
    return PreparedArtifacts(
        index_bytes=index_bytes,
        quality_report_bytes=quality_bytes,
        manifest_bytes=manifest_bytes,
        manifest=manifest,
        counts_by_split_and_class=counts,
    )


def _verify_source_tree(raw_root: Path, inventory: InventoryResult) -> None:
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise PreparationError(f"raw dataset root is unavailable: {raw_root}")
    expected_paths = {record.relative_path for record in inventory.records}
    discovered_paths = {
        path.relative_to(raw_root).as_posix()
        for path in raw_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if discovered_paths != expected_paths:
        missing = sorted(expected_paths - discovered_paths)
        extra = sorted(discovered_paths - expected_paths)
        raise PreparationError(
            f"raw source tree differs from inventory; missing={missing!r}, "
            f"extra={extra!r}"
        )
    resolved_root = raw_root.resolve()
    for record in inventory.records:
        normalized = normalize_relative_path(record.relative_path)
        if normalized != record.relative_path:
            raise PreparationError(
                f"inventory path is not canonical: {record.relative_path}"
            )
        source = raw_root / normalized
        if source.is_symlink() or not source.is_file():
            raise PreparationError(f"referenced source file is invalid: {source}")
        try:
            source.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise PreparationError(f"source path escapes raw root: {source}") from error
        if sha256_file(source) != record.sha256:
            raise PreparationError(
                f"source SHA-256 differs from inventory: {record.relative_path}"
            )


def _build_index_rows(
    inventory: InventoryResult,
    exact_result: ExactDeduplicationResult,
    split_artifact: SplitAssignmentsArtifact,
) -> list[dict[str, object]]:
    assignments = {
        assignment.sample_id: assignment
        for assignment in split_artifact.assignments
    }
    outcomes = {}
    group_by_sample: dict[str, str] = {}
    for group in exact_result.groups:
        for outcome in group.outcomes:
            if outcome.sample_id in outcomes:
                raise PreparationError("sample appears in multiple exact groups")
            outcomes[outcome.sample_id] = outcome
            group_by_sample[outcome.sample_id] = group.group_id

    rows: list[dict[str, object]] = []
    consumed_assignments: set[str] = set()
    for record in inventory.records:
        if record.status is AuditStatus.REVIEW:
            raise PreparationError(
                f"inventory review remains unresolved: {record.sample_id}"
            )
        status = record.status.value
        reasons = list(record.reasons)
        split = ""
        exact_outcome = outcomes.get(record.sample_id)
        if record.status is AuditStatus.ACCEPTED:
            action = (
                exact_outcome.action
                if exact_outcome is not None
                else SampleAction.KEEP
            )
            if action is SampleAction.BLOCK:
                raise PreparationError(
                    f"exact duplicate remains blocked: {record.sample_id}"
                )
            if action is SampleAction.EXCLUDE:
                status = AuditStatus.EXCLUDED.value
                reasons.append(exact_outcome.reason)
            else:
                assignment = assignments.get(record.sample_id)
                if assignment is None:
                    raise PreparationError(
                        f"accepted sample lacks split assignment: {record.sample_id}"
                    )
                consumed_assignments.add(record.sample_id)
                split = assignment.split.value
                if exact_outcome is not None:
                    reasons.append(exact_outcome.reason)
        elif record.sample_id in assignments:
            raise PreparationError(
                f"non-accepted inventory record has an assignment: {record.sample_id}"
            )
        rows.append(
            {
                "sample_id": record.sample_id,
                "relative_path": record.relative_path,
                "class_name": record.class_name,
                "class_id": record.class_id,
                "source_partition": record.source_partition,
                "split": split,
                "status": status,
                "reason": ";".join(sorted(set(reasons))),
                "sha256": record.sha256,
                "duplicate_group_id": group_by_sample.get(record.sample_id, ""),
                "perceptual_method": "",
                "perceptual_evidence": "",
                "format": record.image_format,
                "width": record.width,
                "height": record.height,
                "channels": record.channels,
                "is_animated": record.is_animated,
                "exif_orientation_applied": record.exif_transpose_required,
                "color_conversion": record.color_conversion,
            }
        )
    if consumed_assignments != set(assignments):
        raise PreparationError("split assignments contain IDs absent from final index")
    return rows


def _build_quality_report(
    rows: list[dict[str, object]],
    *,
    dataset_version_id: str,
    exact_result: ExactDeduplicationResult,
    perceptual_summary: dict[str, object],
) -> dict[str, object]:
    status_counts = Counter(str(row["status"]) for row in rows)
    class_counts = Counter(str(row["class_name"] or "unidentified") for row in rows)
    source_counts = Counter(
        str(row["source_partition"] or "none") for row in rows
    )
    split_counts = Counter(str(row["split"] or "none") for row in rows)
    format_counts = Counter(str(row["format"] or "unidentified") for row in rows)
    reason_counts = Counter(
        reason
        for row in rows
        for reason in str(row["reason"]).split(";")
        if reason
    )
    excluded = [
        {
            "sample_id": str(row["sample_id"]),
            "relative_path": str(row["relative_path"]),
            "reason": str(row["reason"]),
        }
        for row in rows
        if row["status"] == AuditStatus.EXCLUDED.value
    ]
    return {
        "artifact_type": "final_quality_report",
        "schema_version": "1.0",
        "dataset_version_id": dataset_version_id,
        "total_discovered_files": len(rows),
        "reconciled_files": sum(status_counts.values()),
        "counts_by_status": dict(sorted(status_counts.items())),
        "counts_by_class": dict(sorted(class_counts.items())),
        "counts_by_source_partition": dict(sorted(source_counts.items())),
        "counts_by_split": dict(sorted(split_counts.items())),
        "counts_by_format": dict(sorted(format_counts.items())),
        "counts_by_reason": dict(sorted(reason_counts.items())),
        "accepted_counts_by_split_and_class": (
            _accepted_counts_by_split_and_class(rows)
        ),
        "excluded_occurrences": excluded,
        "review_occurrences": [],
        "exact_deduplication": {
            "method": "sha256_original_bytes",
            "group_count": len(exact_result.groups),
            "excluded_sample_count": exact_result.excluded_sample_count,
            "blocked_sample_count": exact_result.blocked_sample_count,
            "blocking_review_ids": list(exact_result.blocking_review_ids),
        },
        "perceptual_audit": perceptual_summary,
        "perceptual_limitation_accepted_for_didactic_use": True,
        "originals_modified": False,
        "final_release_blocked": False,
    }


def _build_manifest(
    inputs: PipelineInputs,
    *,
    index_sha256: str,
    quality_report_sha256: str,
    index_row_count: int,
    counts_by_split_and_class: dict[str, dict[str, int]],
    quality_report: dict[str, object],
) -> dict[str, object]:
    config = inputs.config
    dataset = config.dataset
    authorization = dataset.human_authorization
    split_report = inputs.split_report
    return {
        "artifact_type": "dataset_manifest",
        "schema_version": "1.0",
        "manifest_sha256": "",
        "dataset_version_id": inputs.inventory.dataset_version_id,
        "dataset": {
            "source": dataset.source,
            "source_id": dataset.source_id,
            "handle": dataset.handle,
            "owner": dataset.owner,
            "version": dataset.version,
            "version_status": dataset.version_status,
            "source_url": dataset.source_url,
            "metadata_url": dataset.metadata_url,
            "metadata_checked_at": dataset.metadata_checked_at,
            "raw_root": inputs.raw_root_logical,
        },
        "license": {
            "declared_value": dataset.license,
            "status": dataset.license_status,
            "note": dataset.license_note,
            "confirmed": False,
        },
        "acquisition": {
            "acquired_at": dataset.acquired_at,
            "tool": dataset.acquisition_tool,
            "tool_version": dataset.acquisition_tool_version,
            "file_count": dataset.acquisition_file_count,
            "originals_preserved": True,
            "download_performed_by_increment_8": False,
        },
        "human_authorization": {
            "status": authorization.status,
            "source_id": authorization.source_id,
            "dataset_handle": authorization.dataset_handle,
            "dataset_version": authorization.dataset_version,
            "purpose": authorization.purpose,
            "decision_date": authorization.decision_date,
            "authorized_by": authorization.authorized_by,
            "license_ambiguity_acknowledged": (
                authorization.license_ambiguity_acknowledged
            ),
            "local_use_only": authorization.local_use_only,
            "redistribution_allowed": authorization.redistribution_allowed,
        },
        "class_mapping": {"chihuahua": 0, "muffin": 1},
        "positive_class": "muffin",
        "split": {
            "scenario": "A",
            "seed": 42,
            "algorithm": "stratified_sha256_rank_v1",
            "source_partitions_preserved": ["train", "test"],
            "validation_source": "train",
            "requested_validation_fraction_of_source_train": 0.15,
            "effective_train_fraction_of_source_train": split_report[
                "effective_train_fraction_of_source_train"
            ],
            "effective_validation_fraction_of_source_train": split_report[
                "effective_validation_fraction_of_source_train"
            ],
            "counts_by_split_and_class": counts_by_split_and_class,
            "counts_by_source_partition_and_class": split_report[
                "counts_by_source_partition_and_class"
            ],
            "test_original_preserved": True,
        },
        "preprocessing": _preprocessing_payload(config),
        "duplicates": {
            "exact": quality_report["exact_deduplication"],
            "perceptual": inputs.perceptual_summary,
        },
        "counts": {
            "discovered_files": index_row_count,
            "accepted_images": sum(
                count
                for split in counts_by_split_and_class.values()
                for count in split.values()
            ),
            "by_status": quality_report["counts_by_status"],
            "by_split_and_class": counts_by_split_and_class,
        },
        "artifacts": {
            "index": {
                "path": "index.csv",
                "sha256": index_sha256,
                "row_count": index_row_count,
            },
            "quality_report": {
                "path": "quality_report.json",
                "sha256": quality_report_sha256,
            },
        },
        "source_artifacts": inputs.source_artifacts,
        "producer": {
            "project": "chihuahua-muffin",
            "project_version": "0.0.0",
            "python": platform.python_version(),
            "Pillow": package_version("Pillow"),
            "ImageHash": package_version("ImageHash"),
            "kagglehub": package_version("kagglehub"),
        },
        "limitations": [
            f"The {inputs.perceptual_summary['candidate_pair_count']} pHash candidate pairs are not confirmed duplicates.",
            "No pair-by-pair human review was performed for those candidates.",
            "Visual overlap may remain across train, validation, and test.",
            "Future generalization metrics must be interpreted with possible leakage.",
            "The declared license remains ambiguous; local educational use relies on the recorded exception.",
        ],
        "methodological_exception": {
            "purpose": "didactic_experiment",
            "perceptual_review_waived": True,
            "release_blocking": False,
            "exact_integrity_errors_remain_blocking": True,
        },
    }


def _preprocessing_payload(config: ProjectConfig) -> dict[str, object]:
    preprocessing = config.preprocessing
    return {
        "height": preprocessing.height,
        "width": preprocessing.width,
        "channels": preprocessing.channels,
        "color_mode": preprocessing.color_mode,
        "resize": preprocessing.resize,
        "interpolation": preprocessing.interpolation,
        "padding_color": preprocessing.padding_color,
        "padding_rgb": list(preprocessing.padding_rgb),
        "padding_normalized": list(preprocessing.padding_normalized),
        "pixel_min": preprocessing.pixel_min,
        "pixel_max": preprocessing.pixel_max,
        "fix_exif_orientation": preprocessing.fix_exif_orientation,
        "rgba_background": preprocessing.rgba_background,
        "grayscale_to_rgb": preprocessing.grayscale_to_rgb,
        "exclude_animated": preprocessing.exclude_animated,
        "materialize_resized_images": (
            preprocessing.materialize_resized_images
        ),
        "application_stage": "read_time_by_spec_02_tf_data",
    }


def _assignment_counts(
    assignments: tuple[SplitAssignment, ...],
) -> dict[str, dict[str, int]]:
    counts = Counter(
        (assignment.split.value, assignment.class_name)
        for assignment in assignments
    )
    return {
        split.value: {
            class_name: counts[(split.value, class_name)]
            for class_name in CLASS_IDS
        }
        for split in DatasetSplit
    }


def _accepted_counts_by_split_and_class(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, int]]:
    counts = Counter(
        (str(row["split"]), str(row["class_name"]))
        for row in rows
        if row["status"] == AuditStatus.ACCEPTED.value
    )
    return {
        split.value: {
            class_name: counts[(split.value, class_name)]
            for class_name in CLASS_IDS
        }
        for split in DatasetSplit
    }


ArtifactWriter = Callable[[Path, bytes], None]


def publish_prepared_artifacts(
    artifacts: PreparedArtifacts,
    destination: str | Path,
    *,
    writer: ArtifactWriter | None = None,
) -> PublicationResult:
    """Publish a complete directory atomically or recognize identical output."""

    target = Path(destination)
    expected = artifacts.files()
    if target.is_symlink():
        raise PreparationError(f"prepared destination must not be a symlink: {target}")
    if target.exists():
        if not target.is_dir():
            raise PreparationError(f"prepared destination is not a directory: {target}")
        existing_names = {path.name for path in target.iterdir()}
        if existing_names != set(expected):
            raise PreparationError(
                "existing prepared directory is incomplete or contains unexpected files"
            )
        for name, content in expected.items():
            path = target / name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                raise PreparationError(
                    f"existing prepared artifact diverges and will not be overwritten: {name}"
                )
        return PublicationResult(status="up_to_date", destination=target)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    write = writer or _exclusive_write
    try:
        for name in FINAL_ARTIFACT_NAMES:
            write(temporary / name, expected[name])
        if {
            path.name for path in temporary.iterdir() if path.is_file()
        } != set(expected):
            raise PreparationError("temporary publication is incomplete")
        for name, content in expected.items():
            if (temporary / name).read_bytes() != content:
                raise PreparationError(f"temporary artifact verification failed: {name}")
        temporary.rename(target)
    except Exception as error:
        if temporary.exists():
            shutil.rmtree(temporary)
        if isinstance(error, PreparationError):
            raise
        raise PreparationError(f"atomic publication failed: {error}") from error
    return PublicationResult(status="created", destination=target)


def _exclusive_write(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()


def validate_prepared_dataset(
    layout: PreparationLayout,
    *,
    approved_source_digests: dict[str, str],
    expected_split_counts: dict[str, dict[str, int]],
    expected_perceptual_candidate_count: int,
) -> ValidationResult:
    """Validate final artifacts and referenced originals without writing."""

    actual_source_digests = _verify_approved_source_artifacts(
        layout, approved_source_digests
    )
    config = load_config(layout.config_path, mode=ValidationMode.ACQUISITION)
    provenance = provenance_from_config(config.dataset)
    authorization = authorization_from_config(config.dataset)
    provenance.require_acquisition_ready(authorization)
    dataset_version_id = provenance.immutable_version_id_for(authorization)

    prepared = layout.prepared_dir
    if prepared.is_symlink() or not prepared.is_dir():
        raise PreparationError(f"prepared directory is unavailable: {prepared}")
    names = {path.name for path in prepared.iterdir()}
    if names != set(FINAL_ARTIFACT_NAMES):
        raise PreparationError("prepared directory has missing or unexpected entries")
    for name in FINAL_ARTIFACT_NAMES:
        path = prepared / name
        if path.is_symlink() or not path.is_file():
            raise PreparationError(f"prepared artifact is not a regular file: {name}")

    manifest_path = prepared / "dataset_manifest.json"
    quality_path = prepared / "quality_report.json"
    index_path = prepared / "index.csv"
    manifest = _load_canonical_json_object(
        manifest_path, description="dataset manifest"
    )
    quality = _load_canonical_json_object(
        quality_path, description="final quality report"
    )
    _validate_manifest_shape(manifest)
    _validate_quality_shape(quality)
    if manifest.get("schema_version") != "1.0":
        raise PreparationError("manifest schema_version is incompatible")
    self_digest = manifest.get("manifest_sha256")
    if type(self_digest) is not str or self_digest != manifest_sha256(
        manifest, digest_field="manifest_sha256"
    ):
        raise PreparationError("manifest self-digest is invalid")

    try:
        index_bytes = index_path.read_bytes()
    except OSError as error:
        raise PreparationError(f"cannot read final index: {error}") from error
    rows = _parse_canonical_index(index_bytes)
    index_digest = sha256_hex(index_bytes)
    quality_bytes = quality_path.read_bytes()
    quality_digest = sha256_hex(quality_bytes)
    artifacts = manifest["artifacts"]
    if type(artifacts) is not dict:
        raise PreparationError("manifest artifacts must be an object")
    index_reference = artifacts.get("index")
    quality_reference = artifacts.get("quality_report")
    if index_reference != {
        "path": "index.csv",
        "sha256": index_digest,
        "row_count": len(rows),
    }:
        raise PreparationError("manifest index reference or digest is invalid")
    if quality_reference != {
        "path": "quality_report.json",
        "sha256": quality_digest,
    }:
        raise PreparationError("manifest quality report reference or digest is invalid")

    counts = _validate_final_rows(
        rows,
        raw_root=layout.raw_root,
        dataset_version_id=dataset_version_id,
    )
    if counts != expected_split_counts:
        raise PreparationError("final split counts differ from approved counts")
    _validate_quality_against_rows(
        quality,
        rows,
        dataset_version_id=dataset_version_id,
        expected_counts=expected_split_counts,
        expected_perceptual_candidate_count=expected_perceptual_candidate_count,
    )
    _validate_manifest_contract(
        manifest,
        config=config,
        layout=layout,
        dataset_version_id=dataset_version_id,
        expected_counts=expected_split_counts,
        quality=quality,
        actual_source_digests=actual_source_digests,
    )
    accepted_count = sum(
        count for split in counts.values() for count in split.values()
    )
    return ValidationResult(
        dataset_version_id=dataset_version_id,
        discovered_file_count=len(rows),
        accepted_image_count=accepted_count,
        counts_by_split_and_class=counts,
        index_sha256=index_digest,
        quality_report_sha256=quality_digest,
        manifest_sha256=self_digest,
        manifest_file_sha256=sha256_file(manifest_path),
    )


def validate_default_dataset(project_root: str | Path = ".") -> ValidationResult:
    return validate_prepared_dataset(
        PreparationLayout.default(project_root),
        approved_source_digests=REAL_APPROVED_SOURCE_DIGESTS,
        expected_split_counts=REAL_EXPECTED_SPLIT_COUNTS,
        expected_perceptual_candidate_count=REAL_PERCEPTUAL_CANDIDATE_COUNT,
    )


_MANIFEST_FIELDS = {
    "artifact_type",
    "schema_version",
    "manifest_sha256",
    "dataset_version_id",
    "dataset",
    "license",
    "acquisition",
    "human_authorization",
    "class_mapping",
    "positive_class",
    "split",
    "preprocessing",
    "duplicates",
    "counts",
    "artifacts",
    "source_artifacts",
    "producer",
    "limitations",
    "methodological_exception",
}
_QUALITY_FIELDS = {
    "artifact_type",
    "schema_version",
    "dataset_version_id",
    "total_discovered_files",
    "reconciled_files",
    "counts_by_status",
    "counts_by_class",
    "counts_by_source_partition",
    "counts_by_split",
    "counts_by_format",
    "counts_by_reason",
    "accepted_counts_by_split_and_class",
    "excluded_occurrences",
    "review_occurrences",
    "exact_deduplication",
    "perceptual_audit",
    "perceptual_limitation_accepted_for_didactic_use",
    "originals_modified",
    "final_release_blocked",
}


def _validate_manifest_shape(manifest: dict[str, object]) -> None:
    if set(manifest) != _MANIFEST_FIELDS:
        raise PreparationError("dataset manifest fields do not match schema")
    if manifest.get("artifact_type") != "dataset_manifest":
        raise PreparationError("unexpected dataset manifest artifact_type")


def _validate_quality_shape(quality: dict[str, object]) -> None:
    if set(quality) != _QUALITY_FIELDS:
        raise PreparationError("quality report fields do not match schema")
    if quality.get("artifact_type") != "final_quality_report":
        raise PreparationError("unexpected quality report artifact_type")
    if quality.get("schema_version") != "1.0":
        raise PreparationError("quality report schema_version is incompatible")


def _parse_canonical_index(content: bytes) -> list[dict[str, object]]:
    try:
        text = content.decode("utf-8")
        reader = csv.DictReader(StringIO(text, newline=""))
        if tuple(reader.fieldnames or ()) != INDEX_CSV_HEADER:
            raise PreparationError("index CSV header is incompatible")
        rows = [_parse_index_row(row, number=index) for index, row in enumerate(reader, 1)]
    except UnicodeError as error:
        raise PreparationError("index CSV is not valid UTF-8") from error
    try:
        canonical = canonical_index_csv_bytes(rows)
    except CanonicalizationError as error:
        raise PreparationError(f"index CSV violates the canonical contract: {error}") from error
    if canonical != content:
        raise PreparationError("index CSV is not canonical")
    return rows


def _parse_index_row(row: dict[str, str | None], *, number: int) -> dict[str, object]:
    if set(row) != set(INDEX_CSV_HEADER) or any(value is None for value in row.values()):
        raise PreparationError(f"index row {number} does not match the schema")
    parsed: dict[str, object] = {field: str(row[field]) for field in INDEX_CSV_HEADER}
    for field in ("class_id", "width", "height", "channels"):
        value = str(row[field])
        if value:
            try:
                parsed[field] = int(value)
            except ValueError as error:
                raise PreparationError(
                    f"index row {number} field {field} must be an integer"
                ) from error
    for field in ("is_animated", "exif_orientation_applied"):
        value = str(row[field])
        if value:
            if value not in {"true", "false"}:
                raise PreparationError(
                    f"index row {number} field {field} must be a boolean"
                )
            parsed[field] = value == "true"
    return parsed


def _validate_final_rows(
    rows: list[dict[str, object]],
    *,
    raw_root: Path,
    dataset_version_id: str,
) -> dict[str, dict[str, int]]:
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise PreparationError("raw dataset root is unavailable")
    sample_ids: set[str] = set()
    relative_paths: set[str] = set()
    accepted_hashes: set[str] = set()
    accepted_groups: dict[str, set[str]] = defaultdict(set)
    resolved_root = raw_root.resolve()
    for number, row in enumerate(rows, 1):
        sample_id = row["sample_id"]
        relative_path = row["relative_path"]
        if type(sample_id) is not str or not sample_id:
            raise PreparationError(f"index row {number} has an invalid sample_id")
        if sample_id in sample_ids:
            raise PreparationError(f"duplicate sample_id in final index: {sample_id}")
        sample_ids.add(sample_id)
        if type(relative_path) is not str:
            raise PreparationError(f"index row {number} has an invalid path")
        try:
            normalized = normalize_relative_path(relative_path)
        except ValueError as error:
            raise PreparationError(
                f"index row {number} has an unsafe path: {error}"
            ) from error
        if normalized != relative_path or relative_path in relative_paths:
            raise PreparationError(f"index row {number} path is duplicate or noncanonical")
        relative_paths.add(relative_path)
        if sample_id != make_sample_id(dataset_version_id, relative_path):
            raise PreparationError(f"index row {number} sample_id is inconsistent")
        source = raw_root / relative_path
        if source.is_symlink() or not source.is_file():
            raise PreparationError(f"index row {number} source file is unavailable")
        try:
            source.resolve().relative_to(resolved_root)
        except ValueError as error:
            raise PreparationError(f"index row {number} escapes raw root") from error
        file_hash = row["sha256"]
        if (
            type(file_hash) is not str
            or len(file_hash) != 64
            or any(character not in "0123456789abcdef" for character in file_hash)
            or sha256_file(source) != file_hash
        ):
            raise PreparationError(f"index row {number} has an invalid source SHA-256")

        status = row["status"]
        if status not in {AuditStatus.ACCEPTED.value, AuditStatus.EXCLUDED.value}:
            raise PreparationError(f"index row {number} has an unreleased status")
        if row["perceptual_method"] or row["perceptual_evidence"]:
            raise PreparationError(
                "unconfirmed perceptual candidates must not become index groups"
            )
        if status == AuditStatus.EXCLUDED.value:
            if row["split"]:
                raise PreparationError(f"excluded index row {number} has a split")
            if not row["reason"]:
                raise PreparationError(f"excluded index row {number} lacks a reason")
            continue

        class_name = row["class_name"]
        class_id = row["class_id"]
        source_partition = row["source_partition"]
        split = row["split"]
        if (
            type(class_name) is not str
            or class_name not in CLASS_IDS
            or type(class_id) is not int
            or class_id != CLASS_IDS[class_name]
        ):
            raise PreparationError(f"accepted index row {number} has invalid class")
        if source_partition not in {"train", "test"}:
            raise PreparationError(
                f"accepted index row {number} has invalid source partition"
            )
        if split not in {item.value for item in DatasetSplit}:
            raise PreparationError(f"accepted index row {number} has invalid split")
        if source_partition == "test" and split != DatasetSplit.TEST.value:
            raise PreparationError("original test sample was redistributed")
        if source_partition == "train" and split == DatasetSplit.TEST.value:
            raise PreparationError("test contains a source-train sample")
        if not row["format"] or not all(
            type(row[field]) is int and int(row[field]) > 0
            for field in ("width", "height", "channels")
        ):
            raise PreparationError(f"accepted index row {number} lacks image metadata")
        if type(row["is_animated"]) is not bool or row["is_animated"]:
            raise PreparationError(f"accepted index row {number} is animated or invalid")
        if type(row["exif_orientation_applied"]) is not bool:
            raise PreparationError(
                f"accepted index row {number} lacks EXIF contract metadata"
            )
        if file_hash in accepted_hashes:
            raise PreparationError("accepted SHA-256 appears more than once")
        accepted_hashes.add(file_hash)
        duplicate_group = row["duplicate_group_id"]
        if duplicate_group:
            accepted_groups[str(duplicate_group)].add(str(split))
    if any(len(splits) > 1 for splits in accepted_groups.values()):
        raise PreparationError("confirmed duplicate group crosses final splits")
    discovered = {
        path.relative_to(raw_root).as_posix()
        for path in raw_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if discovered != relative_paths:
        raise PreparationError("final index does not reconcile the raw source tree")
    return _accepted_counts_by_split_and_class(rows)


def _validate_quality_against_rows(
    quality: dict[str, object],
    rows: list[dict[str, object]],
    *,
    dataset_version_id: str,
    expected_counts: dict[str, dict[str, int]],
    expected_perceptual_candidate_count: int,
) -> None:
    status_counts = Counter(str(row["status"]) for row in rows)
    class_counts = Counter(str(row["class_name"] or "unidentified") for row in rows)
    source_counts = Counter(
        str(row["source_partition"] or "none") for row in rows
    )
    split_counts = Counter(str(row["split"] or "none") for row in rows)
    format_counts = Counter(str(row["format"] or "unidentified") for row in rows)
    reason_counts = Counter(
        reason
        for row in rows
        for reason in str(row["reason"]).split(";")
        if reason
    )
    expected_excluded = [
        {
            "sample_id": str(row["sample_id"]),
            "relative_path": str(row["relative_path"]),
            "reason": str(row["reason"]),
        }
        for row in rows
        if row["status"] == AuditStatus.EXCLUDED.value
    ]
    expected_values: dict[str, object] = {
        "dataset_version_id": dataset_version_id,
        "total_discovered_files": len(rows),
        "reconciled_files": len(rows),
        "counts_by_status": dict(sorted(status_counts.items())),
        "counts_by_class": dict(sorted(class_counts.items())),
        "counts_by_source_partition": dict(sorted(source_counts.items())),
        "counts_by_split": dict(sorted(split_counts.items())),
        "counts_by_format": dict(sorted(format_counts.items())),
        "counts_by_reason": dict(sorted(reason_counts.items())),
        "accepted_counts_by_split_and_class": expected_counts,
        "excluded_occurrences": expected_excluded,
        "review_occurrences": [],
        "perceptual_limitation_accepted_for_didactic_use": True,
        "originals_modified": False,
        "final_release_blocked": False,
    }
    for field, expected in expected_values.items():
        if quality.get(field) != expected:
            raise PreparationError(f"quality report field {field!r} diverged")
    exact = quality.get("exact_deduplication")
    if type(exact) is not dict or exact.get("blocked_sample_count") != 0 or exact.get(
        "blocking_review_ids"
    ) != []:
        raise PreparationError("quality report contains blocking exact conflicts")
    perceptual = quality.get("perceptual_audit")
    if type(perceptual) is not dict or any(
        perceptual.get(field) != expected
        for field, expected in {
            "candidate_pair_count": expected_perceptual_candidate_count,
            "reviewed_candidate_count": 0,
            "candidates_are_confirmed_duplicates": False,
            "used_for_exclusion": False,
            "used_for_grouping": False,
            "used_for_split_assignment": False,
        }.items()
    ):
        raise PreparationError("quality report perceptual limitation is invalid")


def _validate_manifest_contract(
    manifest: dict[str, object],
    *,
    config: ProjectConfig,
    layout: PreparationLayout,
    dataset_version_id: str,
    expected_counts: dict[str, dict[str, int]],
    quality: dict[str, object],
    actual_source_digests: dict[str, str],
) -> None:
    if manifest.get("dataset_version_id") != dataset_version_id:
        raise PreparationError("manifest dataset version diverged")
    if manifest.get("class_mapping") != {"chihuahua": 0, "muffin": 1}:
        raise PreparationError("manifest class mapping is invalid")
    if manifest.get("positive_class") != "muffin":
        raise PreparationError("manifest positive class is invalid")
    dataset = manifest.get("dataset")
    if type(dataset) is not dict or any(
        dataset.get(field) != expected
        for field, expected in {
            "source": config.dataset.source,
            "source_id": config.dataset.source_id,
            "handle": config.dataset.handle,
            "owner": config.dataset.owner,
            "version": config.dataset.version,
            "version_status": "confirmed",
            "source_url": config.dataset.source_url,
            "metadata_url": config.dataset.metadata_url,
            "metadata_checked_at": config.dataset.metadata_checked_at,
            "raw_root": _logical_path(layout.raw_root, layout.project_root),
        }.items()
    ):
        raise PreparationError("manifest dataset provenance diverged")
    license_record = manifest.get("license")
    if type(license_record) is not dict or any(
        license_record.get(field) != expected
        for field, expected in {
            "declared_value": config.dataset.license,
            "status": "ambiguous",
            "note": config.dataset.license_note,
            "confirmed": False,
        }.items()
    ):
        raise PreparationError("manifest license state is invalid")
    authorization = manifest.get("human_authorization")
    if type(authorization) is not dict or any(
        authorization.get(field) != expected
        for field, expected in {
            "status": "approved",
            "purpose": "educational_local_use",
            "license_ambiguity_acknowledged": True,
            "local_use_only": True,
            "redistribution_allowed": False,
        }.items()
    ):
        raise PreparationError("manifest human authorization is invalid")
    acquisition = manifest.get("acquisition")
    if type(acquisition) is not dict or any(
        acquisition.get(field) != expected
        for field, expected in {
            "acquired_at": config.dataset.acquired_at,
            "tool": config.dataset.acquisition_tool,
            "tool_version": config.dataset.acquisition_tool_version,
            "file_count": _quality_total_count(quality),
            "originals_preserved": True,
            "download_performed_by_increment_8": False,
        }.items()
    ):
        raise PreparationError("manifest acquisition evidence is invalid")
    split = manifest.get("split")
    if type(split) is not dict or any(
        split.get(field) != expected
        for field, expected in {
            "scenario": "A",
            "seed": 42,
            "algorithm": "stratified_sha256_rank_v1",
            "source_partitions_preserved": ["train", "test"],
            "validation_source": "train",
            "requested_validation_fraction_of_source_train": 0.15,
            "counts_by_split_and_class": expected_counts,
            "test_original_preserved": True,
        }.items()
    ):
        raise PreparationError("manifest split contract is invalid")
    if manifest.get("preprocessing") != _preprocessing_payload(config):
        raise PreparationError("manifest preprocessing contract is invalid")
    duplicates = manifest.get("duplicates")
    if type(duplicates) is not dict or duplicates.get("exact") != quality.get(
        "exact_deduplication"
    ) or duplicates.get("perceptual") != quality.get("perceptual_audit"):
        raise PreparationError("manifest duplicate evidence diverged")
    expected_sources = {
        name: {
            "path": _logical_path(path, layout.project_root),
            "sha256": actual_source_digests[name],
        }
        for name, path in sorted(layout.source_artifacts().items())
    }
    if manifest.get("source_artifacts") != expected_sources:
        raise PreparationError("manifest source artifact evidence diverged")
    limitation = manifest.get("methodological_exception")
    if type(limitation) is not dict or any(
        limitation.get(field) != expected
        for field, expected in {
            "purpose": "didactic_experiment",
            "perceptual_review_waived": True,
            "release_blocking": False,
            "exact_integrity_errors_remain_blocking": True,
        }.items()
    ):
        raise PreparationError("manifest methodological exception is invalid")
    counts = manifest.get("counts")
    accepted_count = sum(
        count for values in expected_counts.values() for count in values.values()
    )
    if type(counts) is not dict or any(
        counts.get(field) != expected
        for field, expected in {
            "discovered_files": quality["total_discovered_files"],
            "accepted_images": accepted_count,
            "by_status": quality["counts_by_status"],
            "by_split_and_class": expected_counts,
        }.items()
    ):
        raise PreparationError("manifest counts diverged")


def _quality_total_count(quality: dict[str, object]) -> int:
    count = quality.get("total_discovered_files")
    if type(count) is not int or count < 0:
        raise PreparationError("quality report total count is invalid")
    return count


def _print_validation(result: ValidationResult) -> None:
    print(f"dataset_version_id={result.dataset_version_id}")
    print(f"discovered_files={result.discovered_file_count}")
    print(f"accepted_images={result.accepted_image_count}")
    for split in ("train", "validation", "test"):
        values = result.counts_by_split_and_class[split]
        print(
            f"{split}: chihuahua={values['chihuahua']} "
            f"muffin={values['muffin']} total={sum(values.values())}"
        )
    print(f"index_sha256={result.index_sha256}")
    print(f"quality_report_sha256={result.quality_report_sha256}")
    print(f"manifest_sha256={result.manifest_sha256}")
    print(f"manifest_file_sha256={result.manifest_file_sha256}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "validate"))
    parser.add_argument("--project-root", default=".")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "generate":
            result = prepare_default_dataset(arguments.project_root)
            print(f"publication_status={result.publication.status}")
            print(f"prepared_dir={result.publication.destination}")
            _print_validation(result.validation)
        else:
            _print_validation(validate_default_dataset(arguments.project_root))
    except (PreparationError, ValueError, OSError) as error:
        parser.exit(2, f"ERRO: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
