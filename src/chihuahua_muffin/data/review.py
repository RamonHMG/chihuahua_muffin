"""Canonical local records for auditable human review decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
import json
from pathlib import Path

from .canonical import canonical_json_bytes


class ReviewError(ValueError):
    """Raised when a human-review record is incomplete or inconsistent."""


class ReviewDecisionState(StrEnum):
    """Allowed human conclusions about a duplicate occurrence."""

    CONFIRMED_DUPLICATE = "confirmed_duplicate"
    NOT_DUPLICATE = "not_duplicate"
    UNCERTAIN = "uncertain"


class ReviewOccurrenceType(StrEnum):
    """Duplicate occurrence types supported by the human-review registry."""

    EXACT_SAME_LABEL = "exact_duplicate_same_label"
    EXACT_LABEL_CONFLICT = "exact_duplicate_label_conflict"
    PERCEPTUAL_SAME_LABEL = "perceptual_duplicate_same_label"
    PERCEPTUAL_LABEL_CONFLICT = "perceptual_duplicate_label_conflict"


_DECISION_FIELDS = {
    "review_id",
    "target_sample_ids",
    "occurrence_type",
    "decision",
    "justification",
    "reviewer",
    "decision_date",
    "retained_sample_ids",
    "excluded_sample_ids",
}


@dataclass(frozen=True, slots=True)
class HumanReviewDecision:
    """One review conclusion plus an independent sample disposition.

    ``decision`` answers whether the occurrence is a duplicate. The retained
    and excluded IDs separately answer what happens to the samples. A confirmed
    label conflict without a complete disposition remains valid evidence but is
    still blocking for training.
    """

    review_id: str
    target_sample_ids: tuple[str, ...]
    occurrence_type: ReviewOccurrenceType
    decision: ReviewDecisionState
    justification: str
    reviewer: str
    decision_date: str
    retained_sample_ids: tuple[str, ...] = ()
    excluded_sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("review_id", "justification", "reviewer", "decision_date"):
            value = getattr(self, field)
            if type(value) is not str or not value.strip():
                raise ReviewError(f"{field} must be a non-empty string")
        if not isinstance(self.occurrence_type, ReviewOccurrenceType):
            raise ReviewError("occurrence_type must be ReviewOccurrenceType")
        if not isinstance(self.decision, ReviewDecisionState):
            raise ReviewError("decision must be ReviewDecisionState")
        try:
            date.fromisoformat(self.decision_date)
        except ValueError as error:
            raise ReviewError("decision_date must use YYYY-MM-DD") from error

        targets = _normalized_ids(self.target_sample_ids, "target_sample_ids")
        retained = _normalized_ids(
            self.retained_sample_ids,
            "retained_sample_ids",
            allow_empty=True,
        )
        excluded = _normalized_ids(
            self.excluded_sample_ids,
            "excluded_sample_ids",
            allow_empty=True,
        )
        if len(targets) < 2:
            raise ReviewError("target_sample_ids must identify at least two samples")
        target_set = set(targets)
        retained_set = set(retained)
        excluded_set = set(excluded)
        if retained_set & excluded_set:
            raise ReviewError("retained and excluded sample IDs must be disjoint")
        if not retained_set | excluded_set <= target_set:
            raise ReviewError("sample disposition must refer only to target IDs")

        if self.decision is ReviewDecisionState.UNCERTAIN:
            if retained or excluded:
                raise ReviewError("uncertain decisions cannot define a disposition")
        elif self.decision is ReviewDecisionState.NOT_DUPLICATE:
            if retained != targets or excluded:
                raise ReviewError(
                    "not_duplicate must retain every target and exclude none"
                )
        elif retained or excluded:
            if retained_set | excluded_set != target_set:
                raise ReviewError(
                    "a duplicate disposition must account for every target"
                )
            if len(retained) != 1:
                raise ReviewError(
                    "a resolved duplicate disposition must retain exactly one sample"
                )

        object.__setattr__(self, "target_sample_ids", targets)
        object.__setattr__(self, "retained_sample_ids", retained)
        object.__setattr__(self, "excluded_sample_ids", excluded)

    @property
    def has_resolved_disposition(self) -> bool:
        """Whether every target has an explicit keep/exclude outcome."""

        return bool(self.retained_sample_ids or self.excluded_sample_ids)

    def as_json(self) -> dict[str, object]:
        return {
            "review_id": self.review_id,
            "target_sample_ids": list(self.target_sample_ids),
            "occurrence_type": self.occurrence_type.value,
            "decision": self.decision.value,
            "justification": self.justification,
            "reviewer": self.reviewer,
            "decision_date": self.decision_date,
            "retained_sample_ids": list(self.retained_sample_ids),
            "excluded_sample_ids": list(self.excluded_sample_ids),
        }

    @classmethod
    def from_json(cls, value: object) -> HumanReviewDecision:
        if type(value) is not dict:
            raise ReviewError("each human decision must be a JSON object")
        if set(value) != _DECISION_FIELDS:
            missing = sorted(_DECISION_FIELDS - set(value))
            extra = sorted(set(value) - _DECISION_FIELDS)
            raise ReviewError(
                f"human decision fields do not match schema; missing={missing}, extra={extra}"
            )
        try:
            occurrence_type = ReviewOccurrenceType(value["occurrence_type"])
            decision = ReviewDecisionState(value["decision"])
        except (TypeError, ValueError) as error:
            raise ReviewError("unknown occurrence_type or decision") from error
        return cls(
            review_id=_required_string(value["review_id"], "review_id"),
            target_sample_ids=_json_id_list(
                value["target_sample_ids"], "target_sample_ids"
            ),
            occurrence_type=occurrence_type,
            decision=decision,
            justification=_required_string(value["justification"], "justification"),
            reviewer=_required_string(value["reviewer"], "reviewer"),
            decision_date=_required_string(value["decision_date"], "decision_date"),
            retained_sample_ids=_json_id_list(
                value["retained_sample_ids"], "retained_sample_ids"
            ),
            excluded_sample_ids=_json_id_list(
                value["excluded_sample_ids"], "excluded_sample_ids"
            ),
        )


def human_decisions_payload(
    dataset_version_id: str,
    decisions: tuple[HumanReviewDecision, ...] | list[HumanReviewDecision],
) -> dict[str, object]:
    """Build a deterministic local decisions artifact."""

    if type(dataset_version_id) is not str or not dataset_version_id.strip():
        raise ReviewError("dataset_version_id must be a non-empty string")
    ordered = sorted(decisions, key=lambda decision: decision.review_id)
    review_ids = [decision.review_id for decision in ordered]
    if len(review_ids) != len(set(review_ids)):
        raise ReviewError("review_id values must be unique")
    return {
        "artifact_type": "human_review_decisions",
        "artifact_version": "1.0",
        "dataset_version_id": dataset_version_id,
        "decisions": [decision.as_json() for decision in ordered],
    }


def canonical_human_decisions_bytes(
    dataset_version_id: str,
    decisions: tuple[HumanReviewDecision, ...] | list[HumanReviewDecision],
) -> bytes:
    return canonical_json_bytes(human_decisions_payload(dataset_version_id, decisions))


def load_human_decisions(
    path: str | Path,
    *,
    expected_dataset_version_id: str | None = None,
) -> tuple[HumanReviewDecision, ...]:
    """Load and validate a local JSON decisions artifact."""

    decision_path = Path(path)
    try:
        document = json.loads(decision_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReviewError(f"cannot read human decisions: {error}") from error
    if type(document) is not dict:
        raise ReviewError("human decisions artifact must be a JSON object")
    expected_fields = {
        "artifact_type",
        "artifact_version",
        "dataset_version_id",
        "decisions",
    }
    if set(document) != expected_fields:
        raise ReviewError("human decisions artifact fields do not match schema")
    if document["artifact_type"] != "human_review_decisions":
        raise ReviewError("unexpected human decisions artifact_type")
    if document["artifact_version"] != "1.0":
        raise ReviewError("unsupported human decisions artifact_version")
    dataset_version_id = _required_string(
        document["dataset_version_id"], "dataset_version_id"
    )
    if (
        expected_dataset_version_id is not None
        and dataset_version_id != expected_dataset_version_id
    ):
        raise ReviewError("human decisions belong to a different dataset version")
    if type(document["decisions"]) is not list:
        raise ReviewError("decisions must be a JSON list")
    decisions = tuple(
        HumanReviewDecision.from_json(value) for value in document["decisions"]
    )
    human_decisions_payload(dataset_version_id, decisions)
    return tuple(sorted(decisions, key=lambda decision: decision.review_id))


def _normalized_ids(
    values: tuple[str, ...],
    field: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(values) is not tuple or any(
        type(value) is not str or not value.strip() for value in values
    ):
        raise ReviewError(f"{field} must be a tuple of non-empty strings")
    if not allow_empty and not values:
        raise ReviewError(f"{field} must not be empty")
    if len(values) != len(set(values)):
        raise ReviewError(f"{field} must not contain duplicates")
    return tuple(sorted(values))


def _json_id_list(value: object, field: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ReviewError(f"{field} must be a JSON list of strings")
    return tuple(value)


def _required_string(value: object, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ReviewError(f"{field} must be a non-empty string")
    return value
