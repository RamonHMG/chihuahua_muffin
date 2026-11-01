import json
from pathlib import Path

import pytest

from chihuahua_muffin.data.review import (
    HumanReviewDecision,
    ReviewDecisionState,
    ReviewError,
    ReviewOccurrenceType,
    canonical_human_decisions_bytes,
    load_human_decisions,
)


VERSION_ID = "kaggle:2733586:v2"


def _decision(**overrides: object) -> HumanReviewDecision:
    values: dict[str, object] = {
        "review_id": "exact-sha256:abc",
        "target_sample_ids": ("a", "b"),
        "occurrence_type": ReviewOccurrenceType.EXACT_LABEL_CONFLICT,
        "decision": ReviewDecisionState.CONFIRMED_DUPLICATE,
        "justification": "Os arquivos possuem bytes idênticos.",
        "reviewer": "reviewer@example",
        "decision_date": "2026-11-01",
        "retained_sample_ids": ("a",),
        "excluded_sample_ids": ("b",),
    }
    values.update(overrides)
    return HumanReviewDecision(**values)  # type: ignore[arg-type]


def test_confirmed_duplicate_records_separate_sample_disposition() -> None:
    decision = _decision()

    assert decision.decision is ReviewDecisionState.CONFIRMED_DUPLICATE
    assert decision.retained_sample_ids == ("a",)
    assert decision.excluded_sample_ids == ("b",)
    assert decision.has_resolved_disposition is True


def test_confirmed_duplicate_may_remain_blocking_without_disposition() -> None:
    decision = _decision(retained_sample_ids=(), excluded_sample_ids=())

    assert decision.decision is ReviewDecisionState.CONFIRMED_DUPLICATE
    assert decision.has_resolved_disposition is False


def test_not_duplicate_and_uncertain_states_have_explicit_invariants() -> None:
    not_duplicate = _decision(
        decision=ReviewDecisionState.NOT_DUPLICATE,
        retained_sample_ids=("b", "a"),
        excluded_sample_ids=(),
    )
    uncertain = _decision(
        decision=ReviewDecisionState.UNCERTAIN,
        retained_sample_ids=(),
        excluded_sample_ids=(),
    )

    assert not_duplicate.target_sample_ids == ("a", "b")
    assert not_duplicate.retained_sample_ids == ("a", "b")
    assert uncertain.has_resolved_disposition is False


def test_decisions_serialize_canonically_independent_of_input_order() -> None:
    first = _decision(review_id="group-b")
    second = _decision(review_id="group-a")

    forward = canonical_human_decisions_bytes(VERSION_ID, [first, second])
    reverse = canonical_human_decisions_bytes(VERSION_ID, [second, first])

    assert forward == reverse
    assert not forward.endswith(b"\n")
    assert json.loads(forward)["decisions"][0]["review_id"] == "group-a"


def test_human_decisions_round_trip_from_local_json(tmp_path: Path) -> None:
    decisions = (_decision(),)
    path = tmp_path / "human_decisions.json"
    path.write_bytes(canonical_human_decisions_bytes(VERSION_ID, decisions))

    loaded = load_human_decisions(
        path,
        expected_dataset_version_id=VERSION_ID,
    )

    assert loaded == decisions


@pytest.mark.parametrize(
    "overrides",
    [
        {"justification": ""},
        {"reviewer": ""},
        {"decision_date": "01/11/2026"},
        {"target_sample_ids": ("a",)},
        {"target_sample_ids": ("a", "a")},
        {
            "decision": ReviewDecisionState.UNCERTAIN,
            "retained_sample_ids": ("a",),
            "excluded_sample_ids": ("b",),
        },
        {
            "decision": ReviewDecisionState.NOT_DUPLICATE,
            "retained_sample_ids": ("a",),
            "excluded_sample_ids": ("b",),
        },
        {"retained_sample_ids": ("a",), "excluded_sample_ids": ()},
        {"retained_sample_ids": ("a",), "excluded_sample_ids": ("a",)},
    ],
)
def test_invalid_human_decisions_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ReviewError):
        _decision(**overrides)


def test_duplicate_review_ids_are_rejected() -> None:
    decision = _decision()

    with pytest.raises(ReviewError, match="unique"):
        canonical_human_decisions_bytes(VERSION_ID, [decision, decision])


def test_loader_rejects_unknown_state_and_wrong_dataset(tmp_path: Path) -> None:
    payload = json.loads(canonical_human_decisions_bytes(VERSION_ID, [_decision()]))
    payload["decisions"][0]["decision"] = "approved"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ReviewError, match="unknown"):
        load_human_decisions(path)

    valid_path = tmp_path / "valid.json"
    valid_path.write_bytes(canonical_human_decisions_bytes(VERSION_ID, []))
    with pytest.raises(ReviewError, match="different dataset"):
        load_human_decisions(valid_path, expected_dataset_version_id="other:v1")
