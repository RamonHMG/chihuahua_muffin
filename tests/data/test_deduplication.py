from dataclasses import replace
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from chihuahua_muffin.data.deduplication import (
    DeduplicationError,
    ExactOccurrenceType,
    SampleAction,
    analyze_exact_duplicates,
    assess_validation_feasibility,
    write_exact_audit,
)
from chihuahua_muffin.data.inventory import inventory_dataset
from chihuahua_muffin.data.review import (
    HumanReviewDecision,
    ReviewDecisionState,
    ReviewOccurrenceType,
    load_human_decisions,
)


VERSION_ID = "kaggle:2733586:v2"


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), color).save(path)
    return path


def _copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def _conflict_inventory(tmp_path: Path):
    source = _image(tmp_path / "train/chihuahua/original.png", (1, 2, 3))
    _copy(source, tmp_path / "test/muffin/copied.png")
    return inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)


def _decision_for_group(
    group: object,
    *,
    state: ReviewDecisionState = ReviewDecisionState.CONFIRMED_DUPLICATE,
    resolved: bool = True,
) -> HumanReviewDecision:
    outcomes = group.outcomes  # type: ignore[attr-defined]
    targets = tuple(sorted(outcome.sample_id for outcome in outcomes))
    test_id = next(
        outcome.sample_id for outcome in outcomes if outcome.source_partition == "test"
    )
    excluded = tuple(sample_id for sample_id in targets if sample_id != test_id)
    if state is ReviewDecisionState.UNCERTAIN or not resolved:
        retained: tuple[str, ...] = ()
        excluded = ()
    elif state is ReviewDecisionState.NOT_DUPLICATE:
        retained = targets
        excluded = ()
    else:
        retained = (test_id,)
    return HumanReviewDecision(
        review_id=group.group_id,  # type: ignore[attr-defined]
        target_sample_ids=targets,
        occurrence_type=ReviewOccurrenceType.EXACT_LABEL_CONFLICT,
        decision=state,
        justification="Revisão dos arquivos byte a byte e dos rótulos.",
        reviewer="reviewer@example",
        decision_date="2026-11-01",
        retained_sample_ids=retained,
        excluded_sample_ids=excluded,
    )


def test_same_bytes_with_different_names_form_one_exact_group(tmp_path: Path) -> None:
    source = _image(tmp_path / "train/chihuahua/b.png", (10, 20, 30))
    _copy(source, tmp_path / "train/chihuahua/a.png")

    result = analyze_exact_duplicates(
        inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)
    )

    assert len(result.groups) == 1
    group = result.groups[0]
    assert group.occurrence_type is ExactOccurrenceType.SAME_LABEL
    assert {outcome.action for outcome in group.outcomes} == {
        SampleAction.KEEP,
        SampleAction.EXCLUDE,
    }
    kept = next(outcome for outcome in group.outcomes if outcome.action is SampleAction.KEEP)
    assert kept.relative_path == "train/chihuahua/a.png"
    assert result.excluded_sample_count == 1


def test_test_partition_is_preferred_over_train_for_same_label_duplicate(
    tmp_path: Path,
) -> None:
    source = _image(tmp_path / "train/muffin/a.png", (10, 20, 30))
    _copy(source, tmp_path / "test/muffin/z.png")

    result = analyze_exact_duplicates(
        inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)
    )
    outcomes = result.groups[0].outcomes
    kept = next(outcome for outcome in outcomes if outcome.action is SampleAction.KEEP)
    excluded = next(
        outcome for outcome in outcomes if outcome.action is SampleAction.EXCLUDE
    )

    assert kept.source_partition == "test"
    assert excluded.source_partition == "train"
    assert excluded.reason == "exact_duplicate_of_preserved_test"
    assert result.groups[0].crosses_source_partitions is True


def test_label_conflict_blocks_training_until_human_review(tmp_path: Path) -> None:
    result = analyze_exact_duplicates(_conflict_inventory(tmp_path))

    assert result.groups[0].occurrence_type is ExactOccurrenceType.LABEL_CONFLICT
    assert all(
        outcome.action is SampleAction.BLOCK for outcome in result.groups[0].outcomes
    )
    assert result.training_release_blocked is True
    assert result.blocked_sample_count == 2


def test_confirmed_conflict_still_blocks_without_sample_disposition(
    tmp_path: Path,
) -> None:
    inventory = _conflict_inventory(tmp_path)
    preliminary = analyze_exact_duplicates(inventory)
    decision = _decision_for_group(preliminary.groups[0], resolved=False)

    result = analyze_exact_duplicates(inventory, human_decisions=(decision,))

    assert result.training_release_blocked is True
    assert {outcome.reason for outcome in result.groups[0].outcomes} == {
        "confirmed_duplicate_requires_sample_disposition"
    }


def test_confirmed_conflict_with_disposition_is_resolved_without_relabeling(
    tmp_path: Path,
) -> None:
    inventory = _conflict_inventory(tmp_path)
    preliminary = analyze_exact_duplicates(inventory)
    decision = _decision_for_group(preliminary.groups[0])

    result = analyze_exact_duplicates(inventory, human_decisions=(decision,))
    group = result.groups[0]

    assert result.training_release_blocked is False
    assert result.excluded_sample_count == 1
    kept = next(outcome for outcome in group.outcomes if outcome.action is SampleAction.KEEP)
    assert kept.source_partition == "test"
    assert kept.class_name == "muffin"
    assert group.human_reviewer == "reviewer@example"


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (ReviewDecisionState.UNCERTAIN, "human_review_uncertain"),
        (
            ReviewDecisionState.NOT_DUPLICATE,
            "not_duplicate_decision_conflicts_with_identical_sha256",
        ),
    ],
)
def test_uncertain_or_contradictory_review_remains_blocking(
    tmp_path: Path, state: ReviewDecisionState, reason: str
) -> None:
    inventory = _conflict_inventory(tmp_path)
    preliminary = analyze_exact_duplicates(inventory)
    decision = _decision_for_group(preliminary.groups[0], state=state)

    result = analyze_exact_duplicates(inventory, human_decisions=(decision,))

    assert result.training_release_blocked is True
    assert {outcome.reason for outcome in result.groups[0].outcomes} == {reason}


def test_mismatched_or_unused_human_decision_is_rejected(tmp_path: Path) -> None:
    inventory = _conflict_inventory(tmp_path)
    preliminary = analyze_exact_duplicates(inventory)
    decision = replace(
        _decision_for_group(preliminary.groups[0]),
        review_id="exact-sha256:other",
    )

    with pytest.raises(DeduplicationError, match="do not match"):
        analyze_exact_duplicates(inventory, human_decisions=(decision,))


def test_exact_analysis_is_reproducible_independent_of_inventory_order(
    tmp_path: Path,
) -> None:
    source = _image(tmp_path / "train/chihuahua/b.png", (1, 2, 3))
    _copy(source, tmp_path / "test/chihuahua/a.png")
    inventory = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)
    reversed_inventory = replace(inventory, records=tuple(reversed(inventory.records)))

    first = analyze_exact_duplicates(inventory)
    second = analyze_exact_duplicates(reversed_inventory)

    assert first.payload() == second.payload()


def test_analysis_never_modifies_original_files(tmp_path: Path) -> None:
    source = _image(tmp_path / "train/chihuahua/a.png", (1, 2, 3))
    duplicate = _copy(source, tmp_path / "train/chihuahua/b.png")
    before = {path: path.read_bytes() for path in (source, duplicate)}

    analyze_exact_duplicates(inventory_dataset(tmp_path, dataset_version_id=VERSION_ID))

    assert {path: path.read_bytes() for path in (source, duplicate)} == before


def test_preliminary_validation_assessment_estimates_counts_without_split(
    tmp_path: Path,
) -> None:
    for class_name, base in (("chihuahua", 10), ("muffin", 100)):
        for index in range(4):
            _image(
                tmp_path / f"train/{class_name}/{index}.png",
                (base + index, index, 0),
            )
        _image(tmp_path / f"test/{class_name}/test.png", (base, 50, 50))
    result = analyze_exact_duplicates(
        inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)
    )

    assessment = assess_validation_feasibility(result)
    payload = assessment.payload()

    assert assessment.assessment == "exact_stage_supports_candidate_validation_analysis"
    assert assessment.final_scenario_decision == "deferred_until_perceptual_audit"
    assert assessment.estimated_candidate_validation_counts == {
        "chihuahua": 1,
        "muffin": 1,
    }
    assert assessment.estimated_remaining_train_counts == {
        "chihuahua": 3,
        "muffin": 3,
    }
    assert payload["split_created"] is False


def test_blocking_review_is_reflected_in_validation_assessment(tmp_path: Path) -> None:
    result = analyze_exact_duplicates(_conflict_inventory(tmp_path))

    assessment = assess_validation_feasibility(result)

    assert assessment.assessment == "blocked_pending_exact_human_review"
    assert assessment.exact_blocked_sample_count == 2


def test_exact_audit_writes_nonfinal_artifacts_and_empty_review_registry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _image(source / "train/chihuahua/a.png", (1, 2, 3))
    _image(source / "train/muffin/b.png", (3, 2, 1))
    result = analyze_exact_duplicates(
        inventory_dataset(source, dataset_version_id=VERSION_ID)
    )
    assessment = assess_validation_feasibility(result)
    destination = tmp_path / "audit"

    dedup_path, decisions_path, feasibility_path = write_exact_audit(
        result,
        assessment,
        (),
        destination,
    )

    assert json.loads(dedup_path.read_text())["group_count"] == 0
    assert load_human_decisions(decisions_path) == ()
    assert json.loads(feasibility_path.read_text())["split_created"] is False
    with pytest.raises(DeduplicationError, match="already exists"):
        write_exact_audit(result, assessment, (), destination)
