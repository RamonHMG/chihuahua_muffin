from dataclasses import replace
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance
import pytest

from chihuahua_muffin.data.deduplication import analyze_exact_duplicates
from chihuahua_muffin.data.inventory import inventory_dataset
from chihuahua_muffin.data.perceptual import (
    ComparisonScope,
    PerceptualAuditError,
    analyze_perceptual_duplicates,
    assess_perceptual_validation_feasibility,
    compute_perceptual_hashes,
    render_review_panel,
    write_perceptual_audit,
)
from chihuahua_muffin.data.review import (
    HumanReviewDecision,
    ReviewDecisionState,
)


VERSION_ID = "kaggle:2733586:v2"


def _base_image(variant: int = 0) -> Image.Image:
    image = Image.new("RGB", (128, 96), "white")
    draw = ImageDraw.Draw(image)
    for x in range(128):
        draw.line(
            (x, 0, x, 95),
            fill=(
                (x * 3 + variant * 17) % 256,
                (x * 5 + 40) % 256,
                (x * 7 + 90) % 256,
            ),
        )
    draw.ellipse(
        (18 + variant * 20, 12, 78 + variant * 10, 72),
        fill=(230, 30 + variant * 50, 70),
        outline="black",
        width=4,
    )
    draw.rectangle(
        (72, 38 + variant * 10, 116, 84),
        fill=(20, 180, 80 + variant * 40),
        outline="white",
        width=3,
    )
    draw.line(
        (0, 95, 127, 0 if variant == 0 else 30),
        fill="black",
        width=5,
    )
    return image


def _synthetic_dataset(root: Path) -> tuple[Path, ...]:
    original = _base_image()
    paths = (
        root / "train/chihuahua/original.png",
        root / "test/muffin/recompressed.jpg",
        root / "train/chihuahua/resized.png",
        root / "train/chihuahua/brighter.png",
        root / "train/chihuahua/similar_but_different.png",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    original.save(paths[0])
    original.save(paths[1], quality=35)
    original.resize((73, 55)).save(paths[2])
    ImageEnhance.Brightness(original).enhance(1.08).save(paths[3])
    _base_image(variant=1).save(paths[4])
    original.close()
    return paths


def _audit(root: Path, *, decisions: tuple[HumanReviewDecision, ...] = ()):
    inventory = inventory_dataset(root, dataset_version_id=VERSION_ID)
    exact = analyze_exact_duplicates(inventory)
    result = analyze_perceptual_duplicates(
        root,
        inventory,
        exact,
        candidate_max_distance=4,
        human_decisions=decisions,
        block_size=2,
    )
    return inventory, exact, result


def test_controlled_transformations_are_screened_but_similar_image_is_not(
    tmp_path: Path,
) -> None:
    _synthetic_dataset(tmp_path)

    _, _, result = _audit(tmp_path)
    candidate_paths = {
        frozenset((pair.left.relative_path, pair.right.relative_path))
        for pair in result.candidates
    }
    original_path = "train/chihuahua/original.png"

    assert frozenset(
        (original_path, "test/muffin/recompressed.jpg")
    ) in candidate_paths
    assert frozenset(
        (original_path, "train/chihuahua/resized.png")
    ) in candidate_paths
    assert frozenset(
        (original_path, "train/chihuahua/brighter.png")
    ) in candidate_paths
    assert all(
        "similar_but_different.png"
        not in {pair.left.relative_path, pair.right.relative_path}
        for pair in result.candidates
    )
    assert result.pair_comparison_count == 10
    assert sum(result.distance_histogram) == 10


def test_cross_partition_label_conflict_is_explicit_and_never_automatic(
    tmp_path: Path,
) -> None:
    _synthetic_dataset(tmp_path)

    _, _, result = _audit(tmp_path)
    pair = next(
        pair
        for pair in result.candidates
        if pair.comparison_scope is ComparisonScope.TRAIN_TEST
    )

    assert pair.distance <= 4
    assert pair.label_conflict is True
    assert pair.as_json()["automatic_duplicate_confirmation"] is False
    assert pair.as_json()["automatic_exclusion"] is False
    assert result.train_test_candidate_count >= 1
    assert result.label_conflict_candidate_count >= 1


def test_perceptual_human_decision_uses_increment_5_registry(tmp_path: Path) -> None:
    _synthetic_dataset(tmp_path)
    _, _, preliminary = _audit(tmp_path)
    pair = next(pair for pair in preliminary.candidates if pair.label_conflict)
    decision = HumanReviewDecision(
        review_id=pair.review_id,
        target_sample_ids=pair.target_sample_ids,
        occurrence_type=pair.occurrence_type,
        decision=ReviewDecisionState.UNCERTAIN,
        justification="A proximidade de pHash não basta para decidir.",
        reviewer="reviewer@example",
        decision_date="2026-11-01",
    )

    _, _, reviewed = _audit(tmp_path, decisions=(decision,))

    reviewed_pair = next(
        candidate
        for candidate in reviewed.candidates
        if candidate.review_id == decision.review_id
    )
    assert reviewed_pair.human_decision == decision
    assert reviewed.reviewed_candidate_count == 1


def test_mismatched_perceptual_human_decision_is_rejected(tmp_path: Path) -> None:
    _synthetic_dataset(tmp_path)
    _, _, preliminary = _audit(tmp_path)
    pair = preliminary.candidates[0]
    decision = HumanReviewDecision(
        review_id="perceptual-phash64:wrong",
        target_sample_ids=pair.target_sample_ids,
        occurrence_type=pair.occurrence_type,
        decision=ReviewDecisionState.UNCERTAIN,
        justification="Registro não corresponde ao candidato.",
        reviewer="reviewer@example",
        decision_date="2026-11-01",
    )

    with pytest.raises(PerceptualAuditError, match="does not match"):
        _audit(tmp_path, decisions=(decision,))


def test_report_is_deterministic_independent_of_inventory_order(
    tmp_path: Path,
) -> None:
    _synthetic_dataset(tmp_path)
    inventory = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)
    exact = analyze_exact_duplicates(inventory)
    reversed_inventory = replace(
        inventory, records=tuple(reversed(inventory.records))
    )

    first = analyze_perceptual_duplicates(
        tmp_path,
        inventory,
        exact,
        candidate_max_distance=4,
        block_size=2,
    )
    second = analyze_perceptual_duplicates(
        tmp_path,
        reversed_inventory,
        exact,
        candidate_max_distance=4,
        block_size=2,
    )

    assert first.payload() == second.payload()


def test_audit_preserves_every_source_byte(tmp_path: Path) -> None:
    paths = _synthetic_dataset(tmp_path)
    before = {path: path.read_bytes() for path in paths}

    _audit(tmp_path)

    assert {path: path.read_bytes() for path in paths} == before


def test_source_change_after_inventory_is_rejected(tmp_path: Path) -> None:
    paths = _synthetic_dataset(tmp_path)
    inventory = inventory_dataset(tmp_path, dataset_version_id=VERSION_ID)
    exact = analyze_exact_duplicates(inventory)
    Image.new("RGB", (8, 8), "black").save(paths[0])

    with pytest.raises(PerceptualAuditError, match="changed after inventory"):
        compute_perceptual_hashes(tmp_path, inventory, exact)


def test_feasibility_remains_pending_and_does_not_create_split(
    tmp_path: Path,
) -> None:
    _synthetic_dataset(tmp_path)
    for variant in (2, 3):
        muffin_path = tmp_path / f"train/muffin/independent-{variant}.png"
        muffin_path.parent.mkdir(parents=True, exist_ok=True)
        _base_image(variant=variant).save(muffin_path)
    _, exact, result = _audit(tmp_path)

    assessment = assess_perceptual_validation_feasibility(exact, result)
    payload = assessment.payload()

    assert assessment.assessment == (
        "quantitatively_feasible_pending_perceptual_review"
    )
    assert assessment.scenario_decision == "awaiting_human_decision"
    assert payload["scenario_selected"] is False
    assert payload["split_created"] is False


def test_local_artifacts_and_review_panel_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_dataset(source)
    _, exact, result = _audit(source)
    assessment = assess_perceptual_validation_feasibility(exact, result)
    destination = tmp_path / "audit"
    panel_path = destination / "review_panel.html"

    first_panel = render_review_panel(result, source, panel_path, max_pairs=4)
    second_panel = render_review_panel(result, source, panel_path, max_pairs=4)
    paths = write_perceptual_audit(
        result,
        assessment,
        (),
        source,
        destination,
        panel_pair_limit=4,
    )

    assert first_panel == second_panel
    assert paths[-1].read_bytes() == first_panel
    assert b"Imagens lado a lado" not in first_panel
    assert b"Exportar decis" in first_panel
    assert b"confirmed_duplicate" in first_panel
    assert json.loads(paths[0].read_text())["split_created"] is False
    assert json.loads(paths[1].read_text())["decisions"] == []
    assert json.loads(paths[2].read_text())["scenario_selected"] is False
    with pytest.raises(PerceptualAuditError, match="already exists"):
        write_perceptual_audit(result, assessment, (), source, destination)
