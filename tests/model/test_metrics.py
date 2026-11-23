from pathlib import Path

from PIL import Image
import pytest

from chihuahua_muffin.model.contract import DatasetSample
from chihuahua_muffin.model.metrics import (
    PANEL_CATEGORY_ORDER,
    classification_metrics,
    classification_panel_png,
    select_classification_examples,
)


def _sample(tmp_path: Path, sample_id: str, class_id: int) -> DatasetSample:
    path = tmp_path / f"{sample_id}.png"
    Image.new("RGB", (8, 6), color=(20 + class_id * 200, 40, 60)).save(path)
    return DatasetSample(
        sample_id=sample_id,
        relative_path=f"test/{'muffin' if class_id else 'chihuahua'}/{path.name}",
        path=path,
        class_name="muffin" if class_id else "chihuahua",
        class_id=class_id,
        source_partition="test",
        split="test",
        sha256="0" * 64,
    )


def test_binary_metrics_and_confusion_matrix_use_muffin_as_positive() -> None:
    result = classification_metrics(
        [0, 0, 1, 1], [0.1, 0.8, 0.7, 0.2], threshold=0.5
    )

    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["macro_f1"] == 0.5


def test_zero_division_is_explicit_for_always_chihuahua_baseline() -> None:
    result = classification_metrics([0, 0, 1], [0.1, 0.2, 0.3], threshold=0.5)

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["zero_division"] == 0.0
    assert result["macro_f1"] == pytest.approx(0.4)


def test_panel_selection_has_three_deterministic_examples_per_category(
    tmp_path: Path,
) -> None:
    specifications = (
        (1, 0.9),
        (0, 0.1),
        (0, 0.9),
        (1, 0.1),
    )
    samples = []
    probabilities = []
    for group, (class_id, probability) in enumerate(specifications):
        for suffix in ("c", "a", "b"):
            samples.append(_sample(tmp_path, f"{group}-{suffix}", class_id))
            probabilities.append(probability)

    examples = select_classification_examples(
        samples, probabilities, threshold=0.5, examples_per_category=3
    )

    assert [example.category for example in examples] == [
        category for category in PANEL_CATEGORY_ORDER for _ in range(3)
    ]
    for offset in range(0, 12, 3):
        assert [example.sample_id for example in examples[offset : offset + 3]] == [
            f"{offset // 3}-a",
            f"{offset // 3}-b",
            f"{offset // 3}-c",
        ]


def test_panel_render_uses_preprocessed_inputs(tmp_path: Path) -> None:
    samples = [
        _sample(tmp_path, "tp", 1),
        _sample(tmp_path, "tn", 0),
        _sample(tmp_path, "fp", 0),
        _sample(tmp_path, "fn", 1),
    ]
    examples = select_classification_examples(
        samples, [0.9, 0.1, 0.9, 0.1], threshold=0.5, examples_per_category=1
    )

    payload = classification_panel_png(examples, examples_per_category=1)

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")


def test_panel_selection_rejects_category_with_too_few_examples(
    tmp_path: Path,
) -> None:
    samples = [_sample(tmp_path, "only-tp", 1)]

    with pytest.raises(ValueError, match="true_positive has 1 examples; 3 required"):
        select_classification_examples(
            samples, [0.9], threshold=0.5, examples_per_category=3
        )
