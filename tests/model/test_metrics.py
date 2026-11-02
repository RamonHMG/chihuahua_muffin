import pytest

from chihuahua_muffin.model.metrics import classification_metrics


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
