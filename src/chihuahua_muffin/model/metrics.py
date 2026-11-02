"""Reproducible binary metrics and plots with explicit zero-division policy."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Sequence


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(
    labels: Sequence[int], probabilities: Sequence[float], *, threshold: float
) -> dict[str, object]:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be nonempty and aligned")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    if any(label not in {0, 1} for label in labels):
        raise ValueError("labels must be binary")
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    precision_muffin = _divide(tp, tp + fp)
    recall_muffin = _divide(tp, tp + fn)
    f1_muffin = _divide(2 * precision_muffin * recall_muffin, precision_muffin + recall_muffin)
    precision_chihuahua = _divide(tn, tn + fn)
    recall_chihuahua = _divide(tn, tn + fp)
    f1_chihuahua = _divide(
        2 * precision_chihuahua * recall_chihuahua,
        precision_chihuahua + recall_chihuahua,
    )
    return {
        "threshold": threshold,
        "sample_count": len(labels),
        "accuracy": _divide(tn + tp, len(labels)),
        "precision": precision_muffin,
        "recall": recall_muffin,
        "f1": f1_muffin,
        "macro_f1": (f1_chihuahua + f1_muffin) / 2,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "support": {"chihuahua": tn + fp, "muffin": fn + tp},
        "per_class": {
            "chihuahua": {
                "precision": precision_chihuahua,
                "recall": recall_chihuahua,
                "f1": f1_chihuahua,
            },
            "muffin": {
                "precision": precision_muffin,
                "recall": recall_muffin,
                "f1": f1_muffin,
            },
        },
        "zero_division": 0.0,
    }


def save_training_curves(history: dict[str, list[float]], path: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "chihuahua-muffin-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    epochs = list(range(1, len(history.get("loss", [])) + 1))
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, metric in zip(axes.flat, ("loss", "accuracy", "precision", "recall")):
        axis.plot(epochs, history.get(metric, []), label="train")
        axis.plot(epochs, history.get(f"val_{metric}", []), label="validation")
        axis.set_title(metric)
        axis.set_xlabel("epoch")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def save_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "chihuahua-muffin-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks((0, 1), labels=("chihuahua", "muffin"))
    axis.set_yticks((0, 1), labels=("chihuahua", "muffin"))
    axis.set_xlabel("predicted")
    axis.set_ylabel("actual")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
