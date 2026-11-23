"""Reproducible binary metrics and plots with explicit zero-division policy."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np

from chihuahua_muffin.data.preprocessing import preprocess_image

from .contract import DatasetSample


PANEL_CATEGORY_ORDER = (
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
)

_PANEL_CATEGORY_DETAILS = {
    "true_positive": ("VP", "muffin", "muffin", "#1b7f3a"),
    "true_negative": ("VN", "chihuahua", "chihuahua", "#1b7f3a"),
    "false_positive": ("FP", "chihuahua", "muffin", "#b42318"),
    "false_negative": ("FN", "muffin", "chihuahua", "#b42318"),
}


@dataclass(frozen=True, slots=True)
class ClassificationExample:
    category: str
    sample_id: str
    relative_path: str
    path: Path
    actual_class: str
    predicted_class: str
    muffin_probability: float

    def manifest_payload(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "relative_path": self.relative_path,
            "actual_class": self.actual_class,
            "predicted_class": self.predicted_class,
            "muffin_probability": self.muffin_probability,
        }


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


def select_classification_examples(
    samples: Sequence[DatasetSample],
    probabilities: Sequence[float],
    *,
    threshold: float,
    examples_per_category: int = 3,
) -> tuple[ClassificationExample, ...]:
    """Select an auditable sample-id ordered panel from the confusion matrix."""

    if len(samples) != len(probabilities) or not samples:
        raise ValueError("samples and probabilities must be nonempty and aligned")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    if examples_per_category <= 0:
        raise ValueError("examples_per_category must be positive")

    grouped: dict[str, list[ClassificationExample]] = {
        category: [] for category in PANEL_CATEGORY_ORDER
    }
    for sample, raw_probability in zip(samples, probabilities):
        probability = float(raw_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probabilities must be finite and within [0, 1]")
        predicted_id = 1 if probability >= threshold else 0
        if sample.class_id == 1 and predicted_id == 1:
            category = "true_positive"
        elif sample.class_id == 0 and predicted_id == 0:
            category = "true_negative"
        elif sample.class_id == 0 and predicted_id == 1:
            category = "false_positive"
        else:
            category = "false_negative"
        grouped[category].append(
            ClassificationExample(
                category=category,
                sample_id=sample.sample_id,
                relative_path=sample.relative_path,
                path=sample.path,
                actual_class=sample.class_name,
                predicted_class="muffin" if predicted_id == 1 else "chihuahua",
                muffin_probability=probability,
            )
        )

    selected: list[ClassificationExample] = []
    for category in PANEL_CATEGORY_ORDER:
        candidates = sorted(grouped[category], key=lambda example: example.sample_id)
        if len(candidates) < examples_per_category:
            raise ValueError(
                f"category {category} has {len(candidates)} examples; "
                f"{examples_per_category} required"
            )
        selected.extend(candidates[:examples_per_category])
    return tuple(selected)


def classification_panel_png(
    examples: Sequence[ClassificationExample], *, examples_per_category: int = 3
) -> bytes:
    """Render the exact preprocessed model inputs as a four-by-three PNG panel."""

    grouped = {
        category: [example for example in examples if example.category == category]
        for category in PANEL_CATEGORY_ORDER
    }
    if any(len(grouped[category]) != examples_per_category for category in grouped):
        raise ValueError("panel must contain the requested count for every category")

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "chihuahua-muffin-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(
        len(PANEL_CATEGORY_ORDER),
        examples_per_category,
        figsize=(4 * examples_per_category, 14),
        squeeze=False,
    )
    for row, category in enumerate(PANEL_CATEGORY_ORDER):
        abbreviation, expected_actual, expected_prediction, color = (
            _PANEL_CATEGORY_DETAILS[category]
        )
        for column, example in enumerate(grouped[category]):
            if (
                example.actual_class != expected_actual
                or example.predicted_class != expected_prediction
            ):
                raise ValueError(f"example is inconsistent with category {category}")
            processed = preprocess_image(example.path)
            pixels = np.asarray(processed.pixels, dtype=np.float32).reshape(
                processed.shape
            )
            axis = axes[row][column]
            axis.imshow(pixels)
            axis.set_title(
                f"{abbreviation} {column + 1}\n"
                f"real: {example.actual_class} | prevista: {example.predicted_class}\n"
                f"P(muffin): {example.muffin_probability:.4f}",
                fontsize=9,
                color=color,
            )
            axis.set_xlabel(
                f"{Path(example.relative_path).name}\nID: {example.sample_id[:12]}",
                fontsize=7,
            )
            axis.set_xticks(())
            axis.set_yticks(())
            for spine in axis.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2.5)
    figure.suptitle(
        "Painel de exemplos da avaliação — entradas 128 × 128 vistas pelo modelo",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    output = BytesIO()
    figure.savefig(output, format="png", dpi=140)
    plt.close(figure)
    return output.getvalue()


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
