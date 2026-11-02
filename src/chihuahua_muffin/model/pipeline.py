"""Deterministic ``tf.data`` application of the SPEC 01 preprocessing reference."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from chihuahua_muffin.data.preprocessing import preprocess_image

from .contract import DatasetSample


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    dataset: tf.data.Dataset
    sample_ids: tuple[str, ...]
    sample_count: int
    shuffled: bool


def _reference_array(path_value: object) -> np.ndarray:
    """Invoke the single Pillow reference from a TensorFlow Python boundary."""

    if isinstance(path_value, np.ndarray):
        path_value = path_value.item()
    if isinstance(path_value, bytes):
        path = Path(os.fsdecode(path_value))
    else:
        path = Path(str(path_value))
    result = preprocess_image(path)
    return np.asarray(result.pixels, dtype=np.float32).reshape(result.shape)


def preprocess_path_tensor(path: tf.Tensor) -> tf.Tensor:
    image = tf.numpy_function(_reference_array, [path], Tout=tf.float32)
    image.set_shape((128, 128, 3))
    return image


def build_dataset(
    samples: tuple[DatasetSample, ...],
    *,
    batch_size: int,
    training: bool,
    seed: int,
    prefetch: bool = True,
) -> DatasetBundle:
    """Build a pipeline without changing IDs, labels, or split membership."""

    if not samples:
        raise ValueError("cannot build a dataset from an empty split")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    paths = [str(sample.path) for sample in samples]
    labels = [float(sample.class_id) for sample in samples]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(samples),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    options = tf.data.Options()
    options.deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        lambda path, label: (preprocess_path_tensor(path), label),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return DatasetBundle(
        dataset=dataset,
        sample_ids=tuple(sample.sample_id for sample in samples),
        sample_count=len(samples),
        shuffled=training,
    )


def balanced_smoke_subset(
    samples: tuple[DatasetSample, ...], *, samples_per_class: int
) -> tuple[DatasetSample, ...]:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    chosen: list[DatasetSample] = []
    for class_id in (0, 1):
        class_samples = [sample for sample in samples if sample.class_id == class_id]
        if len(class_samples) < samples_per_class:
            raise ValueError(f"not enough class {class_id} samples for smoke subset")
        chosen.extend(class_samples[:samples_per_class])
    return tuple(sorted(chosen, key=lambda sample: sample.sample_id))
