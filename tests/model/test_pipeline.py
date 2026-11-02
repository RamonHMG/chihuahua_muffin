from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from chihuahua_muffin.data.preprocessing import preprocess_image
from chihuahua_muffin.model.contract import DatasetSample
from chihuahua_muffin.model.pipeline import (
    balanced_smoke_subset,
    build_dataset,
    preprocess_path_tensor,
)


def _sample(path: Path, index: int, class_id: int, split: str = "train") -> DatasetSample:
    return DatasetSample(
        sample_id=f"{index:064x}",
        relative_path=path.name,
        path=path,
        class_name="chihuahua" if class_id == 0 else "muffin",
        class_id=class_id,
        source_partition="train",
        split=split,
        sha256="0" * 64,
    )


def test_tf_preprocessing_is_exactly_the_shared_reference_for_supported_modes(
    tmp_path: Path,
) -> None:
    paths = []
    rgb = tmp_path / "rgb.png"
    Image.new("RGB", (8, 4), (255, 10, 20)).save(rgb)
    paths.append(rgb)
    gray = tmp_path / "gray.png"
    Image.new("L", (3, 9), 64).save(gray)
    paths.append(gray)
    rgba = tmp_path / "rgba.png"
    Image.new("RGBA", (4, 2), (255, 0, 0, 128)).save(rgba)
    paths.append(rgba)
    oriented = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (6, 2), (200, 20, 10))
    exif = image.getexif()
    exif[274] = 6
    image.save(oriented, exif=exif, quality=100, subsampling=0)
    paths.append(oriented)

    for path in paths:
        reference = preprocess_image(path)
        expected = np.asarray(reference.pixels, dtype=np.float32).reshape(reference.shape)
        actual = preprocess_path_tensor(tf.constant(str(path))).numpy()
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (128, 128, 3)
        assert 0.0 <= float(actual.min()) <= float(actual.max()) <= 1.0


def test_dataset_shuffles_only_when_requested_and_preserves_labels(tmp_path: Path) -> None:
    samples = []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (4, 4), (index * 20, 0, 0)).save(path)
        samples.append(_sample(path, index, index % 2))

    validation = build_dataset(tuple(samples), batch_size=4, training=False, seed=42, prefetch=False)
    training = build_dataset(tuple(samples), batch_size=4, training=True, seed=42, prefetch=False)
    _, labels = next(iter(validation.dataset))

    assert validation.shuffled is False
    assert training.shuffled is True
    assert labels.numpy().tolist() == [0.0, 1.0, 0.0, 1.0]
    assert validation.sample_ids == tuple(sample.sample_id for sample in samples)


def test_smoke_subset_is_balanced_without_changing_membership(tmp_path: Path) -> None:
    samples = []
    for index in range(8):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (2, 2)).save(path)
        samples.append(_sample(path, index, 0 if index < 5 else 1))

    subset = balanced_smoke_subset(tuple(samples), samples_per_class=2)

    assert [sample.class_id for sample in subset].count(0) == 2
    assert [sample.class_id for sample in subset].count(1) == 2
    assert {sample.sample_id for sample in subset} <= {sample.sample_id for sample in samples}
