from pathlib import Path

from PIL import Image
import pytest

from chihuahua_muffin.data.preprocessing import (
    PreprocessingError,
    preprocess_image,
)


def _save(path: Path, image: Image.Image, *, image_format: str | None = None) -> Path:
    image.save(path, format=image_format)
    return path


def test_rgb_resize_preserves_aspect_ratio_and_uses_normalized_white_padding(
    tmp_path: Path,
) -> None:
    path = _save(tmp_path / "wide.png", Image.new("RGB", (4, 2), (255, 0, 0)))

    result = preprocess_image(path)

    assert result.shape == (128, 128, 3)
    assert result.source_mode == "RGB"
    assert result.color_conversion == "preserve_rgb"
    assert result.pixel(0, 64) == (1.0, 1.0, 1.0)
    assert result.pixel(64, 64) == (1.0, 0.0, 0.0)
    assert min(result.pixels) == 0.0
    assert max(result.pixels) == 1.0


def test_grayscale_is_replicated_to_three_channels(tmp_path: Path) -> None:
    path = _save(tmp_path / "gray.png", Image.new("L", (2, 2), 64))

    result = preprocess_image(path)

    expected = 64 / 255.0
    assert result.color_conversion == "grayscale_replicate"
    assert result.pixel(50, 50) == (expected, expected, expected)


def test_rgba_is_composited_over_white(tmp_path: Path) -> None:
    path = _save(
        tmp_path / "alpha.png",
        Image.new("RGBA", (2, 2), (255, 0, 0, 128)),
    )

    result = preprocess_image(path)

    red, green, blue = result.pixel(50, 50)
    assert result.color_conversion == "rgba_on_white"
    assert red == 1.0
    assert green == blue
    assert green == 127 / 255.0


def test_exif_orientation_is_applied_before_resize(tmp_path: Path) -> None:
    path = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (4, 2), (220, 10, 10))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif, quality=100, subsampling=0)

    result = preprocess_image(path)

    assert result.exif_orientation == 6
    assert result.exif_orientation_applied is True
    assert result.pixel(64, 0) == (1.0, 1.0, 1.0)
    assert result.pixel(64, 64)[0] > 0.8


def test_resize_uses_bilinear_interpolation(tmp_path: Path) -> None:
    image = Image.new("RGB", (2, 2))
    image.putdata([(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
    path = _save(tmp_path / "corners.png", image)

    result = preprocess_image(path)
    center = result.pixel(64, 64)

    assert all(0.0 < channel < 1.0 for channel in center)


def test_preprocessing_preserves_original_file(tmp_path: Path) -> None:
    path = _save(tmp_path / "source.png", Image.new("RGBA", (5, 3), (1, 2, 3, 4)))
    before = path.read_bytes()

    preprocess_image(path)

    assert path.read_bytes() == before


def test_invalid_image_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jpg"
    path.write_bytes(b"invalid")

    with pytest.raises(PreprocessingError, match="cannot decode"):
        preprocess_image(path)


def test_animation_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "animated.gif"
    frames = [Image.new("RGB", (2, 2), color) for color in ("red", "blue")]
    frames[0].save(path, save_all=True, append_images=frames[1:])

    with pytest.raises(PreprocessingError, match="animated"):
        preprocess_image(path)


def test_unsupported_palette_mode_requires_review_not_implicit_conversion(
    tmp_path: Path,
) -> None:
    path = _save(tmp_path / "palette.png", Image.new("P", (2, 2)))

    with pytest.raises(PreprocessingError, match="unsupported color mode"):
        preprocess_image(path)


def test_pixel_coordinates_are_checked(tmp_path: Path) -> None:
    path = _save(tmp_path / "image.png", Image.new("RGB", (2, 2)))
    result = preprocess_image(path)

    with pytest.raises(IndexError):
        result.pixel(128, 0)
