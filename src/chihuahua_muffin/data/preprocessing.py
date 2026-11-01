"""Pillow reference implementation of the SPEC 01 preprocessing contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


TARGET_HEIGHT = 128
TARGET_WIDTH = 128
TARGET_CHANNELS = 3
WHITE_RGB = (255, 255, 255)
GRAYSCALE_MODES = {"1", "L"}


class PreprocessingError(ValueError):
    """Raised when an image cannot satisfy the approved input contract."""


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    """Normalized row-major HWC pixels produced by the reference pipeline."""

    height: int
    width: int
    channels: int
    pixels: tuple[float, ...]
    source_mode: str
    color_conversion: str
    exif_orientation: int | None
    exif_orientation_applied: bool

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.height, self.width, self.channels

    def pixel(self, row: int, column: int) -> tuple[float, float, float]:
        """Return one RGB pixel from the normalized HWC representation."""

        if not 0 <= row < self.height or not 0 <= column < self.width:
            raise IndexError("pixel coordinates are outside the image")
        offset = (row * self.width + column) * self.channels
        return (
            self.pixels[offset],
            self.pixels[offset + 1],
            self.pixels[offset + 2],
        )


def preprocess_image(path: str | Path) -> PreprocessedImage:
    """Apply EXIF, RGB conversion, bilinear resize, white pad and normalization.

    The source file is opened read-only. This reference is owned and tested by
    SPEC 01; SPEC 02 must reproduce the same behavior in its ``tf.data`` path.
    """

    source_path = Path(path)
    try:
        with Image.open(source_path) as opened:
            source_mode = opened.mode
            exif_orientation = opened.getexif().get(274)
            if type(exif_orientation) is not int or exif_orientation not in range(1, 9):
                exif_orientation = None
            if bool(
                getattr(opened, "is_animated", False)
                or getattr(opened, "n_frames", 1) > 1
            ):
                raise PreprocessingError("animated images are excluded")
            opened.load()
            oriented = ImageOps.exif_transpose(opened)
            rgb, conversion = _to_contract_rgb(oriented)
    except PreprocessingError:
        raise
    except (OSError, SyntaxError, UnidentifiedImageError, ValueError) as error:
        raise PreprocessingError(f"cannot decode image: {source_path}") from error

    resized = ImageOps.pad(
        rgb,
        (TARGET_WIDTH, TARGET_HEIGHT),
        method=Image.Resampling.BILINEAR,
        color=WHITE_RGB,
        centering=(0.5, 0.5),
    )
    pixel_data = resized.get_flattened_data()
    pixels = tuple(channel / 255.0 for pixel in pixel_data for channel in pixel)
    expected_values = TARGET_HEIGHT * TARGET_WIDTH * TARGET_CHANNELS
    if len(pixels) != expected_values:
        raise PreprocessingError(
            f"preprocessing produced {len(pixels)} values; expected {expected_values}"
        )
    if any(value < 0.0 or value > 1.0 for value in pixels):
        raise PreprocessingError("normalized pixels are outside [0, 1]")
    return PreprocessedImage(
        height=TARGET_HEIGHT,
        width=TARGET_WIDTH,
        channels=TARGET_CHANNELS,
        pixels=pixels,
        source_mode=source_mode,
        color_conversion=conversion,
        exif_orientation=exif_orientation,
        exif_orientation_applied=(
            exif_orientation is not None and exif_orientation != 1
        ),
    )


def _to_contract_rgb(image: Image.Image) -> tuple[Image.Image, str]:
    if image.mode == "RGB":
        return image.copy(), "preserve_rgb"
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (*WHITE_RGB, 255))
        return Image.alpha_composite(background, image).convert("RGB"), "rgba_on_white"
    if image.mode in GRAYSCALE_MODES:
        return image.convert("RGB"), "grayscale_replicate"
    raise PreprocessingError(f"unsupported color mode: {image.mode}")
