"""Typed, immutable records used by the initial SPEC 01 configuration layer."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Dataset source metadata and local storage locations."""

    source: str
    source_id: str
    handle: str
    owner: str
    version: str
    version_status: str
    version_note: str
    license: str
    license_status: str
    license_note: str
    source_url: str
    metadata_url: str
    metadata_checked_at: str
    raw_dir: Path
    prepared_dir: Path
    preserve_valid_official_splits: bool


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Approved preferred split policy."""

    train: float
    validation: float
    test: float
    stratified: bool
    group_aware: bool


@dataclass(frozen=True, slots=True)
class ClassConfig:
    """Immutable binary class mapping shared with SPEC 02."""

    chihuahua: int
    muffin: int
    positive_class: str


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Deterministic image-input contract owned by SPEC 01."""

    height: int
    width: int
    channels: int
    color_mode: str
    resize: str
    interpolation: str
    padding_color: str
    padding_rgb: tuple[int, int, int]
    padding_normalized: tuple[float, float, float]
    pixel_min: float
    pixel_max: float
    fix_exif_orientation: bool
    rgba_background: str
    grayscale_to_rgb: str
    exclude_animated: bool
    materialize_resized_images: bool


@dataclass(frozen=True, slots=True)
class DuplicatePolicyConfig:
    """Approved exact policy and explicitly pending perceptual policy."""

    exact_hash: str
    perceptual_method: str
    perceptual_method_status: str
    perceptual_candidate_hash_size: int
    perceptual_candidate_highfreq_factor: int
    perceptual_threshold: str
    automatic_removal_by_perceptual_distance: bool
    require_human_review_for_conflicting_labels: bool


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Validated subset of the project configuration needed by SPEC 01."""

    schema_version: str
    seed: int
    dataset: DatasetConfig
    split: SplitConfig
    classes: ClassConfig
    preprocessing: PreprocessingConfig
    duplicates: DuplicatePolicyConfig
