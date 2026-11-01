"""Load and validate the approved SPEC 01 configuration contract."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
import tomllib
from typing import Any, TypeVar

from .records import (
    ClassConfig,
    DatasetConfig,
    DuplicatePolicyConfig,
    PreprocessingConfig,
    ProjectConfig,
    SplitConfig,
)


DEFAULT_CONFIG_PATH = Path("config/example.toml")


class ValidationMode(StrEnum):
    """Validation strictness for documentation versus data acquisition."""

    STRUCTURAL = "structural"
    ACQUISITION = "acquisition"


class ConfigError(ValueError):
    """Raised when a configuration field is missing or incompatible."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


T = TypeVar("T")


def _table(container: Mapping[str, Any], key: str, field: str) -> Mapping[str, Any]:
    if key not in container:
        raise ConfigError(field, "required table is missing")
    value = container[key]
    if type(value) is not dict:
        raise ConfigError(field, "must be a TOML table")
    return value


def _value(
    container: Mapping[str, Any],
    key: str,
    expected_type: type[T],
    field: str,
) -> T:
    if key not in container:
        raise ConfigError(field, "required field is missing")
    value = container[key]
    if type(value) is not expected_type:
        raise ConfigError(field, f"must have type {expected_type.__name__}")
    return value


def _triplet(
    container: Mapping[str, Any],
    key: str,
    item_type: type[T],
    field: str,
) -> tuple[T, T, T]:
    values = _value(container, key, list, field)
    if len(values) != 3 or any(type(item) is not item_type for item in values):
        raise ConfigError(field, f"must contain exactly three {item_type.__name__} values")
    return values[0], values[1], values[2]


def _nonempty(field: str, value: str) -> None:
    if not value.strip():
        raise ConfigError(field, "must not be empty")


def _exact(field: str, value: object, expected: object) -> None:
    if value != expected:
        raise ConfigError(field, f"must be {expected!r}; received {value!r}")


def _parse_mode(mode: ValidationMode | str) -> ValidationMode:
    try:
        return ValidationMode(mode)
    except ValueError as error:
        choices = ", ".join(item.value for item in ValidationMode)
        raise ConfigError("validation_mode", f"must be one of: {choices}") from error


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    mode: ValidationMode | str = ValidationMode.STRUCTURAL,
) -> ProjectConfig:
    """Load a TOML file and validate the approved SPEC 01 invariants.

    Structural validation accepts empty source version and license values so the
    repository's example remains usable by tests and tooling. Acquisition mode
    additionally requires those two values to be non-empty.
    """

    config_path = Path(path)
    validation_mode = _parse_mode(mode)
    try:
        with config_path.open("rb") as stream:
            document = tomllib.load(stream)
    except FileNotFoundError as error:
        raise ConfigError("config", f"file does not exist: {config_path}") from error
    except OSError as error:
        raise ConfigError("config", f"cannot read {config_path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError("config", f"invalid TOML in {config_path}: {error}") from error

    schema_version = _value(document, "schema_version", str, "schema_version")
    _exact("schema_version", schema_version, "1.0")

    project_table = _table(document, "project", "project")
    seed = _value(project_table, "seed", int, "project.seed")
    _exact("project.seed", seed, 42)

    dataset_table = _table(document, "dataset", "dataset")
    handle = _value(dataset_table, "handle", str, "dataset.handle")
    version = _value(dataset_table, "version", str, "dataset.version")
    license_name = _value(dataset_table, "license", str, "dataset.license")
    source_url = _value(dataset_table, "source_url", str, "dataset.source_url")
    raw_dir_text = _value(dataset_table, "raw_dir", str, "dataset.raw_dir")
    prepared_dir_text = _value(
        dataset_table, "prepared_dir", str, "dataset.prepared_dir"
    )
    preserve_splits = _value(
        dataset_table,
        "preserve_valid_official_splits",
        bool,
        "dataset.preserve_valid_official_splits",
    )
    for field, value in (
        ("dataset.handle", handle),
        ("dataset.source_url", source_url),
        ("dataset.raw_dir", raw_dir_text),
        ("dataset.prepared_dir", prepared_dir_text),
    ):
        _nonempty(field, value)
    _exact("dataset.preserve_valid_official_splits", preserve_splits, True)
    raw_dir = Path(raw_dir_text)
    prepared_dir = Path(prepared_dir_text)
    if raw_dir == prepared_dir:
        raise ConfigError("dataset.prepared_dir", "must differ from dataset.raw_dir")

    if validation_mode is ValidationMode.ACQUISITION:
        _nonempty("dataset.version", version)
        _nonempty("dataset.license", license_name)

    split_table = _table(dataset_table, "preferred_split", "dataset.preferred_split")
    train = _value(split_table, "train", float, "dataset.preferred_split.train")
    validation = _value(
        split_table, "validation", float, "dataset.preferred_split.validation"
    )
    test = _value(split_table, "test", float, "dataset.preferred_split.test")
    stratified = _value(
        split_table, "stratified", bool, "dataset.preferred_split.stratified"
    )
    group_aware = _value(
        split_table, "group_aware", bool, "dataset.preferred_split.group_aware"
    )
    for field, value, expected in (
        ("dataset.preferred_split.train", train, 0.70),
        ("dataset.preferred_split.validation", validation, 0.15),
        ("dataset.preferred_split.test", test, 0.15),
        ("dataset.preferred_split.stratified", stratified, True),
        ("dataset.preferred_split.group_aware", group_aware, True),
    ):
        _exact(field, value, expected)
    if abs((train + validation + test) - 1.0) > 1e-12:
        raise ConfigError("dataset.preferred_split", "proportions must sum to 1.0")

    classes_table = _table(document, "classes", "classes")
    chihuahua = _value(classes_table, "chihuahua", int, "classes.chihuahua")
    muffin = _value(classes_table, "muffin", int, "classes.muffin")
    positive_class = _value(
        classes_table, "positive_class", str, "classes.positive_class"
    )
    _exact("classes.chihuahua", chihuahua, 0)
    _exact("classes.muffin", muffin, 1)
    _exact("classes.positive_class", positive_class, "muffin")

    preprocessing_table = _table(document, "preprocessing", "preprocessing")
    preprocessing_values: dict[str, object] = {
        "height": _value(preprocessing_table, "height", int, "preprocessing.height"),
        "width": _value(preprocessing_table, "width", int, "preprocessing.width"),
        "channels": _value(
            preprocessing_table, "channels", int, "preprocessing.channels"
        ),
        "color_mode": _value(
            preprocessing_table, "color_mode", str, "preprocessing.color_mode"
        ),
        "resize": _value(preprocessing_table, "resize", str, "preprocessing.resize"),
        "interpolation": _value(
            preprocessing_table,
            "interpolation",
            str,
            "preprocessing.interpolation",
        ),
        "padding_color": _value(
            preprocessing_table,
            "padding_color",
            str,
            "preprocessing.padding_color",
        ),
        "padding_rgb": _triplet(
            preprocessing_table, "padding_rgb", int, "preprocessing.padding_rgb"
        ),
        "padding_normalized": _triplet(
            preprocessing_table,
            "padding_normalized",
            float,
            "preprocessing.padding_normalized",
        ),
        "pixel_min": _value(
            preprocessing_table, "pixel_min", float, "preprocessing.pixel_min"
        ),
        "pixel_max": _value(
            preprocessing_table, "pixel_max", float, "preprocessing.pixel_max"
        ),
        "fix_exif_orientation": _value(
            preprocessing_table,
            "fix_exif_orientation",
            bool,
            "preprocessing.fix_exif_orientation",
        ),
        "rgba_background": _value(
            preprocessing_table,
            "rgba_background",
            str,
            "preprocessing.rgba_background",
        ),
        "grayscale_to_rgb": _value(
            preprocessing_table,
            "grayscale_to_rgb",
            str,
            "preprocessing.grayscale_to_rgb",
        ),
        "exclude_animated": _value(
            preprocessing_table,
            "exclude_animated",
            bool,
            "preprocessing.exclude_animated",
        ),
        "materialize_resized_images": _value(
            preprocessing_table,
            "materialize_resized_images",
            bool,
            "preprocessing.materialize_resized_images",
        ),
    }
    approved_preprocessing: dict[str, object] = {
        "height": 128,
        "width": 128,
        "channels": 3,
        "color_mode": "RGB",
        "resize": "pad",
        "interpolation": "bilinear",
        "padding_color": "white",
        "padding_rgb": (255, 255, 255),
        "padding_normalized": (1.0, 1.0, 1.0),
        "pixel_min": 0.0,
        "pixel_max": 1.0,
        "fix_exif_orientation": True,
        "rgba_background": "white",
        "grayscale_to_rgb": "replicate",
        "exclude_animated": True,
        "materialize_resized_images": False,
    }
    for key, expected in approved_preprocessing.items():
        _exact(f"preprocessing.{key}", preprocessing_values[key], expected)

    duplicates_table = _table(document, "duplicates", "duplicates")
    duplicate_values: dict[str, object] = {
        "exact_hash": _value(
            duplicates_table, "exact_hash", str, "duplicates.exact_hash"
        ),
        "perceptual_method": _value(
            duplicates_table,
            "perceptual_method",
            str,
            "duplicates.perceptual_method",
        ),
        "perceptual_method_status": _value(
            duplicates_table,
            "perceptual_method_status",
            str,
            "duplicates.perceptual_method_status",
        ),
        "perceptual_candidate_hash_size": _value(
            duplicates_table,
            "perceptual_candidate_hash_size",
            int,
            "duplicates.perceptual_candidate_hash_size",
        ),
        "perceptual_candidate_highfreq_factor": _value(
            duplicates_table,
            "perceptual_candidate_highfreq_factor",
            int,
            "duplicates.perceptual_candidate_highfreq_factor",
        ),
        "perceptual_threshold": _value(
            duplicates_table,
            "perceptual_threshold",
            str,
            "duplicates.perceptual_threshold",
        ),
        "automatic_removal_by_perceptual_distance": _value(
            duplicates_table,
            "automatic_removal_by_perceptual_distance",
            bool,
            "duplicates.automatic_removal_by_perceptual_distance",
        ),
        "require_human_review_for_conflicting_labels": _value(
            duplicates_table,
            "require_human_review_for_conflicting_labels",
            bool,
            "duplicates.require_human_review_for_conflicting_labels",
        ),
    }
    pending_duplicate_policy: dict[str, object] = {
        "exact_hash": "sha256",
        "perceptual_method": "phash",
        "perceptual_method_status": "proposed_pending_audit",
        "perceptual_candidate_hash_size": 8,
        "perceptual_candidate_highfreq_factor": 4,
        "perceptual_threshold": "",
        "automatic_removal_by_perceptual_distance": False,
        "require_human_review_for_conflicting_labels": True,
    }
    for key, expected in pending_duplicate_policy.items():
        _exact(f"duplicates.{key}", duplicate_values[key], expected)

    return ProjectConfig(
        schema_version=schema_version,
        seed=seed,
        dataset=DatasetConfig(
            handle=handle,
            version=version,
            license=license_name,
            source_url=source_url,
            raw_dir=raw_dir,
            prepared_dir=prepared_dir,
            preserve_valid_official_splits=preserve_splits,
        ),
        split=SplitConfig(
            train=train,
            validation=validation,
            test=test,
            stratified=stratified,
            group_aware=group_aware,
        ),
        classes=ClassConfig(
            chihuahua=chihuahua,
            muffin=muffin,
            positive_class=positive_class,
        ),
        preprocessing=PreprocessingConfig(**preprocessing_values),
        duplicates=DuplicatePolicyConfig(**duplicate_values),
    )
