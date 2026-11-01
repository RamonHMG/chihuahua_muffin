from pathlib import Path

import pytest

from chihuahua_muffin.data.config import ConfigError, ValidationMode, load_config


PROJECT_ROOT = Path(__file__).parents[2]
EXAMPLE_CONFIG = PROJECT_ROOT / "config" / "example.toml"


def _modified_config(tmp_path: Path, old: str, new: str) -> Path:
    content = EXAMPLE_CONFIG.read_text(encoding="utf-8")
    assert content.count(old) == 1, f"test mutation is ambiguous: {old!r}"
    path = tmp_path / "config.toml"
    path.write_text(content.replace(old, new), encoding="utf-8", newline="\n")
    return path


def _assert_invalid(path: Path, field: str, *, mode: str = "structural") -> None:
    with pytest.raises(ConfigError) as captured:
        load_config(path, mode=mode)
    assert captured.value.field == field


def test_example_config_is_structurally_valid() -> None:
    config = load_config(EXAMPLE_CONFIG)

    assert config.schema_version == "1.0"
    assert config.seed == 42
    assert config.dataset.source == "kaggle"
    assert config.dataset.source_id == "2733586"
    assert config.dataset.version == "2"
    assert config.dataset.version_status == "confirmed"
    assert config.dataset.license == "CC0: Public Domain"
    assert config.dataset.license_status == "ambiguous"
    assert config.dataset.acquired_at == "2026-11-01T18:31:19.646234Z"
    assert config.dataset.acquisition_tool == "kagglehub"
    assert config.dataset.acquisition_tool_version == "1.0.3"
    assert config.dataset.acquisition_file_count == 5918
    assert config.dataset.human_authorization.status == "approved"
    assert config.dataset.human_authorization.source_id == "2733586"
    assert config.dataset.human_authorization.dataset_version == "2"
    assert config.dataset.human_authorization.local_use_only is True
    assert config.dataset.human_authorization.redistribution_allowed is False
    assert (config.split.train, config.split.validation, config.split.test) == (
        0.70,
        0.15,
        0.15,
    )
    assert (config.classes.chihuahua, config.classes.muffin) == (0, 1)
    assert config.preprocessing.padding_rgb == (255, 255, 255)
    assert config.preprocessing.padding_normalized == (1.0, 1.0, 1.0)


def test_default_path_is_the_example_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    assert load_config().seed == 42


def test_acquisition_rejects_missing_version(tmp_path: Path) -> None:
    path = _modified_config(tmp_path, 'version = "2"', 'version = ""')
    _assert_invalid(path, "dataset.version", mode="acquisition")


def test_acquisition_rejects_missing_license_after_version_is_set(
    tmp_path: Path,
) -> None:
    path = _modified_config(
        tmp_path, 'license = "CC0: Public Domain"', 'license = ""'
    )
    _assert_invalid(path, "dataset.license", mode="acquisition")


def test_acquisition_accepts_nonempty_version_and_license(tmp_path: Path) -> None:
    path = _modified_config(
        tmp_path,
        'license_status = "ambiguous"',
        'license_status = "confirmed"',
    )

    config = load_config(path, mode=ValidationMode.ACQUISITION)

    assert config.dataset.version == "2"
    assert config.dataset.license == "CC0: Public Domain"


def test_acquisition_accepts_recorded_local_educational_exception() -> None:
    config = load_config(EXAMPLE_CONFIG, mode="acquisition")

    assert config.dataset.license_status == "ambiguous"
    assert config.dataset.human_authorization.license_ambiguity_acknowledged is True


def test_acquisition_rejects_pending_license_even_with_local_exception(
    tmp_path: Path,
) -> None:
    path = _modified_config(
        tmp_path,
        'license_status = "ambiguous"',
        'license_status = "pending"',
    )
    _assert_invalid(path, "dataset.license_status", mode="acquisition")


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ('schema_version = "1.0"', 'schema_version = "2.0"', "schema_version"),
        ("seed = 42", 'seed = "42"', "project.seed"),
        ("seed = 42", "seed = 7", "project.seed"),
        ('version = "2"', "version = 2", "dataset.version"),
        (
            'version_status = "confirmed"',
            'version_status = "unknown"',
            "dataset.version_status",
        ),
        ('license = "CC0: Public Domain"', "license = true", "dataset.license"),
        (
            'license_status = "ambiguous"',
            'license_status = "unknown"',
            "dataset.license_status",
        ),
        (
            'acquired_at = "2026-11-01T18:31:19.646234Z"',
            'acquired_at = "2026-11-01"',
            "dataset.acquired_at",
        ),
        (
            'acquisition_tool = "kagglehub"',
            'acquisition_tool = "other"',
            "dataset.acquisition_tool",
        ),
        (
            "acquisition_file_count = 5918",
            "acquisition_file_count = 0",
            "dataset.acquisition_file_count",
        ),
        ("train = 0.70", 'train = "0.70"', "dataset.preferred_split.train"),
        ("train = 0.70", "train = 0.69", "dataset.preferred_split.train"),
        ("validation = 0.15", "validation = 0.14", "dataset.preferred_split.validation"),
        ("test = 0.15", "test = 0.16", "dataset.preferred_split.test"),
        ("stratified = true", "stratified = false", "dataset.preferred_split.stratified"),
        ("chihuahua = 0", "chihuahua = 1", "classes.chihuahua"),
        ("muffin = 1", "muffin = 0", "classes.muffin"),
        (
            '[classes]\nchihuahua = 0\nmuffin = 1\npositive_class = "muffin"',
            '[classes]\nchihuahua = 0\nmuffin = 1\npositive_class = "chihuahua"',
            "classes.positive_class",
        ),
        ("height = 128", "height = 224", "preprocessing.height"),
        ("channels = 3", 'channels = "3"', "preprocessing.channels"),
        ('color_mode = "RGB"', 'color_mode = "RGBA"', "preprocessing.color_mode"),
        ('interpolation = "bilinear"', 'interpolation = "nearest"', "preprocessing.interpolation"),
        ("padding_rgb = [255, 255, 255]", "padding_rgb = [0, 0, 0]", "preprocessing.padding_rgb"),
        (
            "padding_normalized = [1.0, 1.0, 1.0]",
            "padding_normalized = [1, 1, 1]",
            "preprocessing.padding_normalized",
        ),
        ("pixel_max = 1.0", "pixel_max = 255.0", "preprocessing.pixel_max"),
        ("fix_exif_orientation = true", "fix_exif_orientation = false", "preprocessing.fix_exif_orientation"),
        ("materialize_resized_images = false", "materialize_resized_images = true", "preprocessing.materialize_resized_images"),
        ("exact_hash = \"sha256\"", "exact_hash = \"md5\"", "duplicates.exact_hash"),
        ("perceptual_threshold = \"\"", "perceptual_threshold = \"8\"", "duplicates.perceptual_threshold"),
        ("automatic_removal_by_perceptual_distance = false", "automatic_removal_by_perceptual_distance = true", "duplicates.automatic_removal_by_perceptual_distance"),
        (
            'purpose = "educational_local_use"',
            'purpose = "commercial_use"',
            "dataset.human_authorization.purpose",
        ),
        (
            "license_ambiguity_acknowledged = true",
            "license_ambiguity_acknowledged = false",
            "dataset.human_authorization.license_ambiguity_acknowledged",
        ),
        (
            "redistribution_allowed = false",
            "redistribution_allowed = true",
            "dataset.human_authorization.redistribution_allowed",
        ),
        (
            '[dataset.human_authorization]\nstatus = "approved"\nsource_id = "2733586"',
            '[dataset.human_authorization]\nstatus = "approved"\nsource_id = "other-id"',
            "dataset.human_authorization.source_id",
        ),
        (
            'source_locator = "samuelcortinhas/muffin-vs-chihuahua-image-classification"',
            'source_locator = "other/dataset"',
            "dataset.human_authorization.dataset_handle",
        ),
    ],
)
def test_incompatible_values_are_rejected(
    tmp_path: Path, old: str, new: str, field: str
) -> None:
    _assert_invalid(_modified_config(tmp_path, old, new), field)


@pytest.mark.parametrize(
    ("line", "field"),
    [
        ('handle = "samuelcortinhas/muffin-vs-chihuahua-image-classification"\n', "dataset.handle"),
        ('version = "2"\n', "dataset.version"),
        ('license = "CC0: Public Domain"\n', "dataset.license"),
        ('acquired_at = "2026-11-01T18:31:19.646234Z"\n', "dataset.acquired_at"),
        ("validation = 0.15\n", "dataset.preferred_split.validation"),
        ("channels = 3\n", "preprocessing.channels"),
    ],
)
def test_required_fields_cannot_be_omitted(
    tmp_path: Path, line: str, field: str
) -> None:
    _assert_invalid(_modified_config(tmp_path, line, ""), field)


def test_raw_and_prepared_directories_must_differ(tmp_path: Path) -> None:
    path = _modified_config(
        tmp_path, 'prepared_dir = "data/prepared"', 'prepared_dir = "data/raw"'
    )
    _assert_invalid(path, "dataset.prepared_dir")


def test_malformed_toml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("[project\nseed = 42", encoding="utf-8")
    _assert_invalid(path, "config")


def test_unknown_validation_mode_is_rejected() -> None:
    _assert_invalid(EXAMPLE_CONFIG, "validation_mode", mode="training")
