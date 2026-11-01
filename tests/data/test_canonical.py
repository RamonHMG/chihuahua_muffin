import csv
from decimal import Decimal
from io import StringIO
from pathlib import PureWindowsPath

import pytest

from chihuahua_muffin.data.canonical import (
    INDEX_CSV_HEADER,
    CanonicalizationError,
    IdentifierError,
    canonical_index_csv_bytes,
    canonical_index_csv_sha256,
    canonical_json_bytes,
    canonical_json_sha256,
    make_sample_id,
    manifest_sha256,
    normalize_relative_path,
    sha256_hex,
)


def _index_row(sample_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(INDEX_CSV_HEADER, "")
    row["sample_id"] = sample_id
    row.update(overrides)
    return row


def test_json_is_independent_of_dictionary_order() -> None:
    first = {"z": 3, "nested": {"b": 2, "a": 1}, "a": [True, None]}
    second = {"a": [True, None], "nested": {"a": 1, "b": 2}, "z": 3}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_sha256(first) == canonical_json_sha256(second)


def test_json_golden_bytes_and_digest_include_utf8_without_trailing_newline() -> None:
    value = {"nested": {"x": -1.5}, "a": [True, None, "café"], "n": 2}
    expected = '{"a":[true,null,"café"],"n":2,"nested":{"x":-1.5}}'.encode()

    assert canonical_json_bytes(value) == expected
    assert not expected.endswith(b"\n")
    assert canonical_json_sha256(value) == (
        "36ceb8e9c24b9ee2b7c2ae4c698548d6cdef2850b2bc937e4d2f60b3f00a0443"
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_json_rejects_non_finite_floats(invalid: float) -> None:
    with pytest.raises(CanonicalizationError, match="finite float"):
        canonical_json_bytes({"invalid": invalid})


@pytest.mark.parametrize(
    "invalid",
    [
        (1, 2),
        {1, 2},
        b"bytes",
        Decimal("1.0"),
        {1: "non-string-key"},
    ],
)
def test_json_rejects_unsupported_types(invalid: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json_bytes(invalid)  # type: ignore[arg-type]


def test_json_rejects_cycles() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)

    with pytest.raises(CanonicalizationError, match="cycle"):
        canonical_json_bytes(cyclic)  # type: ignore[arg-type]


def test_json_digest_changes_when_serialized_data_changes() -> None:
    original = {"schema_version": "1.0", "count": 2}
    changed = {"schema_version": "1.0", "count": 3}

    assert canonical_json_bytes(original) != canonical_json_bytes(changed)
    assert canonical_json_sha256(original) != canonical_json_sha256(changed)


def test_manifest_digest_excludes_only_its_top_level_self_digest() -> None:
    manifest = {
        "schema_version": "1.0",
        "manifest_digest": "old-value",
        "nested": {"manifest_digest": "part-of-the-payload"},
    }
    original = dict(manifest)
    first = manifest_sha256(manifest, digest_field="manifest_digest")
    assert manifest == original

    manifest["manifest_digest"] = "another-value"

    assert manifest_sha256(manifest, digest_field="manifest_digest") == first
    assert manifest_sha256(
        {
            **manifest,
            "nested": {"manifest_digest": "changed-payload"},
        },
        digest_field="manifest_digest",
    ) != first
    assert manifest["manifest_digest"] == "another-value"


def test_manifest_digest_does_not_require_field_to_be_present() -> None:
    manifest = {"schema_version": "1.0", "count": 2}

    assert manifest_sha256(manifest, digest_field="manifest_digest") == (
        canonical_json_sha256(manifest)
    )


@pytest.mark.parametrize("digest_field", ["", 1, None])
def test_manifest_digest_rejects_invalid_field_name(digest_field: object) -> None:
    with pytest.raises(CanonicalizationError):
        manifest_sha256(  # type: ignore[arg-type]
            {"schema_version": "1.0"}, digest_field=digest_field
        )


def test_sha256_requires_exact_bytes() -> None:
    with pytest.raises(CanonicalizationError, match="bytes"):
        sha256_hex(bytearray(b"content"))  # type: ignore[arg-type]


def test_index_header_is_fixed_and_documented() -> None:
    assert INDEX_CSV_HEADER == (
        "sample_id",
        "relative_path",
        "class_name",
        "class_id",
        "source_partition",
        "split",
        "status",
        "reason",
        "sha256",
        "duplicate_group_id",
        "perceptual_method",
        "perceptual_evidence",
        "format",
        "width",
        "height",
        "channels",
        "is_animated",
        "exif_orientation_applied",
        "color_conversion",
    )


def test_index_csv_golden_bytes_rows_and_digest() -> None:
    rows = [
        _index_row(
            "b",
            relative_path="muffin/b.jpg",
            class_name="muffin",
            class_id=1,
            status="accepted",
            width=128,
            height=128,
            channels=3,
            is_animated=False,
        ),
        _index_row(
            "a",
            relative_path="chihuahua/a.jpg",
            class_name="chihuahua",
            class_id=0,
            status="accepted",
            width=128,
            height=128,
            channels=3,
            is_animated=False,
        ),
    ]
    expected = (
        b"sample_id,relative_path,class_name,class_id,source_partition,split,"
        b"status,reason,sha256,duplicate_group_id,perceptual_method,"
        b"perceptual_evidence,format,width,height,channels,is_animated,"
        b"exif_orientation_applied,color_conversion\n"
        b"a,chihuahua/a.jpg,chihuahua,0,,,accepted,,,,,,,128,128,3,false,,\n"
        b"b,muffin/b.jpg,muffin,1,,,accepted,,,,,,,128,128,3,false,,\n"
    )

    assert canonical_index_csv_bytes(rows) == expected
    assert canonical_index_csv_sha256(rows) == (
        "1f422668ae9ececfafaa0dc3d9c0e830c10c2a019d2ce1b4297ba5e5d1299a36"
    )


def test_index_csv_is_independent_of_row_and_mapping_order() -> None:
    first_row = _index_row("a", relative_path="a.jpg", class_id=0)
    second_row = _index_row("b", relative_path="b.jpg", class_id=1)
    reversed_first = dict(reversed(tuple(first_row.items())))
    reversed_second = dict(reversed(tuple(second_row.items())))

    assert canonical_index_csv_bytes([second_row, first_row]) == (
        canonical_index_csv_bytes([reversed_first, reversed_second])
    )


def test_index_csv_preserves_commas_quotes_unicode_and_newlines() -> None:
    special = 'vírgula, "aspas"\nsegunda linha'
    serialized = canonical_index_csv_bytes(
        [_index_row("id-1", relative_path="cães/um.jpg", reason=special)]
    )

    assert not serialized.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in serialized
    parsed = list(csv.DictReader(StringIO(serialized.decode("utf-8"), newline="")))
    assert parsed[0]["relative_path"] == "cães/um.jpg"
    assert parsed[0]["reason"] == special


@pytest.mark.parametrize(
    "row",
    [
        lambda: {key: "" for key in INDEX_CSV_HEADER if key != "reason"},
        lambda: {**_index_row("id"), "unexpected": "value"},
        lambda: {**_index_row("id"), 1: "non-string-field"},
    ],
)
def test_index_csv_rejects_rows_that_do_not_match_header(row: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_index_csv_bytes([row()])  # type: ignore[operator, list-item]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), (1, 2)])
def test_index_csv_rejects_unsupported_cell_values(invalid: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_index_csv_bytes([_index_row("id", width=invalid)])


def test_index_csv_rejects_duplicate_or_empty_sample_ids() -> None:
    with pytest.raises(CanonicalizationError, match="duplicate sample_id"):
        canonical_index_csv_bytes([_index_row("id"), _index_row("id")])
    with pytest.raises(CanonicalizationError, match="non-empty"):
        canonical_index_csv_bytes([_index_row("")])


def test_index_csv_digest_detects_changed_data() -> None:
    original = [_index_row("id", status="accepted")]
    changed = [_index_row("id", status="excluded")]

    assert canonical_index_csv_sha256(original) != canonical_index_csv_sha256(changed)


def test_sample_id_is_stable_and_normalizes_to_posix() -> None:
    version = "owner/dataset@version-1"
    expected = make_sample_id(version, "train/chihuahua/image.jpg")

    assert make_sample_id(version, "train\\chihuahua\\image.jpg") == expected
    assert make_sample_id(version, "./train//chihuahua/image.jpg") == expected
    assert make_sample_id(
        version, PureWindowsPath("train/chihuahua/image.jpg")
    ) == expected
    assert normalize_relative_path("./train//chihuahua/image.jpg") == (
        "train/chihuahua/image.jpg"
    )


def test_sample_id_golden_value_with_unicode_path() -> None:
    assert make_sample_id(
        "samuelcortinhas/muffin-vs-chihuahua-image-classification@1",
        "train/chihuahua/cão 01.jpg",
    ) == "6b3b2cf6cbeeba272b9076d6a4660f879ee48748e65c2b2b1bd05cb0c1b6055e"


def test_sample_id_changes_with_version_or_path() -> None:
    first = make_sample_id("dataset@1", "class/image.jpg")

    assert make_sample_id("dataset@2", "class/image.jpg") != first
    assert make_sample_id("dataset@1", "class/other.jpg") != first
    assert make_sample_id("dataset", "ab/c") != make_sample_id("dataset/a", "b/c")


@pytest.mark.parametrize("version", ["", "   ", None, 1])
def test_sample_id_rejects_missing_or_invalid_version(version: object) -> None:
    with pytest.raises(IdentifierError, match="dataset_version_id"):
        make_sample_id(version, "image.jpg")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute/image.jpg",
        "../image.jpg",
        "class/../image.jpg",
        r"C:\dataset\image.jpg",
        r"\\server\share\image.jpg",
    ],
)
def test_sample_id_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(IdentifierError):
        make_sample_id("dataset@1", path)


def test_sample_id_is_independent_from_content_hash() -> None:
    content = b"identical file bytes"
    first_content_hash = sha256_hex(content)
    second_content_hash = sha256_hex(content)

    assert first_content_hash == second_content_hash
    assert make_sample_id("dataset@1", "a/image.jpg") != make_sample_id(
        "dataset@1", "b/image.jpg"
    )
