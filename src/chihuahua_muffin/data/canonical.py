"""Canonical serialization, digests, and stable identifiers for SPEC 01.

Canonical JSON accepts only ``None``, exact ``bool``, ``int``, finite ``float``,
valid UTF-8 ``str``, ``list`` and ``dict`` values. Dictionary keys must be
strings. Other Python types and cyclic structures are rejected explicitly.

The canonical index CSV uses :data:`INDEX_CSV_HEADER`, sorts rows by
``sample_id`` and accepts only scalar cell values: ``None``, exact ``bool``,
``int``, finite ``float`` and valid UTF-8 ``str``.
"""

from __future__ import annotations

from collections.abc import Iterable
import csv
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import PurePath, PureWindowsPath
from typing import TypeAlias


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
CsvValue: TypeAlias = JsonScalar

INDEX_CSV_HEADER = (
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

_SAMPLE_ID_DOMAIN = b"chihuahua-muffin:sample-id:v1\x00"


class CanonicalizationError(ValueError):
    """Raised when a value cannot be represented by a canonical contract."""


class IdentifierError(ValueError):
    """Raised when stable identifier inputs are absent or unsafe."""


def _validate_utf8(value: str, path: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CanonicalizationError(f"{path} is not valid UTF-8 text") from error


def _validate_json_value(
    value: object,
    *,
    path: str = "$",
    active_containers: set[int] | None = None,
) -> None:
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise CanonicalizationError(f"{path} must be a finite float")
        return
    if type(value) is str:
        _validate_utf8(value, path)
        return

    if active_containers is None:
        active_containers = set()

    if type(value) is list:
        identity = id(value)
        if identity in active_containers:
            raise CanonicalizationError(f"{path} contains a cycle")
        active_containers.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_json_value(
                    item,
                    path=f"{path}[{index}]",
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(identity)
        return

    if type(value) is dict:
        identity = id(value)
        if identity in active_containers:
            raise CanonicalizationError(f"{path} contains a cycle")
        active_containers.add(identity)
        try:
            for key, item in value.items():
                if type(key) is not str:
                    raise CanonicalizationError(
                        f"{path} contains a non-string dictionary key"
                    )
                _validate_utf8(key, f"{path} key")
                _validate_json_value(
                    item,
                    path=f"{path}[{key!r}]",
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(identity)
        return

    raise CanonicalizationError(
        f"{path} has unsupported type {type(value).__name__}"
    )


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Return compact, sorted UTF-8 JSON bytes with no trailing newline."""

    _validate_json_value(value)
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return serialized.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise CanonicalizationError("value cannot be serialized canonically") from error


def sha256_hex(data: bytes) -> str:
    """Return the lowercase SHA-256 hexadecimal digest of exact bytes."""

    if type(data) is not bytes:
        raise CanonicalizationError("SHA-256 input must have type bytes")
    return sha256(data).hexdigest()


def canonical_json_sha256(value: JsonValue) -> str:
    """Hash the exact bytes returned by :func:`canonical_json_bytes`."""

    return sha256_hex(canonical_json_bytes(value))


def manifest_sha256(manifest: dict[str, JsonValue], *, digest_field: str) -> str:
    """Hash a manifest after removing only its top-level self-digest field."""

    if type(manifest) is not dict:
        raise CanonicalizationError("manifest must have type dict")
    if type(digest_field) is not str or not digest_field:
        raise CanonicalizationError("digest_field must be a non-empty string")
    if digest_field in manifest and type(manifest[digest_field]) is not str:
        raise CanonicalizationError("manifest self-digest field must be a string")

    payload = dict(manifest)
    payload.pop(digest_field, None)
    return canonical_json_sha256(payload)


def _csv_cell(value: object, *, field: str, row_number: int) -> str:
    location = f"row {row_number}, field {field!r}"
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise CanonicalizationError(f"{location} must be a finite float")
        return json.dumps(value, allow_nan=False)
    if type(value) is str:
        _validate_utf8(value, location)
        return value
    raise CanonicalizationError(
        f"{location} has unsupported type {type(value).__name__}"
    )


def canonical_index_csv_bytes(rows: Iterable[dict[str, CsvValue]]) -> bytes:
    """Serialize index rows with a fixed header and deterministic row order.

    Quoting uses ``csv.QUOTE_MINIMAL``, comma as delimiter, double quote as the
    quote character, doubled quotes inside quoted fields, no escape character,
    and LF as the record terminator. Rows must contain exactly the documented
    fields and have unique, non-empty string ``sample_id`` values.
    """

    expected_fields = set(INDEX_CSV_HEADER)
    normalized_rows: list[tuple[str, list[str]]] = []
    seen_sample_ids: set[str] = set()

    try:
        iterator = iter(rows)
    except TypeError as error:
        raise CanonicalizationError("CSV rows must be iterable") from error

    for row_number, row in enumerate(iterator, start=1):
        if type(row) is not dict:
            raise CanonicalizationError(f"row {row_number} must have type dict")
        if any(type(field) is not str for field in row):
            raise CanonicalizationError(
                f"row {row_number} contains a non-string field name"
            )
        row_fields = set(row)
        if row_fields != expected_fields:
            missing = sorted(expected_fields - row_fields)
            extra = sorted(row_fields - expected_fields)
            raise CanonicalizationError(
                f"row {row_number} fields do not match fixed header; "
                f"missing={missing!r}, extra={extra!r}"
            )

        sample_id = row["sample_id"]
        if type(sample_id) is not str or not sample_id:
            raise CanonicalizationError(
                f"row {row_number} sample_id must be a non-empty string"
            )
        _validate_utf8(sample_id, f"row {row_number} sample_id")
        if sample_id in seen_sample_ids:
            raise CanonicalizationError(f"duplicate sample_id: {sample_id!r}")
        seen_sample_ids.add(sample_id)

        cells = [
            _csv_cell(row[field], field=field, row_number=row_number)
            for field in INDEX_CSV_HEADER
        ]
        normalized_rows.append((sample_id, cells))

    normalized_rows.sort(key=lambda item: item[0])

    stream = StringIO(newline="")
    writer = csv.writer(
        stream,
        delimiter=",",
        quotechar='"',
        escapechar=None,
        doublequote=True,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    writer.writerow(INDEX_CSV_HEADER)
    for _, cells in normalized_rows:
        writer.writerow(cells)
    try:
        return stream.getvalue().encode("utf-8")
    except UnicodeEncodeError as error:
        raise CanonicalizationError("CSV contains invalid UTF-8 text") from error


def canonical_index_csv_sha256(rows: Iterable[dict[str, CsvValue]]) -> str:
    """Hash the deterministic index CSV bytes."""

    return sha256_hex(canonical_index_csv_bytes(rows))


def normalize_relative_path(relative_path: str | PurePath) -> str:
    """Normalize a safe source path to relative POSIX form without I/O."""

    if isinstance(relative_path, PurePath):
        raw_path = relative_path.as_posix()
    elif type(relative_path) is str:
        raw_path = relative_path
    else:
        raise IdentifierError("relative_path must be a string or PurePath")

    if not raw_path or "\x00" in raw_path:
        raise IdentifierError("relative_path must be non-empty and contain no NUL")
    _validate_identifier_text(raw_path, "relative_path")

    windows_path = PureWindowsPath(raw_path)
    posix_text = raw_path.replace("\\", "/")
    if posix_text.startswith("/") or windows_path.drive:
        raise IdentifierError("relative_path must not be absolute or drive-qualified")

    parts: list[str] = []
    for part in posix_text.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            raise IdentifierError("relative_path must not contain '..' traversal")
        parts.append(part)

    if not parts:
        raise IdentifierError("relative_path must identify a file")
    return "/".join(parts)


def _validate_identifier_text(value: str, field: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise IdentifierError(f"{field} is not valid UTF-8 text") from error


def _length_prefixed(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(8, byteorder="big", signed=False) + encoded


def make_sample_id(dataset_version_id: str, relative_path: str | PurePath) -> str:
    """Create a stable SHA-256 ID from version identity and normalized path.

    Components are UTF-8 encoded and prefixed by unsigned 64-bit big-endian
    byte lengths after a versioned domain separator, making their boundaries
    unambiguous. The returned identifier is the 64-character lowercase digest.
    """

    if type(dataset_version_id) is not str or not dataset_version_id.strip():
        raise IdentifierError("dataset_version_id must be a non-empty string")
    if "\x00" in dataset_version_id:
        raise IdentifierError("dataset_version_id must not contain NUL")
    _validate_identifier_text(dataset_version_id, "dataset_version_id")

    normalized_path = normalize_relative_path(relative_path)
    payload = (
        _SAMPLE_ID_DOMAIN
        + _length_prefixed(dataset_version_id)
        + _length_prefixed(normalized_path)
    )
    return sha256(payload).hexdigest()
