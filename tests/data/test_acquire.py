from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from chihuahua_muffin.data.acquire import AcquisitionError, KaggleHubAdapter
from chihuahua_muffin.data.provenance import (
    DatasetProvenance,
    EvidenceStatus,
    MetadataEvidence,
)


METADATA_URL = "https://www.kaggle.com/api/v1/datasets/view/owner/dataset"
FIXED_TIME = datetime(2026, 11, 1, 18, 30, tzinfo=timezone.utc)


def _ready_provenance() -> DatasetProvenance:
    return DatasetProvenance(
        source="kaggle",
        source_id="123",
        handle="owner/dataset",
        owner="owner",
        source_url="https://www.kaggle.com/datasets/owner/dataset",
        metadata_url=METADATA_URL,
        metadata_checked_at="2026-11-01T17:13:33Z",
        version=MetadataEvidence("7", EvidenceStatus.CONFIRMED, METADATA_URL),
        license=MetadataEvidence("CC0", EvidenceStatus.CONFIRMED, METADATA_URL),
    )


def _adapter(download_function: object) -> KaggleHubAdapter:
    return KaggleHubAdapter(
        download_function,  # type: ignore[arg-type]
        clock=lambda: FIXED_TIME,
        tool_version="1.0.3",
    )


def test_simulated_acquisition_records_result_and_preserves_files(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, bool]] = []
    original_bytes = b"original image bytes\x00\xff"

    def download(handle: str, *, output_dir: str, force_download: bool) -> str:
        calls.append((handle, output_dir, force_download))
        image = Path(output_dir) / "train" / "chihuahua" / "image.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(original_bytes)
        return output_dir

    destination = tmp_path / "raw" / "version-7"
    result = _adapter(download).acquire(
        _ready_provenance(), destination, download_authorized=True
    )

    assert calls == [("owner/dataset/versions/7", str(destination), False)]
    assert (destination / "train/chihuahua/image.jpg").read_bytes() == original_bytes
    assert result.versioned_handle == "owner/dataset/versions/7"
    assert result.destination == destination
    assert result.acquired_at == "2026-11-01T18:30:00Z"
    assert result.tool_name == "kagglehub"
    assert result.tool_version == "1.0.3"
    assert result.file_count == 1
    assert result.originals_preserved is True


def test_download_requires_explicit_authorization(tmp_path: Path) -> None:
    called = False

    def download(*args: object, **kwargs: object) -> str:
        nonlocal called
        called = True
        return str(tmp_path / "never")

    destination = tmp_path / "raw"
    with pytest.raises(AcquisitionError) as captured:
        _adapter(download).acquire(_ready_provenance(), destination)

    assert captured.value.code == "approval_required"
    assert called is False
    assert not destination.exists()


def test_ambiguous_provenance_blocks_backend_call(tmp_path: Path) -> None:
    called = False

    def download(*args: object, **kwargs: object) -> str:
        nonlocal called
        called = True
        return str(tmp_path / "never")

    provenance = replace(
        _ready_provenance(),
        license=MetadataEvidence(
            "CC0",
            EvidenceStatus.AMBIGUOUS,
            METADATA_URL,
            "rights require review",
        ),
    )
    with pytest.raises(AcquisitionError) as captured:
        _adapter(download).acquire(
            provenance, tmp_path / "raw", download_authorized=True
        )

    assert captured.value.code == "provenance_blocked"
    assert called is False


@pytest.mark.parametrize("existing_kind", ["directory", "file"])
def test_existing_destination_is_never_overwritten(
    tmp_path: Path, existing_kind: str
) -> None:
    destination = tmp_path / "raw"
    if existing_kind == "directory":
        destination.mkdir()
        marker = destination / "marker.txt"
        marker.write_text("keep", encoding="utf-8")
    else:
        destination.write_text("keep", encoding="utf-8")
        marker = destination
    called = False

    def download(*args: object, **kwargs: object) -> str:
        nonlocal called
        called = True
        return str(destination)

    with pytest.raises(AcquisitionError) as captured:
        _adapter(download).acquire(
            _ready_provenance(), destination, download_authorized=True
        )

    assert captured.value.code == "destination_exists"
    assert called is False
    assert marker.read_text(encoding="utf-8") == "keep"


def test_partial_download_is_removed_and_error_is_explicit(tmp_path: Path) -> None:
    destination = tmp_path / "raw"

    def download(handle: str, *, output_dir: str, force_download: bool) -> str:
        Path(output_dir, "partial.bin").write_bytes(b"partial")
        raise ConnectionError("simulated network failure")

    with pytest.raises(AcquisitionError) as captured:
        _adapter(download).acquire(
            _ready_provenance(), destination, download_authorized=True
        )

    assert captured.value.code == "download_failed"
    assert isinstance(captured.value.cause, ConnectionError)
    assert captured.value.partial_data_remaining is False
    assert not destination.exists()


@pytest.mark.parametrize("failure", ["empty", "outside", "missing-return"])
def test_invalid_backend_result_is_cleaned(
    tmp_path: Path, failure: str
) -> None:
    destination = tmp_path / "raw"

    def download(handle: str, *, output_dir: str, force_download: bool) -> str:
        output = Path(output_dir)
        if failure != "empty":
            (output / "file.bin").write_bytes(b"data")
        if failure == "outside":
            return str(tmp_path / "outside")
        if failure == "missing-return":
            return str(output / "missing")
        return output_dir

    with pytest.raises(AcquisitionError) as captured:
        _adapter(download).acquire(
            _ready_provenance(), destination, download_authorized=True
        )

    assert captured.value.code == "download_failed"
    assert not destination.exists()


def test_installed_adapter_reports_approved_kagglehub_version() -> None:
    adapter = KaggleHubAdapter()

    assert adapter.tool_version == "1.0.3"
