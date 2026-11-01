from dataclasses import replace
from pathlib import Path

import pytest

from chihuahua_muffin.data.config import load_config
from chihuahua_muffin.data.provenance import (
    DatasetProvenance,
    EvidenceStatus,
    MetadataEvidence,
    ProvenanceError,
    provenance_from_config,
)


PROJECT_ROOT = Path(__file__).parents[2]
EXAMPLE_CONFIG = PROJECT_ROOT / "config" / "example.toml"
METADATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/view/"
    "samuelcortinhas/muffin-vs-chihuahua-image-classification"
)


def _evidence(
    value: str,
    status: EvidenceStatus = EvidenceStatus.CONFIRMED,
    *,
    note: str = "",
) -> MetadataEvidence:
    return MetadataEvidence(
        value=value,
        status=status,
        evidence_url=METADATA_URL,
        note=note,
    )


def _ready_provenance() -> DatasetProvenance:
    return DatasetProvenance(
        source="kaggle",
        source_id="2733586",
        handle="samuelcortinhas/muffin-vs-chihuahua-image-classification",
        owner="samuelcortinhas",
        source_url=(
            "https://www.kaggle.com/datasets/"
            "samuelcortinhas/muffin-vs-chihuahua-image-classification"
        ),
        metadata_url=METADATA_URL,
        metadata_checked_at="2026-11-01T17:13:33Z",
        version=_evidence("2"),
        license=_evidence("CC0: Public Domain"),
    )


def test_valid_provenance_identifies_source_and_immutable_version() -> None:
    provenance = _ready_provenance()

    provenance.require_acquisition_ready()
    assert provenance.source == "kaggle"
    assert provenance.source_id == "2733586"
    assert provenance.owner == "samuelcortinhas"
    assert provenance.versioned_handle == (
        "samuelcortinhas/muffin-vs-chihuahua-image-classification/versions/2"
    )
    assert provenance.immutable_version_id == "kaggle:2733586:v2"


def test_example_records_confirmed_version_but_ambiguous_license() -> None:
    provenance = provenance_from_config(load_config(EXAMPLE_CONFIG).dataset)

    assert provenance.version.status is EvidenceStatus.CONFIRMED
    assert provenance.version.value == "2"
    assert provenance.license.value == "CC0: Public Domain"
    assert provenance.license.status is EvidenceStatus.AMBIGUOUS
    assert "Google Images" in provenance.license.note
    assert provenance.blockers == (
        "dataset license is ambiguous, not confirmed",
    )
    with pytest.raises(ProvenanceError, match="license is ambiguous"):
        provenance.require_acquisition_ready()


@pytest.mark.parametrize(
    ("field", "evidence", "message"),
    [
        (
            "version",
            _evidence("", EvidenceStatus.PENDING),
            "version is pending",
        ),
        (
            "license",
            _evidence("", EvidenceStatus.PENDING),
            "license is pending",
        ),
        (
            "license",
            _evidence(
                "Other",
                EvidenceStatus.AMBIGUOUS,
                note="license terms are not specific",
            ),
            "license is ambiguous",
        ),
    ],
)
def test_pending_or_ambiguous_metadata_blocks_acquisition(
    field: str, evidence: MetadataEvidence, message: str
) -> None:
    provenance = replace(_ready_provenance(), **{field: evidence})

    with pytest.raises(ProvenanceError, match=message):
        provenance.require_acquisition_ready()


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "value": "",
            "status": EvidenceStatus.CONFIRMED,
            "evidence_url": METADATA_URL,
        },
        {
            "value": "2",
            "status": EvidenceStatus.CONFIRMED,
            "evidence_url": "",
        },
        {
            "value": "CC0",
            "status": EvidenceStatus.AMBIGUOUS,
            "evidence_url": METADATA_URL,
        },
    ],
)
def test_invalid_metadata_evidence_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        MetadataEvidence(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", "other", "source must"),
        ("source_id", "", "source_id"),
        ("handle", "missing-dataset", "owner/dataset"),
        ("metadata_checked_at", "2026-11-01", "timezone"),
        ("metadata_checked_at", "not-a-date", "ISO 8601"),
    ],
)
def test_invalid_provenance_identity_is_rejected(
    field: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_ready_provenance(), **{field: value})


@pytest.mark.parametrize("version", ["latest", "0", "-1", "1/other"])
def test_kaggle_version_must_be_a_positive_integer(version: str) -> None:
    provenance = replace(_ready_provenance(), version=_evidence(version))

    with pytest.raises(ProvenanceError, match="positive integer"):
        _ = provenance.versioned_handle
