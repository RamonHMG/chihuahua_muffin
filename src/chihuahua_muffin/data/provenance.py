"""Provenance records and acquisition gates for SPEC 01."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from .records import DatasetConfig, HumanAuthorizationConfig


class EvidenceStatus(StrEnum):
    """Review state for one public metadata value."""

    CONFIRMED = "confirmed"
    PENDING = "pending"
    AMBIGUOUS = "ambiguous"


class ProvenanceError(ValueError):
    """Raised when provenance is invalid or not ready for acquisition."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


@dataclass(frozen=True, slots=True)
class MetadataEvidence:
    """A metadata value together with its verification state and evidence."""

    value: str
    status: EvidenceStatus
    evidence_url: str
    note: str = ""

    def __post_init__(self) -> None:
        if type(self.value) is not str:
            raise TypeError("metadata value must be a string")
        if not isinstance(self.status, EvidenceStatus):
            raise TypeError("metadata status must be EvidenceStatus")
        if type(self.evidence_url) is not str:
            raise TypeError("metadata evidence_url must be a string")
        if type(self.note) is not str:
            raise TypeError("metadata note must be a string")
        if self.status is EvidenceStatus.CONFIRMED:
            if not self.value.strip():
                raise ValueError("confirmed metadata must have a non-empty value")
            if not self.evidence_url.strip():
                raise ValueError("confirmed metadata must have an evidence URL")
        if self.status is EvidenceStatus.AMBIGUOUS and not self.note.strip():
            raise ValueError("ambiguous metadata must explain the ambiguity")


@dataclass(frozen=True, slots=True)
class HumanUseAuthorization:
    """Explicit exception for one immutable dataset and local educational use."""

    source: str
    source_id: str
    handle: str
    version: str
    purpose: str
    decision_date: str
    authorized_by: str
    license_ambiguity_acknowledged: bool
    local_use_only: bool
    redistribution_allowed: bool

    def __post_init__(self) -> None:
        text_fields = {
            "source": self.source,
            "source_id": self.source_id,
            "handle": self.handle,
            "version": self.version,
            "purpose": self.purpose,
            "decision_date": self.decision_date,
            "authorized_by": self.authorized_by,
        }
        for field, value in text_fields.items():
            if type(value) is not str or not value.strip():
                raise ValueError(f"authorization {field} must be a non-empty string")
        for field in (
            "license_ambiguity_acknowledged",
            "local_use_only",
            "redistribution_allowed",
        ):
            if type(getattr(self, field)) is not bool:
                raise TypeError(f"authorization {field} must be bool")
        try:
            date.fromisoformat(self.decision_date)
        except ValueError as error:
            raise ValueError("authorization decision_date must use YYYY-MM-DD") from error

    def mismatch_reasons(self, provenance: DatasetProvenance) -> tuple[str, ...]:
        """Explain why this exception cannot release the given provenance."""

        reasons: list[str] = []
        expected = {
            "source": provenance.source,
            "source_id": provenance.source_id,
            "handle": provenance.handle,
            "version": provenance.version.value,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                reasons.append(f"authorization {field} does not match provenance")
        if self.purpose != "educational_local_use":
            reasons.append("authorization purpose is not educational_local_use")
        if self.license_ambiguity_acknowledged is not True:
            reasons.append("authorization does not acknowledge license ambiguity")
        if self.local_use_only is not True:
            reasons.append("authorization is not restricted to local use")
        if self.redistribution_allowed is not False:
            reasons.append("authorization must prohibit redistribution")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Public identity and evidence for one dataset version."""

    source: str
    source_id: str
    handle: str
    owner: str
    source_url: str
    metadata_url: str
    metadata_checked_at: str
    version: MetadataEvidence
    license: MetadataEvidence

    def __post_init__(self) -> None:
        text_fields = {
            "source": self.source,
            "source_id": self.source_id,
            "handle": self.handle,
            "owner": self.owner,
            "source_url": self.source_url,
            "metadata_url": self.metadata_url,
            "metadata_checked_at": self.metadata_checked_at,
        }
        for field, value in text_fields.items():
            if type(value) is not str or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if self.source != "kaggle":
            raise ValueError("source must be 'kaggle' for this adapter")
        handle_parts = self.handle.split("/")
        if len(handle_parts) != 2 or any(not part for part in handle_parts):
            raise ValueError("handle must use the Kaggle owner/dataset format")
        _parse_timestamp(self.metadata_checked_at)

    @property
    def blockers(self) -> tuple[str, ...]:
        """Return human-readable reasons that currently block acquisition."""

        return self.acquisition_blockers()

    def acquisition_blockers(
        self,
        authorization: HumanUseAuthorization | None = None,
    ) -> tuple[str, ...]:
        """Return blockers, allowing only a matching ambiguity exception."""

        blockers: list[str] = []
        if self.version.status is not EvidenceStatus.CONFIRMED:
            blockers.append(
                f"dataset version is {self.version.status.value}, not confirmed"
            )
        elif not self.version.value.strip():
            blockers.append("dataset version is absent")
        if self.license.status is EvidenceStatus.AMBIGUOUS and authorization is not None:
            if not self.license.value.strip():
                blockers.append("dataset license is absent")
            else:
                blockers.extend(authorization.mismatch_reasons(self))
        elif self.license.status is not EvidenceStatus.CONFIRMED:
            blockers.append(
                f"dataset license is {self.license.status.value}, not confirmed"
            )
        elif not self.license.value.strip():
            blockers.append("dataset license is absent")
        return tuple(blockers)

    def require_acquisition_ready(
        self,
        authorization: HumanUseAuthorization | None = None,
    ) -> None:
        """Reject metadata that is pending, ambiguous, or absent."""

        blockers = self.acquisition_blockers(authorization)
        if blockers:
            raise ProvenanceError(blockers)

    def versioned_handle_for(
        self,
        authorization: HumanUseAuthorization | None = None,
    ) -> str:
        """Return the immutable KaggleHub handle after applying the gate."""

        self.require_acquisition_ready(authorization)
        version = self.version.value
        if not version.isdecimal() or int(version) <= 0:
            raise ProvenanceError(("Kaggle version must be a positive integer",))
        return f"{self.handle}/versions/{version}"

    @property
    def versioned_handle(self) -> str:
        """Return the immutable KaggleHub dataset handle."""

        return self.versioned_handle_for()

    def immutable_version_id_for(
        self,
        authorization: HumanUseAuthorization | None = None,
    ) -> str:
        """Return the source-scoped identity after applying the gate."""

        self.require_acquisition_ready(authorization)
        return f"kaggle:{self.source_id}:v{self.version.value}"

    @property
    def immutable_version_id(self) -> str:
        """Return the source-scoped identifier intended for stable sample IDs."""

        return self.immutable_version_id_for()


def _parse_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("metadata_checked_at must be an ISO 8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError("metadata_checked_at must include a timezone")
    return timestamp


def provenance_from_config(config: DatasetConfig) -> DatasetProvenance:
    """Build a provenance record from the validated project configuration."""

    try:
        version_status = EvidenceStatus(config.version_status)
        license_status = EvidenceStatus(config.license_status)
    except ValueError as error:
        raise ProvenanceError((f"unknown metadata status: {error}",)) from error

    return DatasetProvenance(
        source=config.source,
        source_id=config.source_id,
        handle=config.handle,
        owner=config.owner,
        source_url=config.source_url,
        metadata_url=config.metadata_url,
        metadata_checked_at=config.metadata_checked_at,
        version=MetadataEvidence(
            value=config.version,
            status=version_status,
            evidence_url=config.metadata_url,
            note=config.version_note,
        ),
        license=MetadataEvidence(
            value=config.license,
            status=license_status,
            evidence_url=config.metadata_url,
            note=config.license_note,
        ),
    )


def authorization_from_config(
    dataset: DatasetConfig,
) -> HumanUseAuthorization:
    """Build the narrow human exception recorded in the project config."""

    config: HumanAuthorizationConfig = dataset.human_authorization
    if config.status != "approved":
        raise ProvenanceError(("human authorization is not approved",))
    return HumanUseAuthorization(
        source=dataset.source,
        source_id=config.source_id,
        handle=config.dataset_handle,
        version=config.dataset_version,
        purpose=config.purpose,
        decision_date=config.decision_date,
        authorized_by=config.authorized_by,
        license_ambiguity_acknowledged=config.license_ambiguity_acknowledged,
        local_use_only=config.local_use_only,
        redistribution_allowed=config.redistribution_allowed,
    )
