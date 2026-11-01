"""Small, injectable KaggleHub acquisition adapter for SPEC 01."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
import shutil

from .provenance import DatasetProvenance, HumanUseAuthorization, ProvenanceError


DownloadFunction = Callable[..., str]
Clock = Callable[[], datetime]


class AcquisitionError(RuntimeError):
    """Structured, explicit acquisition failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        destination: Path,
        cause: BaseException | None = None,
        partial_data_remaining: bool = False,
    ) -> None:
        self.code = code
        self.destination = destination
        self.cause = cause
        self.partial_data_remaining = partial_data_remaining
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Auditable facts from a successful acquisition operation."""

    provenance: DatasetProvenance
    versioned_handle: str
    destination: Path
    acquired_at: str
    tool_name: str
    tool_version: str
    file_count: int
    originals_preserved: bool
    human_authorization_used: bool
    authorization_purpose: str | None


class KaggleHubAdapter:
    """Download one immutable dataset version into a new destination.

    Tests inject a local callable with the same keyword arguments as
    ``kagglehub.dataset_download``. The real callable is imported lazily and is
    never invoked unless :meth:`acquire` is called with explicit authorization.
    """

    def __init__(
        self,
        download_function: DownloadFunction | None = None,
        *,
        clock: Clock | None = None,
        tool_version: str | None = None,
    ) -> None:
        if download_function is None:
            from kagglehub import dataset_download

            download_function = dataset_download
        self._download = download_function
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.tool_version = tool_version or package_version("kagglehub")

    def acquire(
        self,
        provenance: DatasetProvenance,
        destination: str | Path,
        *,
        download_authorized: bool = False,
        human_use_authorization: HumanUseAuthorization | None = None,
    ) -> AcquisitionResult:
        """Acquire without overwriting and clean newly created partial data."""

        target = Path(destination)
        if download_authorized is not True:
            raise AcquisitionError(
                "approval_required",
                "dataset download requires explicit human authorization",
                destination=target,
            )
        try:
            versioned_handle = provenance.versioned_handle_for(human_use_authorization)
        except ProvenanceError as error:
            raise AcquisitionError(
                "provenance_blocked",
                f"dataset provenance blocks acquisition: {error}",
                destination=target,
                cause=error,
            ) from error

        if target.exists() or target.is_symlink():
            raise AcquisitionError(
                "destination_exists",
                f"destination already exists and will not be overwritten: {target}",
                destination=target,
            )

        created = False
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.mkdir(exist_ok=False)
            created = True
        except FileExistsError as error:
            raise AcquisitionError(
                "destination_exists",
                f"destination already exists and will not be overwritten: {target}",
                destination=target,
                cause=error,
            ) from error
        except OSError as error:
            raise AcquisitionError(
                "destination_unavailable",
                f"cannot create acquisition destination {target}: {error}",
                destination=target,
                cause=error,
            ) from error

        try:
            returned_path = Path(
                self._download(
                    versioned_handle,
                    output_dir=str(target),
                    force_download=False,
                )
            )
            target_resolved = target.resolve()
            returned_resolved = returned_path.resolve()
            if (
                returned_resolved != target_resolved
                and target_resolved not in returned_resolved.parents
            ):
                raise RuntimeError(
                    "KaggleHub returned a path outside the requested destination"
                )
            files = tuple(path for path in target.rglob("*") if path.is_file())
            if not files:
                raise RuntimeError("acquisition produced no files")
            if not returned_path.exists():
                raise RuntimeError("KaggleHub returned a path that does not exist")
            acquired_at = _utc_timestamp(self._clock())
        except Exception as error:
            if created:
                cleanup_error = _remove_partial_destination(target)
                if cleanup_error is not None:
                    raise AcquisitionError(
                        "partial_cleanup_failed",
                        "acquisition failed and partial data could not be removed: "
                        f"{cleanup_error}",
                        destination=target,
                        cause=error,
                        partial_data_remaining=True,
                    ) from error
            raise AcquisitionError(
                "download_failed",
                f"dataset acquisition failed: {error}",
                destination=target,
                cause=error,
            ) from error

        return AcquisitionResult(
            provenance=provenance,
            versioned_handle=versioned_handle,
            destination=target,
            acquired_at=acquired_at,
            tool_name="kagglehub",
            tool_version=self.tool_version,
            file_count=len(files),
            originals_preserved=True,
            human_authorization_used=human_use_authorization is not None,
            authorization_purpose=(
                human_use_authorization.purpose
                if human_use_authorization is not None
                else None
            ),
        )


def _remove_partial_destination(destination: Path) -> OSError | None:
    try:
        shutil.rmtree(destination)
    except OSError as error:
        return error
    return None


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("acquisition clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
