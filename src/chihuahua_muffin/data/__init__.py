"""Dataset contract and preparation support for SPEC 01."""

from .acquire import AcquisitionError, AcquisitionResult, KaggleHubAdapter
from .canonical import (
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
from .config import DEFAULT_CONFIG_PATH, ConfigError, ValidationMode, load_config
from .provenance import (
    DatasetProvenance,
    EvidenceStatus,
    MetadataEvidence,
    ProvenanceError,
    provenance_from_config,
)
from .records import ProjectConfig

__all__ = [
    "AcquisitionError",
    "AcquisitionResult",
    "INDEX_CSV_HEADER",
    "CanonicalizationError",
    "DEFAULT_CONFIG_PATH",
    "ConfigError",
    "IdentifierError",
    "DatasetProvenance",
    "EvidenceStatus",
    "KaggleHubAdapter",
    "MetadataEvidence",
    "ProjectConfig",
    "ProvenanceError",
    "ValidationMode",
    "canonical_index_csv_bytes",
    "canonical_index_csv_sha256",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "load_config",
    "make_sample_id",
    "manifest_sha256",
    "normalize_relative_path",
    "provenance_from_config",
    "sha256_hex",
]
