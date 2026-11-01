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
from .inventory import (
    AuditStatus,
    ExactDuplicateCandidate,
    InventoryError,
    InventoryRecord,
    InventoryResult,
    inventory_dataset,
    sha256_file,
    write_initial_audit,
)
from .preprocessing import PreprocessedImage, PreprocessingError, preprocess_image
from .provenance import (
    DatasetProvenance,
    EvidenceStatus,
    HumanUseAuthorization,
    MetadataEvidence,
    ProvenanceError,
    authorization_from_config,
    provenance_from_config,
)
from .records import ProjectConfig

__all__ = [
    "AcquisitionError",
    "AcquisitionResult",
    "AuditStatus",
    "INDEX_CSV_HEADER",
    "CanonicalizationError",
    "DEFAULT_CONFIG_PATH",
    "ConfigError",
    "IdentifierError",
    "DatasetProvenance",
    "EvidenceStatus",
    "ExactDuplicateCandidate",
    "HumanUseAuthorization",
    "InventoryError",
    "InventoryRecord",
    "InventoryResult",
    "KaggleHubAdapter",
    "MetadataEvidence",
    "ProjectConfig",
    "PreprocessedImage",
    "PreprocessingError",
    "ProvenanceError",
    "ValidationMode",
    "canonical_index_csv_bytes",
    "canonical_index_csv_sha256",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "authorization_from_config",
    "inventory_dataset",
    "load_config",
    "make_sample_id",
    "manifest_sha256",
    "normalize_relative_path",
    "provenance_from_config",
    "preprocess_image",
    "sha256_file",
    "sha256_hex",
    "write_initial_audit",
]
