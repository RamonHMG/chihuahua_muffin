import json
from pathlib import Path

import pytest

from chihuahua_muffin.data.canonical import canonical_json_bytes, manifest_sha256
from chihuahua_muffin.model.contract import DatasetContractError, load_dataset_contract


def test_contract_consumes_exact_manifest_assignments(synthetic_model_contract) -> None:
    root, config, _ = synthetic_model_contract

    contract = load_dataset_contract(root, config)

    assert len(contract.samples) == 6
    assert [len(contract.for_split(split)) for split in ("train", "validation", "test")] == [2, 2, 2]
    assert {sample.class_id for sample in contract.samples} == {0, 1}
    assert contract.counts_by_split_and_class["test"] == {
        "chihuahua": 1,
        "muffin": 1,
    }


def test_contract_rejects_changed_index_digest(synthetic_model_contract) -> None:
    root, config, _ = synthetic_model_contract
    index = root / "data/prepared/index.csv"
    index.write_bytes(index.read_bytes() + b"\n")

    with pytest.raises(DatasetContractError, match="index digest"):
        load_dataset_contract(root, config)


def test_contract_rejects_class_mapping_even_with_recomputed_self_digest(
    synthetic_model_contract,
) -> None:
    root, config, _ = synthetic_model_contract
    path = root / "data/prepared/dataset_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["class_mapping"] = {"chihuahua": 1, "muffin": 0}
    manifest["manifest_sha256"] = manifest_sha256(
        manifest, digest_field="manifest_sha256"
    )
    path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(DatasetContractError, match="class mapping"):
        load_dataset_contract(root, config, enforce_approved_digests=False)


def test_contract_rejects_source_file_change(synthetic_model_contract) -> None:
    root, config, _ = synthetic_model_contract
    source = next((root / "data/raw/synthetic").rglob("*.png"))
    source.write_bytes(source.read_bytes() + b"changed")

    with pytest.raises(DatasetContractError, match="source digest"):
        load_dataset_contract(root, config)
