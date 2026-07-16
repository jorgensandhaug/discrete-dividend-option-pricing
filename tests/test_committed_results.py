"""Internal checks for the aggregate-only committed publication receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

_ROOT = Path(__file__).resolve().parents[1]
_RESULTS = _ROOT / "results"
_RECEIPT_PATH = _RESULTS / "reproduction-receipt.json"
_OBSERVATIONS = 678_122
_REPLICATIONS = 12
_SHA256_HEX_LENGTH = 64
_BOOTSTRAP_STREAMS = 18


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_reproduction_receipt_is_canonical_and_internally_bound() -> None:
    """Public aggregates retain one machine-checkable lineage without licensed payloads."""
    raw = _RECEIPT_PATH.read_bytes()
    receipt = cast("dict[str, object]", json.loads(raw))
    canonical = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    assert raw == canonical
    assert set(receipt) == {"feature", "format", "inference", "pricing", "tables", "verification"}
    assert receipt["format"] == "discrete-dividend-option-pricing-reproduction-receipt-v1"

    feature = cast("dict[str, object]", receipt["feature"])
    pricing = cast("dict[str, object]", receipt["pricing"])
    inference = cast("dict[str, object]", receipt["inference"])
    pricing_source = cast("dict[str, object]", pricing["source"])
    inference_source = cast("dict[str, object]", inference["source"])

    assert pricing_source["feature_artifact_sha256"] == feature["output_sha256"]
    assert inference_source["feature_artifact_sha256"] == feature["output_sha256"]
    assert pricing_source["feature_schema_sha256"] == feature["feature_schema_sha256"]
    assert inference_source["feature_schema_sha256"] == feature["feature_schema_sha256"]
    assert inference["feature_receipt_sha256"] == feature["receipt_sha256"]
    assert inference["publication_result_sha256"] == pricing["artifact_sha256"]
    assert inference["publication_result_request_sha256"] == pricing["request_sha256"]
    assert inference["partitions_sha256"] == pricing["partitions_sha256"]
    assert inference["study_sha256"] == pricing["study_sha256"]
    assert inference["observations"] == pricing["observations"] == _OBSERVATIONS

    replications = cast("list[dict[str, object]]", pricing["cash_replications"])
    assert [item["index"] for item in replications] == list(range(_REPLICATIONS))
    for field in ("artifact_sha256", "execution_sha256", "request_sha256"):
        values = [cast("str", item[field]) for item in replications]
        assert len(values) == len(set(values)) == _REPLICATIONS
        assert all(len(value) == _SHA256_HEX_LENGTH for value in values)

    domains = cast("list[int]", inference["bootstrap_domains"])
    configurations = cast("list[dict[str, object]]", inference["bootstrap_configurations"])
    cohorts = cast("list[dict[str, object]]", inference["cohorts"])
    assert (
        len(domains)
        == len(set(domains))
        == len(configurations) * len(cohorts)
        == _BOOTSTRAP_STREAMS
    )
    coverage = cast("list[dict[str, object]]", feature["coverage"])
    assert [item["ticker"] for item in coverage] == ["DUK", "EPD", "ETR", "MO", "SO", "VZ"]
    assert coverage[2]["quote_date_max"] == "2019-11-07"

    tables = cast("dict[str, str]", receipt["tables"])
    artifact_sha256 = cast("str", inference["artifact_sha256"])
    for name, expected_sha256 in tables.items():
        path = _RESULTS / name
        assert _sha256(path) == expected_sha256
        first_line = path.read_text().splitlines()[0]
        assert first_line == f"% source_artifact_sha256={artifact_sha256}"
