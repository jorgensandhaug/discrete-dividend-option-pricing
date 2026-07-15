"""Tests for immutable empirical-inference artifact orchestration."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from finoptions import CpuExecutor, CpuMathBackend

from discrete_dividend_study import publication_inference_artifact
from discrete_dividend_study.neural_run import StudySourceIdentity
from discrete_dividend_study.publication_artifact import (
    run_or_load_publication_cash_replication,
)
from discrete_dividend_study.publication_groups import (
    prepare_publication_partitions,
    publication_partitions_sha256,
)
from discrete_dividend_study.publication_inference_artifact import (
    load_publication_inference_artifact,
    run_or_load_publication_empirical_inference,
)
from discrete_dividend_study.publication_inference_spec import (
    PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC,
)
from discrete_dividend_study.publication_result_artifact import (
    run_or_load_publication_study_result,
)
from discrete_dividend_study.publication_spec import (
    PublicationDeterministicExecutionSpec,
)
from discrete_dividend_study.study_features import (
    StudyFeatureArtifactKind,
    VerifiedStudyFeatureArtifact,
)
from tests.test_publication_artifact import _plan
from tests.test_publication_estimator import publication_inputs, study

if TYPE_CHECKING:
    from pathlib import Path

    from discrete_dividend_study.publication_inference import (
        PublicationEmpiricalInferenceResult,
    )


def _feature(tmp_path: Path) -> tuple[VerifiedStudyFeatureArtifact, StudySourceIdentity]:
    """Create a byte-verified test feature identity without weakening production loaders."""
    output = tmp_path / "feature-input.bin"
    receipt = tmp_path / "feature-input.bin.receipt.json"
    output.write_bytes(b"immutable-feature-input")
    receipt.write_bytes(b'{"fixture":true}\n')
    output_payload = output.read_bytes()
    receipt_payload = receipt.read_bytes()
    feature = VerifiedStudyFeatureArtifact(
        kind=StudyFeatureArtifactKind.PROJECT_EXPERIMENT,
        path=output,
        receipt_path=receipt,
        request_sha256="1" * 64,
        pipeline_sha256="2" * 64,
        feature_schema_sha256="3" * 64,
        output_sha256=hashlib.sha256(output_payload).hexdigest(),
        receipt_sha256=hashlib.sha256(receipt_payload).hexdigest(),
        output_rows=4,
        output_bytes=len(output_payload),
        receipt_bytes=len(receipt_payload),
    )
    source = StudySourceIdentity(
        feature_artifact_sha256=feature.output_sha256,
        feature_schema_sha256=feature.feature_schema_sha256,
        paper_revision="a" * 40,
        engine_revision="b" * 40,
        dependency_lock_sha256="4" * 64,
    )
    return feature, source


def test_inference_artifact_round_trip_recomputes_summaries_and_exactly_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw paired draws bind the result and a valid request never resamples twice."""
    declared = study()
    plan = _plan()
    feature, source_identity = _feature(tmp_path)
    source, schedules = publication_inputs()
    inputs = prepare_publication_partitions(source, schedules, declared)
    cash_paths = tuple(
        run_or_load_publication_cash_replication(
            tmp_path,
            source_identity,
            inputs,
            plan.execution(
                replication,
                threads=2,
                math_backend=CpuMathBackend.RUST_STD_F64,
            ),
            study=declared,
            replication_plan=plan,
        ).artifact.path
        for replication in range(plan.replications)
    )
    final = run_or_load_publication_study_result(
        tmp_path,
        source_identity,
        inputs,
        cash_paths,
        PublicationDeterministicExecutionSpec(
            threads=2,
            math_backend=CpuMathBackend.RUST_STD_F64,
        ),
        study=declared,
        replication_plan=plan,
    )
    frame = pl.DataFrame(
        {
            "ticker": ["DUK", "ETR", "EPD", "MO"],
            "quote_date": [
                date(2019, 11, 6),
                date(2019, 11, 7),
                date(2020, 1, 2),
                date(2020, 1, 3),
            ],
            "call_mid_price": final.target.astype(np.float32),
        },
        schema={"ticker": pl.String, "quote_date": pl.Date, "call_mid_price": pl.Float32},
    )
    frozen = PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC
    inference_spec = replace(
        frozen,
        bootstrap=replace(
            frozen.bootstrap,
            replications=999,
            sensitivity_mean_block_lengths=(20.0,),
        ),
    )
    executor = CpuExecutor(threads=2)
    partition_identity = publication_partitions_sha256(inputs)
    first = run_or_load_publication_empirical_inference(
        tmp_path,
        source_identity,
        feature,
        final,
        frame,
        partition_identity,
        executor,
        study=declared,
        spec=inference_spec,
    )
    assert first.artifact.path.name == f"{first.artifact.sha256}.safetensors"
    assert first.artifact.manifest.publication_result_sha256 == final.artifact.sha256
    assert first.artifact.manifest.partitions_sha256 == partition_identity
    assert first.artifact.manifest.bootstrap_domains == (2_000_000, 2_000_001, 2_000_002)
    assert tuple(item.panels.observations for item in first.result.cohorts) == (4, 3, 2)
    loaded = load_publication_inference_artifact(
        first.artifact.path,
        study=declared,
        spec=inference_spec,
    )
    assert loaded.artifact == first.artifact

    def forbidden_inference(*_: object, **__: object) -> PublicationEmpiricalInferenceResult:
        pytest.fail("verified inference request resampled")

    monkeypatch.setattr(
        publication_inference_artifact,
        "run_publication_empirical_inference",
        forbidden_inference,
    )
    second = run_or_load_publication_empirical_inference(
        tmp_path,
        source_identity,
        feature,
        final,
        frame,
        partition_identity,
        executor,
        study=declared,
        spec=inference_spec,
    )
    assert second.artifact == first.artifact
