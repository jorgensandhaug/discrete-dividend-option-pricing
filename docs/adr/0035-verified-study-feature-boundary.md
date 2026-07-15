# ADR 0035: Admit study features only through explicit verified receipts

## Status

Accepted.

## Context

The classical and neural runners can consume the minimal 19-column project-experiment artifact or
the complete 42-column project artifact. A bare Parquet path proves neither its producer contract
nor the transformations and schema recorded by the adjacent receipt. Inferring the artifact kind
from a filename or trying one loader after another would make malformed inputs eligible for a
different interpretation. Each producer must retain its own schema identity through downstream run
manifests.

## Decision

Both production study commands require `--feature-artifact-kind` with exactly one of
`project-experiment` or `project-full`. The selected kind invokes exactly one detached artifact
loader. There is no filename inference, schema coercion, secondary loader, or fallback. The loader
verifies the producer receipt, request and pipeline identities, exact output hash and size, schema,
and aggregate output invariants before returning a unified `VerifiedStudyFeatureArtifact` handle.

Study source identity uses the verified handle's output SHA-256 and its receipt-declared schema
SHA-256. It never substitutes a schema identity selected by the runner. The dependency lock and
explicit full paper and engine revisions remain independently bound to every result. Preparation
scans through the verified handle and, before any native pricing or neural runtime preflight,
rehashes both the feature output and receipt against their original identities. Any mutation is an
early error.

The unified handle retains request and pipeline hashes for inspection, but result identity stores
the exact input bytes and schema that can affect computation. The producer receipt remains the
relocatable proof connecting those bytes to upstream sources and pipeline policy.

## Consequences

A schema-compatible unreceipted Parquet cannot enter a production study. The 19-column and
42-column contracts retain distinct schema identities while using the same downstream preparation
code. Adding another feature producer requires a new explicit enum
member and loader branch; it cannot become an accidental fallback. Receipt validation and two
post-preparation hashes add bounded I/O before expensive execution and prevent ambiguous cache
reuse or silent source drift.
