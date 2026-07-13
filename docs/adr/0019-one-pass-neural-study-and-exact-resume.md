# ADR 0019: One-pass neural study preparation and exact request resume

Status: accepted

The complete neural study collects one stable date-sorted projection over the union of the five
models' required columns. It validates every schedule partition before training and materializes
each distinct full-horizon representation once: one shared target vector, one base scalar matrix,
one padded event tensor shared by LSTM and attention, one binned tensor, and the two scalar-yield
matrices. A rolling phase is a read-only contiguous row view into those allocations. Consequently,
the 117 overlapping three-year windows do not repeat Parquet scans, float casts, ragged-event
validation, native padding/binning, or full-phase allocation. Null dates are deliberately retained
through the horizon filter and rejected; empty phases, source mutation during preparation, and
every other violated precondition fail before a model run. The whole-study specification explicitly
permits one event-list rechunk after the Parquet scan because row groups are normally fragmented;
this is a declared bounded consolidation, not an automatic execution or mathematical fallback. The
stricter single-window boundary continues to require zero-copy storage unless its caller explicitly
selects the same policy.

Parameter and shuffle randomness use separate unsigned roots. Canonical SHA-256 domain separation
by role, model variant, and window id derives every resolved 64-bit seed; no backend owns seed
derivation. Before training, each run hashes its source/schema/code/lock identity, model and study,
window and exact phase bytes, execution profile and resolved runtime, inference geometry, resolved
seeds, initial parameters, and package/compiler identity. That request hash is the sole resume key.
An immutable receipt with the exact request-hash filename binds one run, checkpoint, and prediction.
A present but malformed receipt, missing referenced artifact, or cross-reference mismatch is an
error; execution never searches directories, selects a nearby result, silently retrains, or rebinds
one request to different output.

The complete study executes in model-major/window-minor order after preflighting all requested
runtimes and all prepared windows. Its content-addressed manifest contains the exact Cartesian set
of request/run/checkpoint/prediction references and resolved seeds. Publication metrics are pooled
separately for each model by test-observation count; MSE is pooled before taking its square root.
Metrics are never averaged across windows or across distinct models. Interrupted studies resume
through already verified per-run receipts and produce the same schedule-level manifest when every
required result is complete. There are no timestamps, local paths, partial mutable manifests, or
“latest” aliases in any identity.
