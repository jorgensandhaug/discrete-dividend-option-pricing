# ADR 0018: Immutable neural rolling-window runs and pooled project metrics

Status: accepted

Each neural run receives one typed source identity, one already validated rolling window, the study,
an explicit execution profile, and two unsigned 64-bit seeds. Polars collects the train, validation,
and test projections together so common-subplan elimination and projection pushdown can operate
across all three half-open phases. The materialized boundary freezes every NumPy/native buffer and
streams its exact model identity, dtype, shape, and bytes into a phase SHA-256 without serializing a
second copy. Empty, aliased, cross-model, or content-identical phases fail before training.

Inference uses the execution profile's explicit bounded physical batch size, transfers source
tensors and the complete index plan once, preserves source order, and includes every test row
exactly once. Training and inference geometries are independent because their measured optima differ;
both are immutable run values. The submitted headline metrics are pooled over observations: MAPE is
`100 * mean(abs(prediction - market) / market)`, and RMSE is
`sqrt(mean((prediction - market) ** 2))`. MAE, signed price bias, unscaled relative signed bias, and
sMAPE are retained as diagnostics. Market prices must be strictly positive and predictions must be
nonnegative and finite. The legacy analysis notebook's negative-price clipping is not an implicit
fallback; an invalid model output fails.

Best-restored weights and test predictions use separate atomic, content-addressed safe-tensors
files. The prediction file embeds its exact targets, predictions, metric values, model/window/data
identity, and checkpoint hash, then reloads and re-evaluates itself before success. A canonical
content-addressed JSON run manifest binds those artifacts to the ordered feature-schema hash,
feature-file hash, exact paper and engine Git commits, dependency-lock hash, all phase hashes and row
counts, study, profile, seeds, resolved runtime, package/compiler versions, training summary, and
pooled test metrics. Local paths, clocks, wildcard discovery, mutable aliases, licensed rows, and
automatic backend substitution are absent from the identity.
