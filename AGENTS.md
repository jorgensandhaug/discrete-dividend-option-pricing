# Agent working agreement

## Mission

Produce a fast, reproducible, publication-grade study while keeping licensed inputs local and the
Python workflow obvious to a financial researcher.

## Non-negotiable rules

- Never commit raw or derived licensed row-level data, credentials, or machine-local paths.
- Never silently drop a row, shift an event date, guess units, fill missing financial values,
  substitute a provider field, or change a study/model/estimator configuration.
- Fail with a typed, contextual error when source semantics or point-in-time availability are
  ambiguous.
- Keep financial and execution specifications separate, then record both in an immutable resolved
  manifest.
- Use Polars lazily for stateless preparation and native Parquet sinks for stage boundaries.
  Stateful pricing/training is never disguised as a dataframe expression.
- Notebooks inspect immutable artifacts; they do not implement production transformations.
- Treat legacy results as qualitative reconstruction targets, not exact-digit truth.
- Keep neural study identity framework-neutral. Backend, device, thread, compilation, and memory
  choices belong to a separate execution specification.
- Event sequence consumers use chronological values with explicit masks or lengths; padding is
  never allowed to alter recurrent state or attention weights.
- Do not make this repository public, publish packages, or redistribute data.
- On an interactive workstation, run at most one heavy pricing or training process. Use the typed
  local resource budget, never start automatic restart daemons, and obtain explicit approval before
  selecting a higher-concurrency budget.

## Supported workflow

- `just fmt` formats Python.
- `just check` runs formatting, Ruff, and strict `ty` checks through the current native sibling build.
- `just test` runs the integration suite through the same state-aware path.
- `just full` is the current complete local gate.

Use current stable mutually compatible dependencies and commit the exact uv lock. The engine
revision used by any immutable result must be recorded even during sibling-path development.
