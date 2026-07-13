# ADR 0012: One project as-submitted feature composition

Status: accepted for the project as-submitted reconstruction

The study exposes one lazy composition from canonical option quotes, underlying history, future
dividend forecasts, observed dividend history, and exact-date NSS parameters to a fixed ordered
42-column artifact. The submitted ticker universe is the explicit immutable tuple `DUK`, `EPD`,
`ETR`, `MO`, `SO`, and `VZ`; a source missing any member fails before filtering. Every constituent
transformation retains its separately typed immutable identity; the composition does not introduce
configurable defaults.

Universe and quote-only predicates run before support joins so Polars can reduce work early. The
submitted complete-row null policy runs after all joins. These stages are cohort-equivalent to the
final submitted cleaner because each join validates support, cardinality, and quote-order
preservation. The final projection removes provider/helper columns and renames only the declared
split-adjusted price field. Notebooks and models consume this artifact rather than reimplementing
the chain.
