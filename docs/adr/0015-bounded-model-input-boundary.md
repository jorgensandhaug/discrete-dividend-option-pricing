# ADR 0015: Bounded Polars-to-model input boundary

Status: accepted

Neural input preparation is a stateful materialization boundary and accepts only a Polars
`DataFrame`. It validates the complete required schema and numeric domain before allocating a
contiguous float32 scalar matrix or target vector. Event list lengths must pair exactly; nulls,
nonfinite values, invalid cash events, capacity overflow, and float narrowing outside the native
contract fail immediately. A `LazyFrame` is never collected implicitly.

Arrow-compatible event child values are borrowed zero-copy from single-chunk Polars list storage.
Ragged offsets and float64 spots are bounded temporary allocations. Fragmented list storage fails
under the default policy and is consolidated only when the caller explicitly selects
`ALLOW_RECHUNK`. The boundary does not manufacture one universal dense tensor: sequence variants
allocate only padded values, masks, and lengths; the bin variant allocates only its bins; scalar
yield variants allocate no event tensor. Native kernels receive an explicit reusable CPU executor
and never select a fallback implementation.
