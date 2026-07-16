# ADR 0014: Framework-neutral project neural-study identity

Status: accepted for the project reconstruction

The five project neural variants are one immutable framework-neutral study specification. Their
ordered scalar inputs, target, submitted result identifiers, dividend representations, shared
four-by-282 MLP, positive Softplus head, optimizer, objective, scheduler, batching, validation, and
checkpoint rules are values rather than framework defaults. The declared parameter counts provide
an independent preflight for any backend implementation. Device, thread, compilation, and memory
policies remain a separate execution specification and cannot alter the financial or model identity.

Cash-event models use one canonical storage representation: chronological events, right padding,
explicit validity masks and lengths, an inclusive maturity boundary, and cash amounts divided by
spot. The direct LSTM reverses each logical prefix and therefore consumes events from farthest to
nearest, as declared in the submitted methodology; it selects the state at the last real event so
right padding never changes the representation. Event attention must exclude every masked slot.
The 12-bin producer uses chronological intervals `(lower, upper]` over a one-year horizon, then
reverses the fixed vector so the farthest monthly bin enters the MLP first, matching the submitted
runner. These semantics are an explicit clean reconstruction contract; the saved result names
identify the submitted aggregate comparison but do not imply recovery of missing model weights or
prediction artifacts.
