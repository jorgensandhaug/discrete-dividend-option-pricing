# ADR 0014: Framework-neutral project neural-study identity

Status: accepted for the project reconstruction

The five project neural variants are one immutable framework-neutral study specification. Their
ordered scalar inputs, target, submitted result identifiers, dividend representations, shared
four-by-282 MLP, positive Softplus head, optimizer, objective, scheduler, batching, validation, and
checkpoint rules are values rather than framework defaults. The declared parameter counts provide
an independent preflight for any backend implementation. Device, thread, compilation, and memory
policies remain a separate execution specification and cannot alter the financial or model identity.

Cash-event models use one canonical representation: chronological events, right padding, explicit
validity masks and lengths, an inclusive maturity boundary, and cash amounts divided by spot. The
direct LSTM must consume exact lengths so padding never updates recurrent state; event attention
must exclude every masked slot. The 12-bin variant uses chronological intervals `(lower, upper]`
over a one-year horizon. These semantics are an explicit clean reconstruction contract; the saved
result names identify the submitted aggregate comparison but do not imply recovery of missing model
weights or prediction artifacts.
