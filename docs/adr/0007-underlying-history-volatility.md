# ADR 0007: Canonical underlying history and submitted volatility estimator

Status: accepted for the project as-submitted reconstruction

Daily underlying history has one non-null row per ticker and observation date, strictly positive
finite `Float64` closes, and no semantic dependence on provider delivery order. Preparation rejects
invalid content before arithmetic and emits a canonical ticker/date order. It does not coerce
precision, repair duplicates, or fill missing observations.

The project as-submitted historical-volatility identity uses dividend-and-split-adjusted close,
daily log returns `log(P_t / P_(t-1))`, complete 30-, 90-, 126-, and 252-observation rolling
windows, sample standard deviation with one degree of freedom, and annualization by `sqrt(252)`.
These values are immutable and explicit; the implementation does not interpret a window as calendar
days or synthesize prices for non-trading days. Split-adjusted close is retained separately for the
historical-dividend-yield stage.

Prepared history joins backward by ticker and date only after both grouped timelines are validated
as nondecreasing. Every distinct quote date must have support. The matched observation date and its
civil-day lag remain explicit output columns, and an immutable join specification records whether
same-date observations are eligible and whether lag is bounded. The as-submitted identity permits
same-date observations and records its historical unlimited backward lag explicitly; alternative
point-in-time policies require separate identities.
