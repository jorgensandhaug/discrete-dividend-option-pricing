# ADR 0009: Point-in-time forecast-dividend features

Status: accepted for the project as-submitted reconstruction

Forecast schedules are computed once per unique ticker, quote date, and expiry, then joined
many-to-one back to option rows in original quote order. Every ticker/date must exist in the daily
snapshot; missing provider coverage is an error and is never interpreted as a zero-dividend stock.
Input rows must already satisfy the strict future-event boundary.

The project as-submitted identity retains zero-amount rows and distinct same-date cash components,
includes events whose ex-date equals expiry, and converts calendar-day offsets using 365 days per
year. Alternative zero, aggregation, or expiry-boundary policies have separate immutable
specifications. Durable dates and amounts remain `List(Date)` and `List(Float64)`.

The submitted scalar feature is named `forecasted_dividend_yield` for compatibility and is defined
exactly as `log(1 + total undiscounted forecast cash / spot) / (calendar DTE / 365)`. It is retained
beside, and does not alter, the discrete event dates and cash amounts.
