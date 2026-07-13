# ADR 0008: Strict ORATS dividend-forecast snapshots

Status: accepted for the project as-submitted reconstruction

One uncompressed `ORATSStockDivForecasts.txt` snapshot is identified by its filename's first eight
date digits. Historical optional four-digit intraday suffixes and one terminal space are supported
explicitly; malformed or ragged content is not truncated. The canonical lazy schema is ticker,
forecast date, expected ex-date, `Float64` split-adjusted USD amount, and annual payout frequency.
The allowed frequency identifiers are 0, 1, 2, 3, 4, and 12. This interpretation follows ORATS'
[dividend service specification](https://orats.com/blog/dividend-service-details).

Normalization preserves delivery order, zero-amount placeholders, records on or before the
snapshot date, and distinct cash components sharing one ex-date. The separately named future-event
stage applies the submitted strict `ex_date > forecast_date` selection and a deterministic order.
It does not use frequency to infer whether a row is a cash event, collapse same-date components, or
replace a missing snapshot with an earlier one; those require explicit later policies and identities.
