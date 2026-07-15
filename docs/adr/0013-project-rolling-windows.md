# ADR 0013: Project rolling-window identity

Status: accepted for the project as-submitted reconstruction

The project schedule uses calendar-day arithmetic and adjacent half-open intervals. Each test start
looks back 1,095 days in total: the first 1,065 days are training and the final 30 are validation.
The following 30 days are test. Test starts advance by exactly 30 days from 2015-01-01 while the
test end remains on or before 2024-09-01, resolving to 117 ordered windows.

The immutable specification names total lookback, validation, test, and step durations separately
so no caller must infer interval semantics. Its resolved five-column schedule can be persisted in a
run manifest. All selections use explicit phase enums and half-open boundaries. A compact preflight
groups only quote dates once, computes all phase counts without expanding rows across overlapping
windows, and rejects any empty required partition before model execution.

This identity is taken from the pre-refactor runner preserved in repository commit `078edaf...`,
which contains the retained submitted-result notebook. That implementation selects every phase as
`[start, end)`, advances the test origin by a fixed 30-day `timedelta`, and stops before a test end
later than 2024-09-01. The retained notebook scans the six-ticker `20150101_20240901` result tree
and records 699,737 test observations for each selected model. Later explicit-bound rerun configs do
not alter this frozen submitted identity.

For stateless classical methods, the adjacent test intervals are projected as one lazy range before
materialization, so every evaluated contract is priced once and Parquet predicate/projection
pushdown remains available. A date-count preflight still rejects an empty individual test window.
