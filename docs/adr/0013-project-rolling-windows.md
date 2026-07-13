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
