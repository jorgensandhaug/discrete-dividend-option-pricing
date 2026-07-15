# ADR 0046: Calendar-owned publication exercise schedules

- Status: accepted architecture; production replication plan remains predeclaration-gated
- Date: 2026-07-15
- Supersedes: ADR 0037 for new publication runs

## Context

The project comparison prices American-style US equity calls under deterministic forecast cash
events. Cboe specifies equity options as American style, exercisable on any business day through
expiration. The Options Industry Council also explains that dividend capture requires exercise
before the ex-dividend date. Therefore a terminal cash event cannot be represented by subtracting
one civil day or by inventing an epsilon before maturity: its eligible predecessor must come from
an explicit market-session calendar.

The recovered cohort proves that this distinction is real. Fifteen terminal-cash groups expire on
2014-12-26, whose preceding XNYS session was 2014-12-24. The source also contains historical
Saturday legal expirations and a small number of provider forecast ex-dates on non-session civil
dates. Expiration dates therefore cannot all be required to be sessions, and forecast event dates
cannot be silently shifted, snapped, or repaired.

## Decision

Check in the exact XNYS session dates over 2009--2025 as a small immutable package resource. The
asset is generated from `exchange_calendars` 4.13.2, cross-checked byte-for-date against XNAS over
the complete interval, and bound by SHA-256
`19683a870cf0eb1983b1727c0c62fded512c1b7440c0812e2984ea666311cd3b`. The loader independently
checks the hash, metadata, count, strict order, and absence of weekend sessions before exposing the
4,276-date calendar. The generator requires the pinned source version and can either reproduce or
byte-check the asset.

Materialize exercise schedules only from a materialized, already verified natural-group frame:

- require every quote date to be a retained XNYS session;
- require quote-to-expiry maturity and quote-to-event time to equal ACT/365F civil-date counts;
- admit at most the one-ULP encoding difference observed between scalar and vector `Float64`
  division, while preserving the source values unchanged;
- retain every strictly pre-expiry forecast event as its declared cash-jump left-limit
  opportunity, whether or not the provider forecast civil date is a retained session;
- when the final forecast cash date equals expiry, replace only that terminal opportunity with the
  exact prior XNYS session and require the expiry itself to be a retained session;
- reject missing dates, changed group boundaries, unordered events, out-of-range dates, absent
  calendar coverage, predecessor collisions, and all source/schedule identity disagreement.

Terminal classification is made from exact civil dates, never floating-point equality. Legal
expiration dates without terminal cash are not reinterpreted as trading sessions. The result owns
Arrow-compatible ragged `Float64` exercise times, an immutable terminal-group mask, the source
prepared/group-key identities, calendar name/source/version/hash, and an independent schedule
SHA-256.

Use two exhaustive publication partitions. Every nonnegative-rate group enters `exact` and every
negative-rate group enters `general`. Cash variants use the exact explicit schedule only in the
first partition; the second uses the predeclared origin-inclusive uniform American mesh because the
cash-call exercise specialization does not hold at negative rates. Both proportional-yield
variants use the uniform mesh in both partitions. The partition is target-free, complete,
disjoint, source-order reconstructible, and has no retry or fallback path.

This replaces ADR 0037's terminal-cash/general split. The generic explicit-schedule engine has
effectively zero measured overhead relative to its older cash-specialized API, while representing
the terminal predecessor directly. Random-domain addressing remains the fixed partition order
`exact, general`; the general root is offset by the exact group count for every paired variant.

## Recovered-data evidence

The full 894,509-row recovered feature artifact materializes into 127,185 groups, 136,128 cash
events, and 2,393 terminal groups. Its schedule SHA-256 is
`26aa9e41e6d3bd1221813438deaad4e0534616f40c1a2893df646b97fd4c8638`; 117,535 groups and 816,421
rows are exact-domain, while 9,650 groups and 78,088 rows are negative-rate general-domain.

The exact submitted 117-window recoverable union materializes into 678,122 rows, 96,666 groups,
98,860 cash events, and 2,325 terminal groups. Its schedule SHA-256 is
`3008e547d044d3985452b41e99a08c9a8df6131750daee8b7e3996bfb13ac5e6`; 94,906 groups and 664,185
rows are exact-domain, while 1,760 groups and 13,937 rows are negative-rate general-domain. A warm
local verification, collection, compaction, calendar materialization, and partition took 0.261
seconds on the M4 Pro. These counts and schedule hashes do not freeze the still-pending production
path budget or empirical replication plan.

## Primary references

- [Cboe equity option specifications](https://www.cboe.com/tradable_products/equity_indices/leaps_options/specifications)
- [Options Industry Council exercise FAQ](https://www.optionseducation.org/referencelibrary/faq/options-exercise)
- [NYSE hours and calendars](https://www.nyse.com/trade/hours-calendars)
- [NYSE 2018 national-day-of-mourning closure notice](https://www.nyse.com/publicdocs/nyse/markets/arca-options/rule-interpretations/2018/NYSE%20Arca%20Options%2018-06.pdf)
- [NYSE Juneteenth rule filing](https://www.nyse.com/publicdocs/nyse/markets/nyse/rule-filings/filings/2021/SR-NYSE-2021-56.pdf)

## Consequences

- Terminal cash groups no longer pay the cost of a dense general mesh solely because the event
  equals maturity.
- Every accepted schedule is reproducible without a runtime calendar dependency or network call.
- Provider forecast dates remain visible model inputs; calendar semantics never mutate them.
- Missing or contradictory calendar evidence stops the run before simulation.
- ADR 0037 receipts and hashes remain valid historical identities but are not reused by new
  publication replications.
