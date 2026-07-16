# Data contract

The empirical data are not distributed with this repository. The original option quotes and
dividend forecasts are licensed; derived row-level features and predictions remain subject to the
same restriction. Reproduction therefore requires licensed source data or an equivalent dataset
adapted to the canonical feature contract below.

## Canonical experiment table

Each row represents one American call quote on `quote_date` for one ticker, strike, and expiry.
Column order and physical Polars dtype are part of the artifact identity.

| Column | Polars dtype | Meaning |
|---|---|---|
| `ticker` | `String` | Underlying symbol |
| `quote_date` | `Date` | Point-in-time observation date |
| `expiry_date` | `Date` | Option expiration date |
| `strike` | `Float32` | Strike in currency units |
| `underlying_price` | `Float32` | Contemporaneous underlying price |
| `call_mid_price` | `Float32` | Call target, `(bid + ask) / 2` |
| `days_to_expiry` | `Int64` | Calendar days from quote to expiry |
| `yte` | `Float64` | `days_to_expiry / 365` |
| `call_moneyness` | `Float32` | `underlying_price / strike` |
| `forecasted_total_dividend_in_period` | `Float64` | Forecast cash through expiry |
| `forecasted_discrete_ex_dates` | `List(Date)` | Future ex-dividend dates through expiry |
| `forecasted_discrete_amounts` | `List(Float64)` | Cash amounts paired with the ex-dates |
| `forecasted_years_to_discrete_ex_dates` | `List(Float64)` | ACT/365 time from quote to each ex-date |
| `forecasted_dividend_count` | `UInt32` | Number of paired forecast events |
| `forecasted_dividend_yield` | `Float64` | Forecast-dividend scalar comparator |
| `historical_volatility_90` | `Float64` | Annualized 90-observation return volatility |
| `risk_free_rate` | `Float64` | Decimal continuously compounded zero rate at `yte` |
| `historical_total_dividend_in_lookback_window` | `Float64` | Trailing realized cash total |
| `historical_dividend_yield` | `Float64` | Trailing realized-dividend scalar comparator |

The three forecast-event lists must have equal logical length, match
`forecasted_dividend_count`, and be ordered by increasing ex-date. Every event must be known on the
quote date, occur after the quote date and at or before option expiry, and have finite nonnegative
cash. The paper models admit at most five events per option horizon. Padding is created by the model
input boundary and must not be stored as a real event.

All scalar prices and rates must be finite. Prices and strikes are strictly positive; the target is
nonnegative. Decimal rates, volatility, and yields use `0.05` for five percent. Missing values,
implicit casts, imputation, clipping, extrapolation, and backend-dependent preprocessing are not
accepted.

## Source information needed

An equivalent upstream dataset must supply:

1. Call quotes with ticker, quote date, expiry, strike, underlying price, bid, ask, volume, and
   option type.
2. Point-in-time dividend forecasts with forecast date, ticker, future ex-date, and cash amount.
3. Split-adjusted daily underlying history sufficient for the 90-observation volatility window.
4. Realized cash dividends with ticker and ex-date for the historical-yield feature.
5. A dated zero-rate curve from which the maturity-matched continuously compounded rate can be
   evaluated.

Provider-specific column names may differ. Adapt them before the canonical boundary; do not change
the downstream model to accommodate ambiguous units or dates.

## Original study universe

The frozen project configuration uses DUK, EPD, ETR, MO, SO, and VZ, with rolling test windows from
2015 through 2024. A new universe is a new study configuration and should receive a distinct
artifact identity.

## Local artifacts

Keep input Parquet files, receipts, predictions, checkpoints, and generated tables outside the Git
working tree. The supplied commands write content-addressed artifacts and verify their hashes before
use. Paths are deliberately excluded from portable result identity; file contents and declared
semantics are not.
