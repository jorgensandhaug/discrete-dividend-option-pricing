# ADR 0050: Predeclared date-panel empirical inference

- Status: accepted before inspecting the selected-identity publication result
- Date: 2026-07-16
- Result-producing paper revision: `6507dbb62f78fd3227d8318c93f21548cbf11a27`
- Engine revision: `cd14031a8b1cd03b8258c73c739155a4161f88bc`

## Context

The submitted project compared row-weighted MAPE and RMSE point estimates. Option rows are not
independent: many strikes share an underlying and quote date, and adjacent dates are serially
dependent. Treating 678,122 option rows as independent observations would therefore produce
unjustifiably narrow uncertainty. Six ticker clusters are also too few for a comfortable
cluster-asymptotic argument.

Numerical Monte Carlo uncertainty is already separated by ADRs 0048--0049. Empirical inference must
instead operate on the completed replication-mean cash predictions and deterministic yield
predictions. Its population, loss functions, contrasts, resampling scheme, random generator,
confidence level, block lengths, and multiplicity family must be fixed before inspecting the
selected-identity result.

One licensed-data limitation is known in advance: ETR option coverage stops on 2019-11-07, while
the other five tickers continue through 2024-09-27. No missing ETR rows can be imputed or silently
reweighted.

## Decision

Freeze `PROJECT_PUBLICATION_EMPIRICAL_INFERENCE_SPEC` with three separately named populations:

1. the primary available-case six-ticker full horizon;
2. the five non-ETR tickers over their full horizon; and
3. all six tickers through the last available ETR date, 2019-11-07.

The latter two are mandatory cohort sensitivities, not alternative populations from which a
favorable conclusion may be selected.

The primary metrics are the submitted row-weighted MAPE and RMSE, in that order. MAE, signed bias,
relative signed bias, and SMAPE are required diagnostics. For every cohort, all six canonical
left-minus-right pairs among the two cash and two yield variants are retained. Lower is better only
for the absolute and squared-loss metrics; signed bias diagnostics are not converted into rankings.

Empirical uncertainty uses the circular stationary bootstrap of complete quote-date panels in
their ordered observed-date sequence. Every time the process restarts, its next block origin is
uniform over observed panels; continuation is geometric with restart probability `1/L`, where `L`
is the declared mean block length. A resampled metric is a ratio over all option rows belonging to
the resampled panels, so dates remain the resampling unit while the original row-weighted estimand
is preserved. Gaps in provider coverage are not filled with invented panels.

The primary mean block length is 20 observed quote-date panels, approximately one trading month.
Lengths 5, 10, 20, 40, and 60 are all mandatory sensitivity outputs. The primary length is not
estimated from realized model losses. Each configuration uses 19,999 replications from
Philox4x32-10 with seed `0x13198A2E03707344`. Counter domains begin at 2,000,000 and are assigned by
`domain_start + cohort_index * block_count + block_index`, so cohort/block configurations never
share a resampling stream. Zero-based bootstrap replication `b` occupies the Philox path lane `b`;
the remaining counter lanes are the low and high words of the sequential Philox block counter and
the declared domain. Two consecutive Random123 output words are packed low-word first into each
`u64` draw.

Circular origins use Lemire's unsigned 64-bit multiply-high mapping with rejection, not a biased
modulo. Geometric lengths use an explicitly digital inverse CDF: map the top 53 bits of a `u64`
word to the open-unit value `(bits + 1) / (2^53 + 1)`, then compute
`1 + floor(log1p(-u) / log1p(-1/L))` with the Rust `libm` implementation. `L = 1` is the exact
one-panel case. The last block is truncated so every bootstrap draw contains exactly as many panel
positions as the observed cohort. These mapping details are part of the hashed specification;
thread scheduling cannot alter a replication stream.

Two-sided confidence level is 0.95. Intervals use the centered basic-bootstrap construction with
Hyndman--Fan type-7 linear sample quantiles. Two-sided p-values use the centered bootstrap
distribution and the finite-sample `(extreme + 1) / (B + 1)` correction. Holm step-down adjustment
controls family-wise error at 0.05 across the single primary family of six model pairs times two
primary metrics. Diagnostic-metric and block/cohort-sensitivity outputs are reported as such and do
not create post-hoc primary claims.

## Consequences

- Empirical and numerical uncertainty cannot be confused: the former resamples observed date
  panels; the latter repeats the stochastic pricing estimator conditional on fixed data.
- Same-date cross-sectional dependence and short-range serial dependence are retained without
  claiming that six ticker clusters support reliable cluster asymptotics.
- The incomplete ETR horizon is visible through two mandatory, separately identified sensitivity
  populations rather than imputation.
- A different block length, bootstrap generator, quantile convention, confidence level, cohort,
  metric, contrast family, or multiplicity rule is a new hashed inference specification.
- Implementation may be optimized in Rust, but it must reproduce this exact resampling and
  estimand identity or use a new specification.

## References

- Politis, D. N., and Romano, J. P. (1994), “The Stationary Bootstrap,” *Journal of the American
  Statistical Association*, 89(428), 1303–1313.
- Holm, S. (1979), “A Simple Sequentially Rejective Multiple Test Procedure,” *Scandinavian Journal
  of Statistics*, 6(2), 65–70.
- Hyndman, R. J., and Fan, Y. (1996), “Sample Quantiles in Statistical Packages,” *The American
  Statistician*, 50(4), 361–365.
