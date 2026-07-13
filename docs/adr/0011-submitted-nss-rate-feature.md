# ADR 0011: Project as-submitted NSS rate feature

Status: accepted for the project as-submitted reconstruction

The reconstruction accepts one exact-date `Float64` parameter row containing the six
Nelson-Siegel-Svensson parameters plus the shortest observed maturity and rate. Missing dates,
duplicates, non-finite values, nonpositive decay scales, and precision changes fail before
evaluation; parameters are never carried forward.

At or below the shortest maturity, the project identity returns the stored shortest rate. Longer
maturities use the submitted NSS rate expression. The implementation evaluates the same NSS
loadings with a small-argument series near zero to avoid cancellation; this is a numerical form of
the same equation, not a different curve model. The resulting `risk_free_rate` remains a scalar
study feature. Engine log-discount curves retain their separate source and adapter identity.
