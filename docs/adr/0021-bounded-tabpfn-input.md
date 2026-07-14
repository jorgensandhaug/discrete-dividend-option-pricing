# ADR 0021: Bounded TabPFN cash-bin input

Status: accepted for input preparation only

A future local TabPFN benchmark receives exactly the reconstructed project's six ordered scalar
features followed by its twelve chronological, spot-scaled cash-dividend bins. This creates one
explicit comparison with the submitted binned network without inventing a flat encoding for the
sequence models. It is a supervised market-price benchmark, not a financial pricing kernel or an
arbitrage-free valuation method.

The full-horizon boundary allocates one C-contiguous float32 feature matrix and retains the already
validated positive market target by reference. Rolling phases are immutable zero-copy row views.
The exact feature names, parent prepared-batch hash, dtype, shape, and bytes define content identity.
No normalization, categorical inference, imputation, target transformation, clipping, or provider
data is introduced at this boundary.

This decision does not admit a TabPFN package, checkpoint, execution profile, or paper result. Those
require separate license review, an explicit existing checkpoint and digest, an offline runtime
with no automatic device/model selection or download, and measured local/cluster validation.
