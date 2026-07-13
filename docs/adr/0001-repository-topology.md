# ADR 0001: Independent sibling engine and paper repositories

Status: accepted

The generic native engine and this study use independent Git histories and lockfiles. During local
development, this repository consumes an editable sibling checkout without package publication.
CI and immutable run manifests use an exact engine commit from `config/engine-revision.txt`.

The separation keeps vendor adapters, licensed-data logic, study selection, neural models, and
publication outputs out of the financial engine. Cross-repository changes are integrated by updating
the pinned revision after both repositories pass their own gates.

