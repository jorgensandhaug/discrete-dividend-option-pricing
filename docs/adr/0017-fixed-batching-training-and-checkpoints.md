# ADR 0017: Fixed batching, training control, and safe checkpoints

Status: accepted

Every epoch has deterministic fixed physical shapes without dropping observations. A PCG64DXSM
permutation covers each training row exactly once; the partial final batch uses index-zero filler
slots plus an explicit validity mask, so filler values contribute neither observations nor loss.
Validation order is fixed. This contract supports compiled execution without changing the logical
sample and fails if coverage, padding, shape, dtype, or mask invariants do not hold.

Training requests bind distinct training and validation phase hashes, canonical initial parameters,
the shuffle seed, and an exact execution request. Backend implementations share the same AdamW,
sMAPE, scheduling, strict best-weight, and early-stopping semantics. The returned network always
contains the best finite validation weights, copied into one canonical backend-neutral parameter
namespace.

Checkpoints use safe-tensors only. Each artifact contains finite float32 tensors, canonical metric
history, and a manifest binding study, data, seeds, runtime, best epoch, and both initial and trained
parameter hashes. The complete file SHA-256 is its filename; writes are atomic and idempotent, and
loads verify the filename, metadata schema, history, tensor hash, and parameter count before use.
Production code never searches for a mutable “latest” checkpoint and never deserializes pickle.
