# ADR 0016: Explicit neural execution and canonical parameters

Status: accepted

The study has two independent neural realizations, PyTorch 2.13.0 and MLX 0.32.0, but no automatic
backend selection. Every run resolves an immutable profile naming the model variant, framework,
device, float32 precision, compilation mode, and CPU thread count where applicable. The declared
request is validated before model construction and fails if its exact package version or device is
unavailable. The repository's portable uv environment pins PyTorch to its explicit CPU-only index,
so the locked Linux workflow never acquires an unused CUDA stack. MLX remains a separate opt-in
extra and uses its explicit CPU distribution on Linux. The measured Apple-silicon profile uses
eager one-thread PyTorch CPU for the five-slot LSTM and event-attention models, and compiled MLX GPU
for the binned and two scalar-yield variants. A cold 100,000-row inference sweep selected an
explicit physical batch of 8,192 for every variant: against 128, throughput improved from roughly
328k to 779k rows/s for LSTM, 455k to 1.09m for event attention, and 0.30-0.38m to 2.19-2.28m for the
three MLX models in isolated processes. The portable profile explicitly requests eager one-thread PyTorch CPU and the same
bounded inference geometry for all variants. A CUDA profile is not admitted until it is measured on
the target cluster.

Framework initialization is not part of experiment identity. NumPy PCG64DXSM produces one
read-only, content-hashed float32 parameter set from an unsigned 64-bit seed, with explicit draw
order, shapes, and names. Both frameworks load that same set before use. The LSTM matrices use
input-forget-cell-output gate order; event features are ordered as option maturity, chronological
dividend time, and cash amount divided by spot. Sequence state is selected at the last real event,
never after padding. Masked attention assigns an exactly zero event representation to an empty
schedule, avoiding an undefined all-masked softmax.

AdamW learning rate, beta coefficients, epsilon, decoupled weight decay, and bias correction are
all study values. Neither framework may supply an optimizer default. Validation-plateau scheduling,
strict best-weight selection, and early stopping use one backend-neutral immutable state machine.
Every nonfinite loss, gradient, or trained parameter fails the run immediately.

Correctness is defined by the declared model, objective, shapes, units, masks, and financial-data
contracts—not by matching one framework's digits. Independent forward and parameter-gradient
graphs are checked from identical weights across every model, including empty and full schedules.
Float32 differences are admitted only within predeclared absolute numerical tolerances; nonfinite
predictions or gradients always fail. No mixed-precision or fallback path is approved.
