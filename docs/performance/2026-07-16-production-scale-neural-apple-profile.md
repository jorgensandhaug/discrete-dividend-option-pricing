# Production-scale Apple neural execution profile — 2026-07-16

## Contract

- Host: Apple M4 Pro, 14 logical CPU cores, 48 GB unified memory.
- Runtime: locked CPython 3.14, PyTorch 2.13.0 and MLX 0.32.0.
- Inputs: deterministic target-free synthetic tensors with the exact project model shapes,
  dividend masks, objective, optimizer and fixed-batch training control.
- A direct scan of the receipt-verified study gives 117 windows, 161,731--240,524 training rows
  (median 198,653), 2,766--9,467 validation rows (median 5,679), and 2,766--9,467 test rows (median
  5,722).
- The representative training comparison uses 200,000 training rows, 6,000 validation rows and ten
  complete epochs. Each one-thread eager PyTorch CPU or compiled MLX GPU run is a cold isolated
  process.
- The representative inference comparison uses 6,000 rows and an explicit physical batch size of
  8,192.
- These are engineering measurements. They do not inspect option-price observations and are not
  paper results.

## Representative-window result

| Variant | PyTorch training | MLX training | Selected | PyTorch peak RSS | MLX peak RSS |
|---|---:|---:|---|---:|---:|
| Direct five-event LSTM | 29.774 s | 30.794 s | PyTorch CPU | 418,185,216 B | 343,064,576 B |
| Masked event attention | 29.192 s | 21.343 s | MLX GPU | 429,867,008 B | 337,281,024 B |
| Twelve cash bins | 20.640 s | 15.450 s | MLX GPU | 421,593,088 B | 335,921,152 B |
| Scalar forecast yield | 20.462 s | 14.669 s | MLX GPU | 397,492,224 B | 315,539,456 B |

| Variant | PyTorch inference | MLX inference | Selected end-to-end profile |
|---|---:|---:|---|
| Direct five-event LSTM | 0.0141 s | 0.0174 s | PyTorch CPU |
| Masked event attention | 0.0102 s | 0.0158 s | MLX GPU |

Direct-LSTM PyTorch is 3.3% faster for the representative training workload. MLX uses 18.0% less
peak memory, but memory is not binding at either 343 MB or 418 MB, so the speed-first profile keeps
PyTorch. For event attention, MLX reduces training wall time by 26.9% and peak memory by 21.5%.
PyTorch is about 5.6 ms faster on the small final inference phase, which is immaterial against the
7.85-second training gain. MLX reduces binned-cash training time by 25.1% and memory by 20.3%; for
the scalar-yield graph the reductions are 28.3% and 20.6%. Forecast and historical yield have the
same tensor graph and dimensions, differing only in the selected scalar input column, so the one
measured comparison owns both execution choices.

## Scale diagnostics

The crossover depends on row count, which is why backend selection cannot be inferred from a toy
batch. At 100,000 training rows and ten epochs, direct-LSTM PyTorch was 6.4% faster. At one million
training rows and three epochs, MLX was 24.6% faster for LSTM and 19.1% faster for event attention,
with 8.5% lower peak memory. On a separate million-row inference pass, MLX was 1.32x and 5.87x
faster, respectively. The representative project-window geometry—not either endpoint—owns the
production decision.

The initial 100,000/10,000-row, ten-epoch sweep also completed all three non-sequence MLX variants
in 7.02--8.71 seconds with 296--306 MB peak RSS. The representative-window CPU comparisons confirm
that retaining MLX is faster as well as smaller.

## Decision

`project_apple_silicon_m4_pro_v4` explicitly assigns direct LSTM to one-thread eager PyTorch CPU and
the other four variants to compiled MLX GPU, all with an 8,192-row inference batch. This is a new
immutable execution identity. It does not mutate or alias `v3`, infer a backend from the host, or
fall back if the requested runtime is unavailable. `project_portable_pytorch_cpu_1_v2` remains the
explicit portable path.

The benchmark entry points now run as importable modules and accept `profile`, `torch_cpu`, or
`mlx_gpu` explicitly, so every retained comparison can be repeated through `just bench-neural` and
`just bench-inference` without editing production configuration.
