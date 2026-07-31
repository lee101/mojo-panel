# mojo-panel

Panel feature kernels in Mojo — rolling statistics along time and ranking across
the cross-section — for the `[T, P]` matrices (T time steps by P entities) that
cross-sectional models actually consume.

```python
import numpy as np, mojo_panel as mp

x = np.random.default_rng(0).normal(size=(65536, 512))   # [time, entity]

mp.rolling_std(x, 24, min_periods=8)    # along time, per entity
mp.rolling_beta(x, market, 72, 24)      # trailing beta vs a shared series
mp.pct_change(x, 6)
mp.shift(x, 1)                          # the leakage guard

mp.xs_zscore(x, clip=4.0)               # across entities, per time step
mp.xs_rank(x, signed=True)
mp.xs_demean(x)
mp.xs_topk(x, 3)
```

Every function takes a 2-D float64 array, treats **NaN as missing** (missing never
participates in a statistic), and returns a new array of the same shape. That one
rule is what makes the output match pandas on *ragged* panels — entities that list
late, delist, or drop a bar — which is where naive ports quietly disagree.

## Speed

pandas is the baseline because it is what this replaces. Best-of-3 wall clock,
36-core Xeon, `bench/bench.py`:

| op | 4096x176 | 16384x512 | 65536x512 |
|---|---|---|---|
| `rolling_mean(72)` | **18.6x** | 4.3x | 3.7x |
| `rolling_std(24)` | **7.7x** | 3.7x | 3.3x |
| `pct_change(6)` | 11.8x | 2.8x | 3.1x |
| `xs_zscore` | 3.3x | 8.2x | **8.9x** |
| `xs_rank` | 1.2x | 1.9x | 2.3x |

The rolling kernels earned most of that from a layout change, not from SIMD. The
obvious implementation walks one entity down the time axis and reads
`x[t * P + col]` — a fresh cache line per row, 33M cache lines for one column of a
65536x512 panel. Sweeping **row by row** with `count/sum/sumsq` vectors of length P
reads every input once in order and lets the update vectorise across entities:
measured 1.3x -> 4.3x on `rolling_mean` at the largest size. Worth remembering
before reaching for intrinsics.

## GPU

`src/gpu.mojo` builds a second library covering the cross-sectional ops, where the
work is `O(T * P^2)` for rank with no dependence between rows — thousands of
independent small reductions, which is the shape a GPU wants. One block per bar
stages the row in shared memory, so the `P^2` comparisons read shared rather than
global, and z-score does a tree reduction instead of an atomic storm.

The rolling ops are deliberately **not** on the GPU: a running-sum recurrence along
time loses to the CPU's cache-resident single pass plus the PCIe round trip at any
realistic panel size. A kernel that exists and is slower is worse than no kernel.

Availability is re-checked at call time, never assumed — another process holding
the card makes the device context fail with OOM, and the CPU kernel then runs.
`MOJO_PANEL_DISABLE_GPU=1` forces that path.

## NumPy fallback

Without the Mojo toolchain every function falls back to an equivalent NumPy/pandas
implementation. It is not a stub: it is the reference the kernels are tested
against, so a fallback run is slower and **not different**.
`mp.backend()` reports which is live; `MOJO_PANEL_FORCE_NUMPY=1` forces it.

## Install

```bash
pixi install
pixi run build     # dist/libmojo-panel.so (+ -gpu.so when MAX is present)
pixi run test
pixi run bench
```

Requires the pinned Mojo nightly from `conda.modular.com/max-nightly`; the pin
lives in `pixi.toml`.

## Notes for porters

Four Mojo 1.0 facts this port ran into, none of which are obvious from an error
message:

- `a == b` on two SIMD vectors returns a **`Bool`**, not an elementwise mask. Use
  `a.eq(b)` / `.lt` / `.ge` — the comparison operators reduce.
- `fn` is a reserved word, so it cannot be a variable name.
- a tuple return type must be written `Tuple[Int, Float64]`, not `(Int, Float64)`.
- splatting a scalar into a SIMD constructor needs the scalar form; the compiler
  points at `fill=` but the working spelling is `SIMD[dtype, W](value)`.

## Licence

MIT.
