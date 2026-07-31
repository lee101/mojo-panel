#!/usr/bin/env python3
"""mojo-panel vs pandas on a realistic panel. Markdown table.

pandas is the reference implementation people actually replace, so it is the
baseline. Timings are best-of-3 wall clock on a warm cache.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import mojo_panel as mp


def timeit(fn, reps=3):
    fn()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    print(f"backend: {mp.backend()}   gpu: {mp.gpu_available()}")
    print("| op | shape | pandas/numpy | mojo-panel | speedup |")
    print("|---|---|---|---|---|")
    for T, P in ((4_096, 176), (16_384, 512), (65_536, 512)):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(T, P))
        x[rng.random((T, P)) < 0.05] = np.nan
        x = np.ascontiguousarray(x)
        df = pd.DataFrame(x)

        cases = [
            ("rolling_std(24)",
             lambda: df.rolling(24, min_periods=8).std().to_numpy(),
             lambda: mp.rolling_std(x, 24, 8)),
            ("rolling_mean(72)",
             lambda: df.rolling(72, min_periods=24).mean().to_numpy(),
             lambda: mp.rolling_mean(x, 72, 24)),
            ("pct_change(6)",
             lambda: df.pct_change(6).to_numpy(),
             lambda: mp.pct_change(x, 6)),
            ("xs_zscore",
             lambda: ((df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1), axis=0)).to_numpy(),
             lambda: mp.xs_zscore(x)),
            ("xs_rank",
             lambda: df.rank(axis=1, pct=True, method="first").to_numpy(),
             lambda: mp.xs_rank(x)),
        ]
        for name, ref, got in cases:
            a = timeit(ref)
            b = timeit(got)
            print(f"| {name} | {T}x{P} | {a * 1e3:.1f} ms | {b * 1e3:.1f} ms | {a / b:.1f}x |")
    print()
    print("Correctness is asserted in tests/, not here; a fast wrong answer is not a result.")


if __name__ == "__main__":
    main()
