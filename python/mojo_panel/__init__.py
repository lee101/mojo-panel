"""mojo-panel — panel (time x entity) feature kernels in Mojo, with a NumPy fallback.

    import numpy as np, mojo_panel as mp

    x = np.random.default_rng(0).normal(size=(4096, 512))   # [T, P]
    mp.rolling_std(x, 24, min_periods=8)                    # along time
    mp.xs_zscore(x)                                         # across entities
    mp.xs_rank(x, signed=True)
    mp.shift(x, 1)                                          # the leakage guard

Every function accepts a `[T, P]` float64 array, treats NaN as missing, and
returns a new array of the same shape. The Mojo kernels are used when the shared
library is present; otherwise an equivalent NumPy/pandas implementation runs, and
the test suite pins the two together.
"""

from ._api import (  # noqa: F401
    backend,
    gpu_available,
    ewma,
    pct_change,
    rolling_beta,
    rolling_mean,
    rolling_std,
    shift,
    xs_demean,
    xs_rank,
    xs_topk,
    xs_zscore,
)

__all__ = [
    "backend",
    "gpu_available",
    "ewma",
    "pct_change",
    "rolling_beta",
    "rolling_mean",
    "rolling_std",
    "shift",
    "xs_demean",
    "xs_rank",
    "xs_topk",
    "xs_zscore",
]
__version__ = "0.1.0"
