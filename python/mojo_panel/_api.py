"""Public API: dispatch to the Mojo kernels, fall back to NumPy.

The fallback is not a stub — it is the reference the kernels are tested against,
so a machine with no Mojo toolchain gets the same numbers at lower speed. That
property is worth more than it costs: it means `mojo_panel` can be a dependency
of something portable, and it makes every kernel bug a test failure rather than a
platform-specific mystery.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-panel.so")
GPU_LIB = os.path.join(ROOT, "dist", "libmojo-panel-gpu.so")
SRC = os.path.join(ROOT, "src", "capi.mojo")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mp_rolling_mean": ([I] * 7, I),
    "mp_rolling_std": ([I] * 7, I),
    "mp_pct_change": ([I] * 5, I),
    "mp_shift": ([I] * 5, I),
    "mp_ewma": ([I, I, I, I, F, I], I),
    "mp_xs_zscore": ([I, I, I, I, I, F, I], I),
    "mp_xs_demean": ([I] * 6, I),
    "mp_xs_rank": ([I] * 7, I),
    "mp_xs_rank_rows": ([I] * 8, I),
    "mp_rolling_beta": ([I, I, I, I, I, I, I, F, I], I),
}

_handle: ctypes.CDLL | None = None
_failed = False
_gpu_handle: ctypes.CDLL | None = None
_gpu_state: bool | None = None

_GPU_SIGNATURES = {
    "mp_gpu_available": ([], I),
    "mp_gpu_xs_rank": ([I] * 6, I),
    "mp_gpu_xs_zscore": ([I, I, I, I, I, F], I),
}

# Cross-sectional work below this many cells is faster on the CPU than the PCIe
# round trip. Rank is O(P^2) per row so it crosses over much earlier than the
# linear ops; both thresholds are measured in bench/bench.py.
GPU_MIN_RANK_WORK = 4_000_000     # rows * cols^2
GPU_MIN_LINEAR_CELLS = 8_000_000  # rows * cols
WORKERS = int(os.environ.get("MOJO_PANEL_WORKERS", str(min(16, (os.cpu_count() or 8) - 2))))
RANK_PARALLEL_WORK = 2_000_000  # rows * cols^2


def _build() -> str:
    if not os.path.exists(SRC):
        raise FileNotFoundError("no Mojo source tree")
    stale = not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC)
    if stale:
        proc = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT, capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode or not os.path.exists(LIB):
            raise RuntimeError((proc.stderr or proc.stdout)[-2000:])
    return LIB


def _lib() -> ctypes.CDLL | None:
    global _handle, _failed
    if _failed:
        return None
    if _handle is None:
        try:
            _handle = ctypes.CDLL(_build())
        except Exception:  # noqa: BLE001 — absence is normal, not an error
            _failed = True
            return None
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_handle, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _handle


def _gpu():
    """The GPU library handle, or None. Availability is re-checked, not assumed:
    another process holding the card makes the device context fail at call time."""
    global _gpu_handle, _gpu_state
    if _gpu_state is False:
        return None
    if _gpu_handle is None:
        if os.environ.get("MOJO_PANEL_DISABLE_GPU") or not os.path.exists(GPU_LIB):
            _gpu_state = False
            return None
        try:
            _gpu_handle = ctypes.CDLL(GPU_LIB)
        except OSError:
            _gpu_state = False
            return None
        for name, (argtypes, restype) in _GPU_SIGNATURES.items():
            fn = getattr(_gpu_handle, name)
            fn.argtypes = argtypes
            fn.restype = restype
    if _gpu_state is None:
        _gpu_state = bool(_gpu_handle.mp_gpu_available())
    return _gpu_handle if _gpu_state else None


def gpu_available() -> bool:
    return _gpu() is not None


def backend() -> str:
    """"mojo" when the kernels are in use, "numpy" when the fallback is."""
    if os.environ.get("MOJO_PANEL_FORCE_NUMPY"):
        return "numpy"
    return "mojo" if _lib() is not None else "numpy"


def _f64(x) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("panel arrays must be 2-D [T, P]")
    return a


def _addr(a: np.ndarray) -> int:
    if not a.flags.c_contiguous:
        raise ValueError("kernel buffers must be C-contiguous")
    return a.ctypes.data


def _native():
    return None if os.environ.get("MOJO_PANEL_FORCE_NUMPY") else _lib()


# --------------------------------------------------------------------------
# NumPy reference implementations
# --------------------------------------------------------------------------


def _ref_rolling(x, window, min_periods, std):
    import pandas as pd

    df = pd.DataFrame(x)
    r = df.rolling(window, min_periods=min_periods)
    return (r.std() if std else r.mean()).to_numpy(dtype=np.float64)


def _ref_pct_change(x, lag):
    out = np.full_like(x, np.nan)
    if lag < x.shape[0]:
        prev = x[:-lag]
        cur = x[lag:]
        with np.errstate(invalid="ignore", divide="ignore"):
            res = cur / prev - 1.0
        res[~np.isfinite(prev) | (prev == 0) | ~np.isfinite(cur)] = np.nan
        out[lag:] = res
    return out


def _ref_shift(x, lag):
    out = np.full_like(x, np.nan)
    if 0 <= lag < x.shape[0]:
        out[lag:] = x[: x.shape[0] - lag] if lag else x
    return out


def _ref_xs(x, valid, kind, clip=0.0, signed=False):
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        f = valid[t] & np.isfinite(x[t])
        n = int(f.sum())
        if n < 2 or (kind == "demean" and n < 1):
            if kind == "demean" and n >= 1:
                pass
            else:
                continue
        v = x[t, f]
        if kind == "zscore":
            sd = v.std()
            z = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
            out[t, f] = np.clip(z, -clip, clip) if clip > 0 else z
        elif kind == "demean":
            out[t, f] = v - v.mean()
        else:  # rank
            order = np.argsort(np.argsort(v, kind="stable"), kind="stable")
            pct = order / (n - 1)
            out[t, f] = pct * 2 - 1 if signed else pct
    return out


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def rolling_mean(x, window: int, min_periods: int = 1) -> np.ndarray:
    """Trailing mean over `window` rows, per entity, NaN-skipping."""
    a = _f64(x)
    lib = _native()
    if lib is None:
        return _ref_rolling(a, window, min_periods, std=False)
    o = np.empty_like(a)
    lib.mp_rolling_mean(_addr(a), _addr(o), a.shape[0], a.shape[1], window, min_periods, WORKERS)
    return o


def rolling_std(x, window: int, min_periods: int = 2) -> np.ndarray:
    """Trailing sample std (ddof=1), per entity, NaN-skipping."""
    a = _f64(x)
    lib = _native()
    if lib is None:
        return _ref_rolling(a, window, min_periods, std=True)
    o = np.empty_like(a)
    lib.mp_rolling_std(_addr(a), _addr(o), a.shape[0], a.shape[1], window, min_periods, WORKERS)
    return o


def pct_change(x, lag: int = 1) -> np.ndarray:
    a = _f64(x)
    lib = _native()
    if lib is None:
        return _ref_pct_change(a, lag)
    o = np.empty_like(a)
    lib.mp_pct_change(_addr(a), _addr(o), a.shape[0], a.shape[1], lag)
    return o


def shift(x, lag: int = 1) -> np.ndarray:
    """Lag the panel by `lag` rows — the leakage guard for a feature pipeline."""
    a = _f64(x)
    lib = _native()
    if lib is None:
        return _ref_shift(a, lag)
    o = np.empty_like(a)
    lib.mp_shift(_addr(a), _addr(o), a.shape[0], a.shape[1], lag)
    return o


def ewma(x, half_life: float, adjust: bool = True) -> np.ndarray:
    a = _f64(x)
    lib = _native()
    if lib is None:
        import pandas as pd

        return pd.DataFrame(a).ewm(halflife=half_life, adjust=adjust).mean().to_numpy(np.float64)
    o = np.empty_like(a)
    lib.mp_ewma(_addr(a), _addr(o), a.shape[0], a.shape[1], ctypes.c_double(half_life), int(adjust))
    return o


def _valid_of(x: np.ndarray, valid) -> np.ndarray:
    if valid is None:
        return np.ascontiguousarray(np.isfinite(x), dtype=np.int64)
    return np.ascontiguousarray(valid, dtype=np.int64)


def xs_zscore(x, valid=None, clip: float = 0.0) -> np.ndarray:
    """Per-row z across entities; `clip > 0` winsorises."""
    a = _f64(x)
    v = _valid_of(a, valid)
    o = np.empty_like(a)
    g = _gpu()
    if g is not None and a.size >= GPU_MIN_LINEAR_CELLS:
        if g.mp_gpu_xs_zscore(_addr(a), _addr(v), _addr(o), a.shape[0], a.shape[1],
                              ctypes.c_double(clip)) == 0:
            return o
    lib = _native()
    if lib is None:
        return _ref_xs(a, v.astype(bool), "zscore", clip=clip)
    lib.mp_xs_zscore(_addr(a), _addr(v), _addr(o), a.shape[0], a.shape[1], ctypes.c_double(clip), WORKERS)
    return o


def xs_demean(x, valid=None) -> np.ndarray:
    a = _f64(x)
    v = _valid_of(a, valid)
    lib = _native()
    if lib is None:
        return _ref_xs(a, v.astype(bool), "demean")
    o = np.empty_like(a)
    lib.mp_xs_demean(_addr(a), _addr(v), _addr(o), a.shape[0], a.shape[1], WORKERS)
    return o


def xs_rank(x, valid=None, signed: bool = False) -> np.ndarray:
    """Per-row percentile rank across entities; ties break by column index."""
    a = _f64(x)
    v = _valid_of(a, valid)
    o = np.empty_like(a)
    g = _gpu()
    if g is not None and a.shape[0] * a.shape[1] ** 2 >= GPU_MIN_RANK_WORK and a.shape[1] <= 2048:
        if g.mp_gpu_xs_rank(_addr(a), _addr(v), _addr(o), a.shape[0], a.shape[1], int(signed)) == 0:
            return o
    lib = _native()
    if lib is None:
        return _ref_xs(a, v.astype(bool), "rank", signed=signed)
    rows, cols = a.shape
    if WORKERS > 1 and rows * cols * cols >= RANK_PARALLEL_WORK:
        # Each bar is an independent P^2 sweep; splitting them across cores is
        # worth it only once there is real work per bar. Measured 6.7x-7.9x on
        # 16 workers at T=20k-50k, P=100-300.
        workers = min(WORKERS, rows)
        edges = [rows * i // workers for i in range(workers + 1)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(
                pool.map(
                    lambda band: lib.mp_xs_rank_rows(
                        _addr(a), _addr(v), _addr(o), rows, cols, int(signed),
                        band[0], band[1],
                    ),
                    zip(edges, edges[1:]),
                )
            )
    else:
        lib.mp_xs_rank(_addr(a), _addr(v), _addr(o), rows, cols, int(signed), WORKERS)
    return o


def xs_topk(x, k: int, valid=None) -> np.ndarray:
    """[T, k] indices of the k largest valid entries per row, descending; -1 pads."""
    a = _f64(x)
    v = _valid_of(a, valid)
    idx = np.full((a.shape[0], k), -1, dtype=np.int64)
    lib = _native()
    if lib is None:
        masked = np.where(v.astype(bool) & np.isfinite(a), a, -np.inf)
        order = np.argsort(-masked, axis=1, kind="stable")[:, :k]
        keep = np.take_along_axis(masked, order, axis=1) > -np.inf
        idx = np.where(keep, order, -1)
        return idx
    lib.mp_xs_topk(_addr(a), _addr(v), _addr(idx), a.shape[0], a.shape[1], k)
    return idx


def rolling_beta(y, market, window: int, min_periods: int = 2, clip: float = 0.0) -> np.ndarray:
    """Trailing beta of each entity against a shared `[T]` series."""
    a = _f64(y)
    m = np.ascontiguousarray(market, dtype=np.float64).ravel()
    lib = _native()
    if lib is None:
        import pandas as pd

        df = pd.DataFrame(a)
        ms = pd.Series(m)
        cov = df.rolling(window, min_periods=min_periods).cov(ms)
        var = ms.rolling(window, min_periods=min_periods).var()
        beta = cov.div(var, axis=0).to_numpy(np.float64)
        return np.clip(beta, -clip, clip) if clip > 0 else beta
    o = np.empty_like(a)
    lib.mp_rolling_beta(
        _addr(a), _addr(m), _addr(o), a.shape[0], a.shape[1], window, min_periods,
        ctypes.c_double(clip), WORKERS,
    )
    return o
