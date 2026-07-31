"""Kernels vs pandas/NumPy on ragged panels.

Ragged is the point: a panel with no missing values agrees under almost any
implementation, and every real one has entities that list late, delist, or drop a
bar. The NaN rule (missing never participates in a statistic) is what these tests
actually pin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mojo_panel as mp
from mojo_panel import _api


def panel(T=600, P=37, seed=0, nan_frac=0.05, ragged=True):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, P))
    if nan_frac:
        x[rng.random((T, P)) < nan_frac] = np.nan
    if ragged:
        for j in range(P):  # late listings and early delistings
            x[: rng.integers(0, T // 4), j] = np.nan
            if rng.random() < 0.2:
                x[rng.integers(3 * T // 4, T) :, j] = np.nan
    return np.ascontiguousarray(x)


def test_backend_is_the_mojo_one():
    assert mp.backend() == "mojo", "built kernels should be in use for these tests"


@pytest.mark.parametrize("window,min_periods", [(5, 1), (24, 8), (72, 24)])
def test_rolling_mean_matches_pandas(window, min_periods):
    x = panel()
    got = mp.rolling_mean(x, window, min_periods)
    ref = pd.DataFrame(x).rolling(window, min_periods=min_periods).mean().to_numpy()
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("window,min_periods", [(5, 2), (24, 8), (72, 24)])
def test_rolling_std_matches_pandas(window, min_periods):
    x = panel(seed=1)
    got = mp.rolling_std(x, window, min_periods)
    ref = pd.DataFrame(x).rolling(window, min_periods=min_periods).std().to_numpy()
    assert np.allclose(got, ref, rtol=1e-8, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("lag", [1, 6, 24])
def test_pct_change_matches_reference(lag):
    x = np.abs(panel(seed=2)) + 0.5
    got = mp.pct_change(x, lag)
    ref = _api._ref_pct_change(x, lag)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_shift_is_exact_and_leakage_safe():
    x = panel(seed=3)
    got = mp.shift(x, 1)
    assert np.isnan(got[0]).all()
    assert np.array_equal(got[1:], x[:-1], equal_nan=True)


def test_xs_zscore_matches_reference():
    x = panel(seed=4)
    got = mp.xs_zscore(x, clip=4.0)
    ref = _api._ref_xs(x, np.isfinite(x), "zscore", clip=4.0)
    assert np.allclose(got, ref, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_xs_rank_ties_break_by_index():
    x = np.array([[1.0, 1.0, 2.0, np.nan]])
    got = mp.xs_rank(x)
    assert np.allclose(got[0, :3], [0.0, 0.5, 1.0])
    assert np.isnan(got[0, 3])


def test_xs_rank_matches_reference_signed():
    x = panel(seed=5)
    got = mp.xs_rank(x, signed=True)
    ref = _api._ref_xs(x, np.isfinite(x), "rank", signed=True)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_xs_demean_sums_to_zero():
    x = panel(seed=6)
    got = mp.xs_demean(x)
    for t in range(x.shape[0]):
        f = np.isfinite(x[t])
        if f.sum() >= 1:
            assert abs(np.nansum(got[t])) < 1e-9


def test_xs_topk_picks_the_leaders():
    x = panel(T=200, P=20, seed=7)
    idx = mp.xs_topk(x, 3)
    for t in range(x.shape[0]):
        f = np.isfinite(x[t])
        if f.sum() < 3:
            continue
        want = np.argsort(-np.where(f, x[t], -np.inf), kind="stable")[:3]
        assert list(idx[t]) == list(want)


def test_rolling_beta_matches_pandas():
    x = panel(T=500, P=16, seed=8, nan_frac=0.0, ragged=False)
    market = np.nanmean(x, axis=1)
    got = mp.rolling_beta(x, market, 72, 24)
    df = pd.DataFrame(x)
    ms = pd.Series(market)
    ref = df.rolling(72, min_periods=24).cov(ms).div(
        ms.rolling(72, min_periods=24).var(), axis=0
    ).to_numpy()
    # The kernel uses the population form of both cov and var, whose ratio equals
    # the sample form's; only the NaN edges are allowed to differ.
    both = np.isfinite(got) & np.isfinite(ref)
    assert both.mean() > 0.7
    assert np.allclose(got[both], ref[both], rtol=1e-6, atol=1e-8)


def test_numpy_fallback_agrees_with_the_kernels(monkeypatch):
    x = panel(seed=9)
    got = mp.rolling_std(x, 24, 8)
    rank = mp.xs_rank(x, signed=True)
    monkeypatch.setenv("MOJO_PANEL_FORCE_NUMPY", "1")
    assert mp.backend() == "numpy"
    assert np.allclose(mp.rolling_std(x, 24, 8), got, rtol=1e-8, atol=1e-9, equal_nan=True)
    assert np.allclose(mp.xs_rank(x, signed=True), rank, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_all_nan_row_yields_all_nan():
    x = panel(T=50, P=8, seed=10, nan_frac=0.0, ragged=False)
    x[7] = np.nan
    assert np.isnan(mp.xs_zscore(x)[7]).all()
    assert np.isnan(mp.xs_rank(x)[7]).all()
