import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import time
import os
import sys

from Costs import *

# PCA Model
def fit_pca(R: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:

    # Standardize: Otherwise PCA dominated by high-vol names.
    mu = R.mean(axis=0, keepdims=True)
    sd = R.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)  # avoid div-by-zero for dead stocks
    Z = (R - mu) / sd

    # PCA via SVD of standardized matrix
    # Z = U @ diag(s) @ Vt; principal components are U @ diag(s); loadings are Vt.T
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt.T  # (N, N) or (N, min(T,N))
    V_k = V[:, :K]  # (N, K)
    F = Z @ V_k  # (T, K) — factor scores
    return F, V_k


def fit_ou_ar1(s: np.ndarray) -> tuple[float, float, float, float]:
	
	# Discrete Ornstein-Uhlenbeck (OU) process as an autoregressive model of order 1
    x = s[:-1]
    y = s[1:]
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 1.0, 1.0

    x_mean = x.mean()
    y_mean = y.mean()
    xc = x - x_mean
    yc = y - y_mean
    var_x = (xc * xc).sum()
    if var_x < 1e-12:
        return 0.0, 0.0, 1.0, 1.0

    b = (xc * yc).sum() / var_x
    a = y_mean - b * x_mean

    # Need |b| < 1 for stationarity / valid OU fit
    if b <= 0.0 or b >= 1.0:
        return 0.0, 0.0, 1.0, b

    kappa = -np.log(b) * 252.0
    m = a / (1.0 - b)
    resid = y - (a + b * x)
    sigma_xi = resid.std()
    sigma_eq = sigma_xi / np.sqrt(1.0 - b * b)
    if sigma_eq < 1e-12:
        sigma_eq = 1e-12
    return kappa, m, sigma_eq, b


def compute_signals(
    R_window: np.ndarray,
    K: int,
    kappa_min_annual: float = 252.0 / 22.0,  # half-life < ~22 days
) -> tuple[np.ndarray, np.ndarray]:

    T, N = R_window.shape
    z = np.full(N, np.nan)
    active = np.zeros(N, dtype=bool)

    # PCA on standardized returns
    F, _ = fit_pca(R_window, K)  # F: (T, K)

    # For each asset, regress R[:, i] on F to get residuals
    # We standardize R per asset for PCA but use raw R for the residual model,
    # following a "eigenportfolio" decomposition. The PCA factors F are
    # already standardized.
    # Regression with intercept: R_i = alpha_i + beta_i @ F + eps_i
    F_design = np.column_stack([np.ones(T), F])  # (T, K+1)
    # beta_full[:, i] = (F_design.T F_design)^-1 F_design.T R[:, i]
    # Vectorized solve:
    try:
        beta_full, _, _, _ = np.linalg.lstsq(F_design, R_window, rcond=None)
        # beta_full: (K+1, N) — row 0 is alpha, rows 1..K are factor loadings
    except np.linalg.LinAlgError:
        return z, active

    residuals = R_window - F_design @ beta_full  # (T, N)
    # Cumulative residual process — the "X-process" in A-L
    s = np.cumsum(residuals, axis=0)  # (T, N)

    # OU fit per asset
    for i in range(N):
        kappa, m, sigma_eq, b = fit_ou_ar1(s[:, i])
        if kappa < kappa_min_annual:
            continue  # mean-reversion too slow -> reject
        if not np.isfinite(sigma_eq) or sigma_eq <= 0:
            continue
        # z-score using the *latest* s value (the current dislocation)
        z_i = (s[-1, i] - m) / sigma_eq
        if not np.isfinite(z_i):
            continue
        z[i] = z_i
        active[i] = True

    return z, active


def positions_from_zscores(
    z: np.ndarray,
    active: np.ndarray,
    prev_w: np.ndarray,
    z_open: float = 1.25,
    z_close_short: float = 0.75,
    z_close_long: float = 0.50,
) -> np.ndarray:

    N = len(z)
    w = np.zeros(N)
    # Active universe
    for i in range(N):
        if not active[i]:
            # Force flat — exit any prior position
            w[i] = 0.0
            continue
        z_i = z[i]
        prev = prev_w[i]
        if z_i > z_open:
            w[i] = -1.0  # short
        elif z_i < -z_open:
            w[i] = +1.0  # long
        elif prev < 0 and z_i < z_close_short:
            w[i] = 0.0   # close short
        elif prev > 0 and z_i > -z_close_long:
            w[i] = 0.0   # close long
        else:
            # Hold — preserve sign of prior position
            w[i] = np.sign(prev)
    return w


def normalize_gross_one(w: np.ndarray) -> np.ndarray:
    gross = np.sum(np.abs(w))
    if gross < 1e-12:
        return w
    return w / gross


def run_backtest(
    returns: np.ndarray,        # (T, N) all-asset return matrix
    mask: np.ndarray,           # (T, N) True where asset is tradeable
    dates: np.ndarray,          # (T,) datetime64 — for logging only
    eval_start_idx: int,        # first index in `returns` we start trading
    K: int = 15,
    pca_window: int = 252,
    rebalance_freq: int = 1,    # 1 = daily
    verbose: bool = True,
) -> dict:

    T, N = returns.shape
    T_eval = T - eval_start_idx
    weights_path = np.zeros((T_eval, N))
    gross_path = np.zeros(T_eval)
    active_counts = np.zeros(T_eval, dtype=int)

    prev_w = np.zeros(N)

    for tau in range(T_eval):
        t = eval_start_idx + tau

        # Build the PCA window using only valid (non-NaN, masked-in) names
        window_slice = slice(t - pca_window, t)
        R_win_full = returns[window_slice, :]  # (pca_window, N)
        mask_win = mask[window_slice, :]       # (pca_window, N)
        # Asset is usable iff it's masked-in for the FULL window
        usable = mask_win.all(axis=0) & mask[t, :]  # also tradeable today
        n_usable = usable.sum()

        if n_usable < 2 * K:
            # Not enough names to even fit PCA. Force flat.
            w_t = np.zeros(N)
        else:
            R_win = R_win_full[:, usable]
            # Replace any residual NaN with 0 (shouldn't exist given mask check)
            R_win = np.where(np.isfinite(R_win), R_win, 0.0)

            z_sub, active_sub = compute_signals(R_win, K=K)
            # Embed back to full universe
            w_sub = positions_from_zscores(z_sub, active_sub, prev_w[usable])
            w_t = np.zeros(N)
            w_t[usable] = w_sub

        # Normalize to gross = 1
        w_t = normalize_gross_one(w_t)

        # Compute next-period gross return: portfolio return *over the next day*
        # weights w_t are set using info up to t-1 (via window ending at t-1),
        # and earn returns[t, :].
        r_today = np.where(np.isfinite(returns[t, :]), returns[t, :], 0.0)
        gross_path[tau] = float(w_t @ r_today)

        weights_path[tau] = w_t
        active_counts[tau] = int(np.sum(np.abs(w_t) > 1e-10))
        prev_w = w_t

        if verbose and tau % 252 == 0:
            print(f"  t={tau}/{T_eval}  active={active_counts[tau]}  "
                  f"gross_today={gross_path[tau]:+.4f}")

    return {
        "weights": weights_path,
        "gross_returns": gross_path,
        "active_counts": active_counts,
        "dates_eval": dates[eval_start_idx:] if dates is not None else None,
    }
    
def find_eval_start(dates: pd.DatetimeIndex, eval_start_date: str = "1998-01-01") -> int:
    """Index of the first date >= eval_start_date."""
    eval_start = pd.Timestamp(eval_start_date)
    idx = int(np.searchsorted(dates.values, np.datetime64(eval_start)))
    return idx


def run_one_K(R, mask, dates, eval_start_idx, K, verbose=True):
    if verbose:
        print(f"\n--- K = {K} ---")
    result = run_backtest(
        returns=R, mask=mask, dates=dates.values,
        eval_start_idx=eval_start_idx,
        K=K, pca_window=252, verbose=verbose,
    )
    net_returns, costs = apply_costs_to_path(
        result["weights"], result["gross_returns"]
    )
    result['net_returns'] = net_returns

    gross_m = annualized_metrics(result["gross_returns"])
    net_m   = annualized_metrics(net_returns)
    turnover = np.mean([
        np.sum(np.abs(result["weights"][t] - result["weights"][t-1]))
        for t in range(1, len(result["weights"]))
    ])

    row = {
        "K": K,
        "gross_sharpe": gross_m["sharpe"],
        "gross_mean":   gross_m["mean"],
        "gross_std":    gross_m["std"],
        "net_sharpe":   net_m["sharpe"],
        "net_mean":     net_m["mean"],
        "mean_active":  float(np.mean(result["active_counts"])),
        "mean_turnover": float(turnover),
    }
    if verbose:
        print(f"  Gross Sharpe: {row['gross_sharpe']:+.3f}   "
              f"Net Sharpe: {row['net_sharpe']:+.3f}   "
              f"Active: {row['mean_active']:.1f}   "
              f"Turnover: {row['mean_turnover']:.3f}")
    return row, result


def train_pca_model(
    data_path: str = "data_tensors.npz",
    eval_start_date: str = "1998-01-01",
    K_grid: list[int] = (1, 3, 5, 8, 15, 30),
):
    data = load_data(data_path)
    R, mask, dates = data["R"], data["mask"], data["dates"]

    print(f"Data shape:        R = {R.shape}, mask = {mask.shape}")
    print(f"Date range:        {dates[0].date()} -> {dates[-1].date()}")
    eval_start_idx = find_eval_start(dates, eval_start_date)
    print(f"Eval start:        {dates[eval_start_idx].date()}  "
          f"(idx={eval_start_idx})")
    print(f"Eval length:       {len(dates) - eval_start_idx} trading days")
    print(f"Survivorship note: this universe is the *current* S&P 500. "
          f"Asset count in early years will be small.")

    # coverage report
    avail = mask.sum(axis=1)
    print(f"\nAsset availability:")
    print(f"  At 1998 start: {avail[eval_start_idx]} / {R.shape[1]}")
    print(f"  At 2010-01:    {avail[find_eval_start(dates, '2010-01-01')]} / {R.shape[1]}")
    print(f"  At 2020-01:    {avail[find_eval_start(dates, '2020-01-01')]} / {R.shape[1]}")

    results = {}
    rows = []
    for K in K_grid:
        row, result = run_one_K(R, mask, dates, eval_start_idx, K, verbose=True)
        rows.append(row)
        results[K] = result


    metrics = pd.DataFrame(rows)
    print(f"\n\n=== Model Summary ===")
    print(metrics.to_string(index=False))

    return metrics, results
