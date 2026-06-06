import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import time
import os
import sys

from Costs import *

# PCA Model

def find_eval_start(dates: pd.DatetimeIndex,
                    eval_start_date: str = "1998-01-01"):
    
    """Index of the first date >= eval_start_date."""
    eval_start = pd.Timestamp(eval_start_date)
    idx = int(np.searchsorted(dates, np.datetime64(eval_start)))
    
    return idx

def fit_pca(returns_window: np.ndarray, K: int):

    # Standardize: Otherwise PCA dominated by high-vol names.
    mu = returns_window.mean(axis=0, keepdims=True)
    sd = returns_window.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)  # avoid div-by-zero for dead stocks
    Z = (returns_window - mu) / sd

    # PCA via SVD of standardized matrix
    # Z = U * diag(s) * Vt: Rotation * Diagonal * Rotation, eigenvectors are Vt.T
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt.T
    K_eigenvectors = V[:, :K]
    factor_weights = Z @ K_eigenvectors

    # Alternatively could have gotten the covariance matrix (Z^T * Z)/window_days
    # With that you can get the same eigenvectors using eigendecomposition
    
    return factor_weights, K_eigenvectors

def factor_weight_returns_residuals(factor_weights,returns_window):

    # Want to see returns[i] = alpha[i] + factor_weights@beta[i] + eps[i]
    # So absorb alpha into factor_weights with a column of 1's, and then solve
    factor_weights_augmented = np.column_stack([np.ones(factor_weights.shape[0]), factor_weights])
    beta_full, _, _, _ = np.linalg.lstsq(factor_weights_augmented, returns_window, rcond=None)

    residuals = returns_window - factor_weights_augmented @ beta_full
    signal = np.cumsum(residuals, axis=0)

    # residuals are like dx(t), while signal is like x(t) for the Ornstein–Uhlenbeck process model we want to use

    return residuals, signal

def ou_fit_per_ticker(returns_window,signal,
                      kappa_min_annual = 252.0*(.693 / 10.0) # kappa = 252*ln(2)/Half life
                     ):
    # OU fit per asset
    n_tradable = returns_window.shape[1]
    z_scores = np.full(n_tradable, np.nan)
    active = np.zeros(n_tradable, dtype=bool)
    
    for i in range(n_tradable):

        # OU fitting
        kappa, m, sigma_eq, b = fit_ou_ar1(signal[:, i])

        if kappa < kappa_min_annual:
            continue  # mean-reversion too slow -> reject
        if not np.isfinite(sigma_eq) or sigma_eq <= 0:
            continue
        # z-score using the *latest* s value (the current dislocation)
        z_i = (signal[-1, i] - m) / sigma_eq
        if not np.isfinite(z_i):
            continue
        z_scores[i] = z_i
        active[i] = True

    return z_scores, active

def fit_ou_ar1(signal):
	
	# Discrete Ornstein-Uhlenbeck (OU) process as an autoregressive model of order 1
    # First want to fit a, b into x_{t+1} = a + b * x_t + epsilon
    x = signal[:-1]
    y = signal[1:]
    assert len(x) >= 2
    x_mean = x.mean()
    y_mean = y.mean()
    xc = x - x_mean
    yc = y - y_mean
    var_x = (xc * xc).sum()
    if var_x < 1e-12:
        return 0.0, 0.0, 1.0, 1.0
    b = (xc * yc).sum() / var_x
    a = y_mean - b * x_mean

    # Now need to understand x_{t+1} = a + b * x_t + epsilon
    # as dX_t = kappa*(m - X_t)dt + sigma*dW_t, given that $X_t$ is an OU process
    # Key is to apply Ito's Lemma to X_t * e^{\kappa t}
    # Result is $X_{t+\Delta t} = X_t e^{-\kappa \Delta t} + m(1-e^{-\kappa \Delta t}) + ...
    # \sigma \int_t^{t+\Delta t} e^{-\kappa(t + \Delta t -s)}

    # Need |b| < 1 for stationarity / valid OU fit
    if b <= 0.0 or b >= 1.0:
        return 0.0, 0.0, 1.0, b
    kappa = -np.log(b) * 252.0
    m = a / (1.0 - b)

    # Epsilon gives the Wiener Process term
    resid = y - (a + b * x)
    sigma_xi = resid.std()

    # Equilibrium volatility:
    # var(X_{t+1}) = 0 + var(b*X_t) + var(\epsilon)
    # (1-b^2)sigma_eq^2 = sigma_xi^2
    sigma_eq = sigma_xi / np.sqrt(1.0 - b * b)
    if sigma_eq < 1e-12:
        sigma_eq = 1e-12
        
    return kappa, m, sigma_eq, b

def positions_from_zscores(
    z_scores,
    active,
    prev_weights,
    z_open = 1.25,
    z_close_short = 0.75,
    z_close_long = -0.50,
    ):

    N = len(z_scores)
    weights_tradable = np.zeros(N)
    for i in range(N):
        if not active[i]:
            continue

        if prev_weights[i] > 0: # Implies the z_score was previously very negative
            if z_scores[i] > z_close_long: weights_tradable[i] = 0 # Close Long Position
            else: weights_tradable[i] = 1 # Hold Long Position
                
        elif prev_weights[i] < 0: # Implies z_score was previously very positive
            if z_scores[i] < z_close_short: weights_tradable[i] = 0 # Close Short position
            else: weights_tradable[i] = -1 # Hold short position

        else: # prev_weights[i] == 0:
            if z_scores[i] > z_open: weights_tradable[i] = -1 # Open short positions
            elif z_scores[i] < -z_open: weights_tradable[i] = 1 # Open long position
        
    return weights_tradable


def normalize_gross_one(weights_tradable):
    
    gross = np.sum(np.abs(weights_tradable))
    if gross < 1e-12:
        return weights_tradable
        
    return weights_tradable / gross

def run_backtest(
    returns: np.ndarray,
    mask: np.ndarray,
    dates: np.ndarray,
    eval_start_idx: int,
    K: int,
    pca_window: int = 252,
    rebalance_freq: int = 1 # Daily
    ):

    total_days, num_tickers = returns.shape
    trading_days = total_days - eval_start_idx
    weights_path = np.zeros((trading_days, num_tickers))
    gross_path = np.zeros(trading_days)
    active_counts = np.zeros(trading_days, dtype=int)
    
    prev_weights = np.zeros(num_tickers)
    
    for trading_day in range(trading_days):
    
        # trading_day index in dates
        t = eval_start_idx + trading_day
    
        # PCA window, 252 prior days, not including trading_day (slice logic doesn't include end)
        window_slice = slice(t - pca_window, t)
        mask_win = mask[window_slice, :]
        tradable = mask_win.all(axis=0) & mask[t, :]
        n_tradable = tradable.sum()
        
        # Initialize position weights for trading day (to be determined)
        weights_t = np.zeros(num_tickers)
    
        # Only determine weights if enough tickers to fit PCA given K components
        if n_tradable >= 2 * K:
            
            returns_window = returns[window_slice, tradable]
            factor_weights, K_eigenvectors = fit_pca(returns_window, K)
            residuals, signal = factor_weight_returns_residuals(factor_weights,returns_window)
            z_scores, active =  ou_fit_per_ticker(returns_window,signal)
            weights_tradable = positions_from_zscores(z_scores, active, prev_weights[tradable])
            weights_t[tradable] = weights_tradable
            weights_t = normalize_gross_one(weights_t)
    
        # Compute gross returns, start by ensuring finite returns
        returns_t = np.where(np.isfinite(returns[t, :]), returns[t, :], 0.0)
        # Then the assumption is that you buy at the start of the day, sell at the end.
        gross_path[trading_day] = float(weights_t @ returns_t)
    
        weights_path[trading_day] = weights_t
        active_counts[trading_day] = int(np.sum(np.abs(weights_t) > 1e-10))
        prev_weights = weights_t
    
        if trading_day % 252 == 0:
            print(f"  t={trading_day}/{trading_days}  active={active_counts[trading_day]}  "
                  f"gross_today={gross_path[trading_day]:+.4f}")

    return {
        "weights": weights_path,
        "gross_returns": gross_path,
        "active_counts": active_counts,
        "dates_eval": dates[eval_start_idx:] if dates is not None else None,
    }

def run_one_K(returns, mask, dates, eval_start_idx, K):

    result = run_backtest(
        returns=returns, mask=mask, dates=dates,
        eval_start_idx=eval_start_idx,
        K=K, pca_window=252,
    )
    
    net_returns, costs = apply_costs_to_path(result["weights"], result["gross_returns"])
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
    print(f"  Gross Sharpe: {row['gross_sharpe']:+.3f}   "
          f"Net Sharpe: {row['net_sharpe']:+.3f}   "
          f"Active: {row['mean_active']:.1f}   "
          f"Turnover: {row['mean_turnover']:.3f}")

    return row, result

def train_pca_model(
    returns, mask, dates,
    eval_start_date = "1998-01-01",
    K_grid = [1, 3, 5, 8, 15, 30],
    ):

    eval_start_idx = find_eval_start(dates, eval_start_date)

    print(f"Data date range:        {dates[0].astype('datetime64[D]')} -> {dates[-1].astype('datetime64[D]')}")
    print(f"Eval date range:        {dates[eval_start_idx].astype('datetime64[D]')} -> {dates[-1].astype('datetime64[D]')}")
    print(f"Eval length:       {len(dates) - eval_start_idx} trading days")
    print(f"Survivorship note: this universe is the *current* S&P 500. "
          f"Asset count in early years will be small.")

    # coverage report
    avail = mask.sum(axis=1)
    print(f"\nAsset availability:")
    print(f"  At 1998 start: {avail[eval_start_idx]} / {returns.shape[1]}")
    print(f"  At 2010-01:    {avail[find_eval_start(dates, '2010-01-01')]} / {returns.shape[1]}")
    print(f"  At 2020-01:    {avail[find_eval_start(dates, '2020-01-01')]} / {returns.shape[1]}")

    results = {}
    rows = []
    for K in K_grid:
        print(f"\n--- K = {K} ---")
        row, result = run_one_K(returns, mask, dates, eval_start_idx, K)
        rows.append(row)
        results[K] = result


    metrics = pd.DataFrame(rows)
    print(f"\n\n=== Model Summary ===")
    print(metrics.to_string(index=False))

    return metrics, results
