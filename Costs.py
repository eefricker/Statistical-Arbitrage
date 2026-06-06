import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import time
import os
import sys

# Costs

def transaction_cost(
    w_t: np.ndarray,
    w_prev: np.ndarray,
    c_trade = 0.0005,
    c_short = 0.0001,
):

    turnover = np.sum(np.abs(w_t - w_prev))
    shorts = np.sum(np.maximum(-w_t, 0.0))
    return c_trade * turnover + c_short * shorts
 
 
def apply_costs_to_path(
    weights: np.ndarray,
    gross_returns: np.ndarray,
    c_trade = 0.0005,
    c_short = 0.0001,
):

    T, N = weights.shape
    costs = np.zeros(T)
    w_prev = np.zeros(N)
    for t in range(T):
        costs[t] = transaction_cost(weights[t], w_prev, c_trade, c_short)
        w_prev = weights[t]
    net_returns = gross_returns - costs
    return net_returns, costs
 
 
def annualized_metrics(returns: np.ndarray, periods_per_year: int = 252):
    
    mu = np.nanmean(returns) * periods_per_year
    sigma = np.nanstd(returns) * np.sqrt(periods_per_year)
    sharpe = mu / sigma if sigma > 0 else 0.0
    return {
        "sharpe": sharpe,
        "mean": mu,
        "std": sigma,
        "total_return": np.nansum(returns),
        "n_obs": int(np.sum(np.isfinite(returns))),
    }
