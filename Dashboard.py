import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import time
import os
import sys

from PCA_OU import *

def compute_drawdown_profile(returns: np.ndarray):
    """
    Computes cumulative returns and drawdown series geometrically.
    """
    clean_rets = np.where(np.isfinite(returns), returns, 0.0)
    equity_curve = np.cumprod(1.0 + clean_rets)
    cum_returns = equity_curve - 1.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / running_max) - 1.0
    return cum_returns, drawdowns

def benchmark_returns(
    result: dict,
    R: np.ndarray,
    mask: np.ndarray,
    eval_start_idx: int):

    gross_rets = result["gross_returns"]
    R_eval = R[eval_start_idx:]
    mask_eval = mask[eval_start_idx:]
    
    benchmark_rets = np.zeros(len(gross_rets))
    for t in range(len(gross_rets)):
        valid_assets = R_eval[t, mask_eval[t]]
        if len(valid_assets) > 0:
            benchmark_rets[t] = np.mean(valid_assets)

    return benchmark_rets
    

def generate_performance_dashboard(
    result: dict,
    net_returns: np.ndarray,
    benchmark_rets: np.ndarray,
    save_path: str = "strategy_performance_dashboard.png"
):
    dates = pd.to_datetime(result["dates_eval"])
    gross_rets = result["gross_returns"]
    active_counts = result["active_counts"]
    
    # 1. Compute Strategy Equity and Drawdown Curves
    gross_cum, gross_dd = compute_drawdown_profile(gross_rets)
    net_cum, net_dd = compute_drawdown_profile(net_returns)
    benchmark_cum, benchmark_dd = compute_drawdown_profile(benchmark_rets)

    # 3. Setup Grid Layout for Dashboard
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    # --- PANEL 1: Cumulative Returns ---
    axes[0].plot(dates, gross_cum * 100, label=f"Gross Portfolio (Final: {gross_cum[-1]*100:.1f}%)", color="navy", lw=1.5)
    axes[0].plot(dates, net_cum * 100, label=f"Net Portfolio (Final: {net_cum[-1]*100:.1f}%)", color="darkorange", lw=1.5)
    axes[0].plot(dates, benchmark_cum * 100, label=f"Equal-Weighted Benchmark (Final: {benchmark_cum[-1]*100:.1f}%)",
                 color="gray", linestyle="--", alpha=0.7, lw=1.2)
    
    axes[0].set_title("Cumulative Geometric Performance Comparison", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Cumulative Return (%)", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=10, frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.5)
    
    # --- PANEL 2: Drawdown Profile ("Underwater Chart") ---
    # Primary Left Y-Axis: Strategy Drawdowns
    axes[1].fill_between(dates, gross_dd * 100, 0, label=f"Gross MDD: {np.min(gross_dd)*100:.1f}%", color="navy", alpha=0.15)
    axes[1].plot(dates, gross_dd * 100, color="navy", lw=0.8, alpha=0.4)
    axes[1].fill_between(dates, net_dd * 100, 0, label=f"Net MDD: {np.min(net_dd)*100:.1f}%", color="darkorange", alpha=0.25)
    axes[1].plot(dates, net_dd * 100, color="darkorange", lw=0.8, alpha=0.6)
    
    axes[1].set_title("Portfolio & Benchmark Drawdown Profile (Underwater Chart)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Strategy Drawdown (%)", color="navy", fontsize=11)
    axes[1].tick_params(axis='y', labelcolor="navy")
    axes[1].set_ylim(None, 0)  # Ceiling of drawdown chart is 0%
    axes[1].grid(True, linestyle=":", alpha=0.5)
    
    # Secondary Right Y-Axis: Benchmark Drawdown
    ax2 = axes[1].twinx()
    ax2.plot(dates, benchmark_dd * 100, label=f"Benchmark MDD: {np.min(benchmark_dd)*100:.1f}%", color="grey", linestyle="--", lw=1.2, alpha=0.4)
    ax2.set_ylabel("Benchmark Drawdown (%)", color="grey", fontsize=11)
    ax2.tick_params(axis='y', labelcolor="grey")
    ax2.set_ylim(None, 0)
    
    # Combine handles/labels from primary axis and twin axis into a single legend box
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="lower left", fontsize=10, frameon=True)
    
    # --- PANEL 3: Strategy Dynamics (Active Allocations) ---
    axes[2].plot(dates, active_counts, color="darkgreen", lw=1.2, label="Active Positions")
    axes[2].set_title("Strategy Asset Engagement Over Time", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Number of Active Positions", fontsize=11)
    axes[2].set_xlabel("Timeline", fontsize=11)
    axes[2].legend(loc="upper left", fontsize=10, frameon=True)
    axes[2].grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()  # Safely displays inline below your Jupyter cell
    
    # Print a professional summary block
    print("=" * 50)
    print("         INSTITUTIONAL PERFORMANCE SUMMARY       ")
    print("=" * 50)
    print(f"Maximum Gross Drawdown:  {np.min(gross_dd)*100:.2f}%")
    print(f"Maximum Net Drawdown:    {np.min(net_dd)*100:.2f}%")
    print(f"Maximum Bench Drawdown:  {np.min(benchmark_dd)*100:.2f}%")
    print(f"Terminal Net Return:     {net_cum[-1]*100:.2f}%")
    print("=" * 50)
    
    return gross_dd, net_dd, benchmark_dd
