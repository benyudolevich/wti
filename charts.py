"""
Chart functions for the WTI inventory-squeeze model. One chart per function,
matplotlib only, saved to the `charts/` directory.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from patsy import dmatrix

import config

OUT_DIR = Path("charts")
OUT_DIR.mkdir(exist_ok=True)


def plot_fair_value_curve(df, fv_model, path=OUT_DIR / "01_fair_value_curve.png"):
    """Scatter of spread vs. each fair-value input, fitted curve overlaid,
    holding the other input at its training median."""
    train = df.loc[df["date"] <= config.TRAIN_END]
    test = df.loc[df["date"] > config.TRAIN_END]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    u_grid = np.linspace(train["utilization"].quantile(0.005), train["utilization"].quantile(0.995), 300)
    grid_df = df.iloc[:0].reindex(range(len(u_grid))).copy()
    grid_df["utilization"] = u_grid
    grid_df["log_gri"] = train["log_gri"].median()
    grid_df["spread"] = 0  # placeholder, required column but unused for prediction
    fv_grid = fv_model.predict(grid_df)

    ax = axes[0]
    ax.scatter(train["utilization"], train["spread"], alpha=0.3, s=14, label="Training", color="#4C72B0")
    ax.scatter(test["utilization"], test["spread"], alpha=0.6, s=18, label="Test", color="#C44E52")
    ax.plot(u_grid, fv_grid, linewidth=2.5, color="black", label="Fitted (log(GRI) at train median)")
    ax.set_title("Spread vs utilization")
    ax.set_xlabel("Utilization")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.axhline(0, color="gray", linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    gri_grid = np.linspace(train["log_gri"].quantile(0.005), train["log_gri"].quantile(0.995), 300)
    grid_df2 = df.iloc[:0].reindex(range(len(gri_grid))).copy()
    grid_df2["log_gri"] = gri_grid
    grid_df2["utilization"] = train["utilization"].median()
    grid_df2["spread"] = 0
    fv_grid2 = fv_model.predict(grid_df2)

    ax = axes[1]
    ax.scatter(train["log_gri"], train["spread"], alpha=0.3, s=14, label="Training", color="#4C72B0")
    ax.scatter(test["log_gri"], test["spread"], alpha=0.6, s=18, label="Test", color="#C44E52")
    ax.plot(gri_grid, fv_grid2, linewidth=2.5, color="black", label="Fitted (utilization at train median)")
    ax.set_title("Spread vs log(GRI)")
    ax.set_xlabel("log(GRI)")
    ax.axhline(0, color="gray", linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_actual_vs_fair_value(df, start=None, path=OUT_DIR / "02_actual_vs_fair_value.png"):
    plot_df = df if start is None else df.loc[df["date"] >= start]

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(plot_df["date"], plot_df["spread"], linewidth=1.6, color="black", label="Actual spread")
    ax.plot(plot_df["date"], plot_df["fair_value"], linewidth=1.6, color="#C44E52", label="Fair value")

    if "action" in plot_df.columns:
        entries = plot_df[plot_df["action"].isin(["ENTER LONG", "ENTER SHORT"])]
        exits = plot_df[plot_df["action"].isin(["EXIT LONG", "EXIT SHORT"])]
        ax.scatter(entries["date"], entries["spread"], marker="^", s=70, color="green", zorder=3, label="Entry")
        ax.scatter(exits["date"], exits["spread"], marker="x", s=70, color="red", zorder=3, label="Exit")

    ax.axvline(config.TRAIN_END, color="gray", linestyle=":", linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Actual spread vs fair value")
    ax.set_ylabel("CL1-CL2 spread, $/bbl")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_zscore_with_bands(df, start=None, path=OUT_DIR / "03_zscore_bands.png"):
    plot_df = df if start is None else df.loc[df["date"] >= start]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(plot_df["date"], plot_df["zscore"], linewidth=1.2, color="#4C72B0")
    ax.axhline(config.ENTRY_Z, color="red", linestyle="--", linewidth=1, label=f"Entry threshold (+/-{config.ENTRY_Z})")
    ax.axhline(-config.ENTRY_Z, color="red", linestyle="--", linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(config.TRAIN_END, color="gray", linestyle=":", linewidth=1.2)
    ax.set_title("Residual z-score with entry thresholds")
    ax.set_ylabel("z-score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_pnl_and_drawdown(df, path=OUT_DIR / "04_pnl_drawdown.png"):
    oos = df.loc[df["date"] > config.TRAIN_END]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    ax = axes[0]
    ax.plot(oos["date"], oos["cumulative_pnl"], linewidth=1.6, color="#4C72B0")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Cumulative P&L (out-of-sample)")
    ax.set_ylabel("$")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(oos["date"], oos["drawdown"], 0, color="#C44E52", alpha=0.5)
    ax.set_title("Drawdown")
    ax.set_ylabel("$")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def generate_all(df, fv_model):
    paths = [
        plot_fair_value_curve(df, fv_model),
        plot_actual_vs_fair_value(df, start=config.TRAIN_END - pd_offset()),
        plot_zscore_with_bands(df, start=config.TRAIN_END),
        plot_pnl_and_drawdown(df),
    ]
    return paths


def pd_offset():
    import pandas as pd
    return pd.Timedelta(days=180)


if __name__ == "__main__":
    from load_data import load_daily_dataset
    from model import apply_fair_value_model
    from strategy import run_backtest

    df = load_daily_dataset()
    df, fv_model = apply_fair_value_model(df)
    df, trades = run_backtest(df)

    for p in generate_all(df, fv_model):
        print(f"Saved {p}")
