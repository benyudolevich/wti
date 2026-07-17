"""
Charts for the volatility-timing strategy (vol_strategy.py). One chart per
figure, matplotlib only, saved to charts/.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config

OUT_DIR = Path("charts")
OUT_DIR.mkdir(exist_ok=True)


def plot_ovx_full_history_with_trades(df, trades, path=OUT_DIR / "05_vol_ovx_full_history.png"):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df["date"], df["ovx"], linewidth=0.8, color="#4C72B0", alpha=0.8, label="OVX")
    ax.axvline(config.TRAIN_END, color="gray", linestyle=":", linewidth=1.2, label="Train/test split")

    for _, t in trades.iterrows():
        exit_date = t["exit_date"] if pd.notna(t["exit_date"]) else df["date"].max()
        ax.axvspan(t["entry_date"], exit_date, color="#C44E52", alpha=0.25)

    ax.axhline(60, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="OVX=60 (episode threshold)")
    ax.set_title("OVX, 2007–present, with LONG\\_VOL holding periods shaded")
    ax.set_ylabel("OVX")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_recent_signal_detail(df, trades, start="2024-09-01", path=OUT_DIR / "06_vol_signal_detail.png"):
    plot_df = df.loc[df["date"] >= start]

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    ax = axes[0]
    ax.plot(plot_df["date"], plot_df["ovx"], linewidth=1.4, color="#4C72B0", label="OVX (implied)", zorder=3)

    for _, t in trades.iterrows():
        exit_date = t["exit_date"] if pd.notna(t["exit_date"]) else df["date"].max()
        color = {"take_profit": "#55A868", "stop_loss": "#C44E52"}.get(t["exit_reason"], "#8172B2")
        ax.axvspan(t["entry_date"], exit_date, color=color, alpha=0.15)

        take_profit_level = t["entry_ovx"] * (1 + config.VOL_STRATEGY_TAKE_PROFIT_PCT)
        stop_loss_level = t["entry_ovx"] * (1 - config.VOL_STRATEGY_STOP_LOSS_PCT)
        ax.hlines(take_profit_level, t["entry_date"], exit_date, color="#55A868", linestyle="--", linewidth=1, alpha=0.7)
        ax.hlines(stop_loss_level, t["entry_date"], exit_date, color="#C44E52", linestyle="--", linewidth=1, alpha=0.7)

        ax.scatter([t["entry_date"]], [t["entry_ovx"]], marker="^", s=70, color="black", zorder=5)
        if pd.notna(t["exit_date"]):
            ax.scatter([t["exit_date"]], [t["exit_ovx"]], marker="x", s=70, color="black", zorder=5)

    ax.set_title("OVX with LONG\\_VOL entries (▲) / exits (×) and per-trade take-profit/stop-loss brackets\n"
                  "(green shading/dashes = take-profit exit, red = stop-loss exit)")
    ax.set_ylabel("OVX")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(plot_df["date"], plot_df["crisis_prob_fwd10"], linewidth=1.4, color="#55A868",
            label="crisis_prob_fwd10 (entry signal)")
    ax.axhline(config.VOL_STRATEGY_ENTRY_THRESHOLD, color="green", linestyle="--", linewidth=1,
               label=f"Entry threshold ({config.VOL_STRATEGY_ENTRY_THRESHOLD})")
    for _, t in trades.iterrows():
        exit_date = t["exit_date"] if pd.notna(t["exit_date"]) else df["date"].max()
        color = {"take_profit": "#55A868", "stop_loss": "#C44E52"}.get(t["exit_reason"], "#8172B2")
        ax.axvspan(t["entry_date"], exit_date, color=color, alpha=0.15)
    ax.set_title("Entry signal: crisis_prob_fwd10 (exit no longer depends on the forecast -- level-based bracket only)")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


PHASE_COLORS = {
    "rapidly_increasing": "#8B0000",
    "increasing": "#E67E22",
    "stable": "#B0B0B0",
    "decreasing": "#5DADE2",
    "rapidly_decreasing": "#1F618D",
    "": "white",
}


def plot_vol_momentum_phases(df, start="2024-09-01", path=OUT_DIR / "07_vol_momentum_phases.png"):
    plot_df = df.loc[df["date"] >= start].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(plot_df["date"], plot_df["ovx"], linewidth=1.4, color="black", zorder=3, label="OVX")

    # Shade each day by its momentum phase as a thin vertical band.
    for i in range(len(plot_df) - 1):
        phase = plot_df.at[i, "vol_phase"]
        ax.axvspan(plot_df.at[i, "date"], plot_df.at[i + 1, "date"],
                   color=PHASE_COLORS.get(phase, "white"), alpha=0.5, linewidth=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in
               ["#8B0000", "#E67E22", "#B0B0B0", "#5DADE2", "#1F618D"]]
    labels = ["rapidly_increasing", "increasing", "stable", "decreasing", "rapidly_decreasing"]
    ax.legend(handles + [ax.get_lines()[0]], labels + ["OVX"], loc="upper left", ncol=2)

    ax.set_title("OVX momentum phase (5-day %% change, training-quantile buckets)")
    ax.set_ylabel("OVX")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_joint_regime_states(df, state_desc, start="2024-09-01", path=OUT_DIR / "08_vol_joint_states.png"):
    plot_df = df.loc[df["date"] >= start].reset_index(drop=True)

    ordered_states = state_desc.sort_values("ovx_level")["state"].tolist()
    cmap = plt.cm.RdYlBu_r
    colors = {f"state_{s}": cmap(i / (len(ordered_states) - 1)) for i, s in enumerate(ordered_states)}

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(plot_df["date"], plot_df["ovx"], linewidth=1.4, color="black", zorder=3, label="OVX")

    for i in range(len(plot_df) - 1):
        state = plot_df.at[i, "joint_state"]
        ax.axvspan(plot_df.at[i, "date"], plot_df.at[i + 1, "date"],
                   color=colors.get(state, "white"), alpha=0.5, linewidth=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f"state_{s}"], alpha=0.5) for s in ordered_states]
    labels = [f"state {s} (OVX~{state_desc.loc[state_desc['state']==s,'ovx_level'].values[0]:.0f}, "
              f"mom={state_desc.loc[state_desc['state']==s,'momentum_5d_log_chg'].values[0]:+.3f})"
              for s in ordered_states]
    ax.legend(handles + [ax.get_lines()[0]], labels + ["OVX"], loc="upper left", fontsize=8, ncol=2)

    ax.set_title("OVX with learned joint level+momentum HMM states")
    ax.set_ylabel("OVX")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def generate_all(df, trades):
    return [
        plot_ovx_full_history_with_trades(df, trades),
        plot_recent_signal_detail(df, trades),
    ]


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np

    from load_data import load_daily_dataset
    from vol_regime import attach_vol_regime
    from vol_strategy import run_vol_backtest

    df = load_daily_dataset()
    df["log_ovx"] = np.log(df["ovx"])
    df, train_result, crisis_regime = attach_vol_regime(df)
    df, trades = run_vol_backtest(df)

    for p in generate_all(df, trades):
        print(f"Saved {p}")