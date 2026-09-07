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


def plot_joint_regime_states_on_price(df, state_desc, start="2024-09-01", path=OUT_DIR / "09_vol_joint_states_price.png"):
    """Same learned states/colors as chart 08, but shown against the actual
    oil price (CL1) instead of OVX -- to see how the vol-regime states
    (fit purely on OVX level+momentum, no price information at all) line up
    with what the underlying price was actually doing."""
    plot_df = df.loc[df["date"] >= start].reset_index(drop=True)

    ordered_states = state_desc.sort_values("ovx_level")["state"].tolist()
    cmap = plt.cm.RdYlBu_r
    colors = {f"state_{s}": cmap(i / (len(ordered_states) - 1)) for i, s in enumerate(ordered_states)}

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(plot_df["date"], plot_df["cl1"], linewidth=1.4, color="black", zorder=3, label="CL1 (WTI front-month)")

    for i in range(len(plot_df) - 1):
        state = plot_df.at[i, "joint_state"]
        ax.axvspan(plot_df.at[i, "date"], plot_df.at[i + 1, "date"],
                   color=colors.get(state, "white"), alpha=0.5, linewidth=0)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[f"state_{s}"], alpha=0.5) for s in ordered_states]
    labels = [f"state {s} (OVX~{state_desc.loc[state_desc['state']==s,'ovx_level'].values[0]:.0f}, "
              f"mom={state_desc.loc[state_desc['state']==s,'momentum_5d_log_chg'].values[0]:+.3f})"
              for s in ordered_states]
    ax.legend(handles + [ax.get_lines()[0]], labels + ["CL1"], loc="upper left", fontsize=8, ncol=2)

    ax.set_title("WTI front-month price (CL1) with the same learned joint vol-regime states as Chart 8")
    ax.set_ylabel("CL1, $/bbl")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_price_with_probability_crossings(
    df, prob_col="crisis_prob_fwd10", threshold=0.50,
    path=OUT_DIR / "11_price_with_prob_crossings.png",
):
    """CL1 price over time with a star marker at each date the given
    forward-looking crisis-probability forecast first crosses above
    `threshold` (rising-edge only -- an episode sitting above threshold for
    weeks gets one star at onset, not one per day)."""
    plot_df = df.dropna(subset=[prob_col, "cl1"]).reset_index(drop=True)

    above = plot_df[prob_col] > threshold
    crossings = above & ~above.shift(1, fill_value=False)
    cross_df = plot_df.loc[crossings]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(plot_df["date"], plot_df["cl1"], linewidth=1.0, color="#4C72B0", zorder=2, label="CL1 (WTI front-month)")
    ax.scatter(cross_df["date"], cross_df["cl1"], marker="*", s=220, color="#B8860B",
               edgecolor="black", linewidth=0.6, zorder=5,
               label=f"{prob_col} first crosses above {threshold:.0%}")
    ax.axvline(config.TRAIN_END, color="gray", linestyle=":", linewidth=1.2, label="Train/test split")

    ax.set_title(f"CL1 price with star markers where {prob_col} first crossed above {threshold:.0%}")
    ax.set_ylabel("CL1, $/bbl")
    ax.set_xlabel("Date")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, cross_df[["date", "cl1", prob_col]]


PROB_BUCKET_COLORS = [
    (0.00, 0.25, "#2ECC71", "0-25% (calm)"),
    (0.25, 0.50, "#F4D03F", "25-50% (watch)"),
    (0.50, 0.75, "#E67E22", "50-75% (elevated)"),
    (0.75, 1.01, "#A569BD", "75-100% (very high)"),
]
ALREADY_IN_COLOR = "#E74C3C"
ALREADY_IN_LABEL = "Already in high-vol regime (OVX > threshold -- forecast no longer actionable)"


def _bucket_color(prob, already_in):
    if already_in:
        return ALREADY_IN_COLOR
    for lo, hi, color, _ in PROB_BUCKET_COLORS:
        if lo <= prob < hi:
            return color
    return "white"


def plot_regime_transition_zoom(
    df, start, end, prob_col="crisis_prob_fwd15", ovx_threshold=60,
    path=OUT_DIR / "15_regime_transition_zoom.png",
):
    """Zoomed-in view of one episode: background colored by the model's
    forward-looking probability bucket (green->yellow->orange->purple as
    crisis_prob_fwdN rises), switching to red once OVX has actually crossed
    into the high-vol regime -- at that point the forecast is no longer the
    relevant signal, we're already living the outcome it was forecasting."""
    plot_df = df.loc[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    already_in = (plot_df["ovx"] > ovx_threshold).to_numpy()
    probs = plot_df[prob_col].to_numpy()

    fig, ax = plt.subplots(figsize=(15, 7))

    for i in range(len(plot_df) - 1):
        color = _bucket_color(probs[i], already_in[i])
        ax.axvspan(plot_df.at[i, "date"], plot_df.at[i + 1, "date"], color=color, alpha=0.35, linewidth=0)

    ax.plot(plot_df["date"], plot_df["ovx"], linewidth=1.8, color="black", zorder=3, label="OVX")
    ax.axhline(ovx_threshold, color="black", linestyle=":", linewidth=1, alpha=0.6,
               label=f"OVX={ovx_threshold} (already-in-regime threshold)")
    ax.set_ylabel("OVX")

    ax2 = ax.twinx()
    ax2.plot(plot_df["date"], plot_df[prob_col], linewidth=1.4, color="#1A5276",
             linestyle="--", alpha=0.85, label=f"{prob_col} (right axis)")
    ax2.set_ylabel(f"{prob_col}")
    ax2.set_ylim(-0.02, 1.02)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.35) for _, _, c, _ in PROB_BUCKET_COLORS]
    labels = [lbl for _, _, _, lbl in PROB_BUCKET_COLORS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=ALREADY_IN_COLOR, alpha=0.35))
    labels.append(ALREADY_IN_LABEL)
    line1, = ax.plot([], [], color="black", linewidth=1.8)
    line2, = ax2.plot([], [], color="#1A5276", linestyle="--", linewidth=1.4)
    ax.legend(handles + [line1, line2], labels + ["OVX (left axis)", f"{prob_col} (right axis)"],
              loc="upper left", fontsize=8, ncol=1)

    ax.set_title(f"Regime-transition forecast zoom: {prob_col}, {plot_df['date'].min():%Y-%m-%d} to {plot_df['date'].max():%Y-%m-%d}")
    ax.set_xlabel("Date")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, alpha=0.25)
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
    df, train_result, crisis_regime = attach_vol_regime(df, driver_col=config.VOL_REGIME_DRIVER)
    df, trades = run_vol_backtest(df)

    for p in generate_all(df, trades):
        print(f"Saved {p}")

    chart_path, crossings = plot_price_with_probability_crossings(df, prob_col="crisis_prob_fwd10")
    print(f"Saved {chart_path}")
    print(f"\n{len(crossings)} rising-edge crossings above 50% (crisis_prob_fwd10, GPR_OIL-driven):")
    print(crossings.to_string(index=False))