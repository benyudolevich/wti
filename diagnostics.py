"""
Model and strategy diagnostics.

Kept to the subset of metrics actually used while developing this model
(fair-value fit quality in/out of sample, residual persistence, trade/PnL
summary). Extend as needed rather than front-loading every possible metric.
"""

import numpy as np
import pandas as pd

import config


def fair_value_diagnostics(df):
    train_mask = (df["date"] <= config.TRAIN_END) & df["residual"].notna()
    test_mask = (df["date"] > config.TRAIN_END) & df["residual"].notna()

    rows = []
    for label, mask in [("Train", train_mask), ("Test (out-of-sample)", test_mask)]:
        r = df.loc[mask, "residual"]
        rows.append({
            "Sample": label,
            "n": len(r),
            "RMSE": np.sqrt(np.mean(r ** 2)),
            "MAE": r.abs().mean(),
            "Bias": r.mean(),
        })
    return pd.DataFrame(rows)


def residual_persistence(df):
    """AR(1) coefficient and approximate half-life of the training residual."""
    train_resid = df.loc[df["date"] <= config.TRAIN_END, "residual"].dropna()
    x = train_resid.iloc[:-1].to_numpy()
    y = train_resid.iloc[1:].to_numpy()
    ar1 = np.corrcoef(x, y)[0, 1]

    if ar1 <= 0 or ar1 >= 1:
        half_life = np.nan
    else:
        half_life = np.log(0.5) / np.log(ar1)

    return {"ar1_coefficient": ar1, "half_life_days": half_life}


def zscore_threshold_sensitivity(df, thresholds=(0.75, 1.0, 1.25, 1.5, 2.0)):
    test = df.loc[(df["date"] > config.TRAIN_END) & df["zscore"].notna()]
    rows = []
    for t in thresholds:
        breach_days = (test["zscore"].abs() >= t).sum()
        episodes = (
            (test["zscore"].abs() >= t)
            .astype(int)
            .diff()
            .eq(1)
            .sum()
        )
        rows.append({"threshold": t, "breach_days": breach_days, "episodes": episodes})
    return pd.DataFrame(rows)


def trade_summary(trades, oos_df):
    closed = trades.loc[trades["status"] == "CLOSED"] if not trades.empty else pd.DataFrame()

    if len(closed) > 0:
        winning = closed["trade_pnl"] > 0
        gross_profit = closed.loc[winning, "trade_pnl"].sum()
        gross_loss = abs(closed.loc[~winning, "trade_pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        win_rate = winning.mean()
        avg_trade = closed["trade_pnl"].mean()
    else:
        profit_factor = np.nan
        win_rate = np.nan
        avg_trade = np.nan

    daily_pnl_std = oos_df["daily_pnl"].std(ddof=1)
    sharpe = (
        oos_df["daily_pnl"].mean() / daily_pnl_std * np.sqrt(252)
        if daily_pnl_std > 0 else np.nan
    )

    pct_in_position = (oos_df["position"] != 0).mean()
    avg_holding = closed["holding_days"].mean() if len(closed) > 0 else np.nan
    max_holding = closed["holding_days"].max() if len(closed) > 0 else np.nan

    return pd.DataFrame({
        "Metric": [
            "Total P&L", "Max drawdown", "Annualized Sharpe (daily P&L)",
            "Closed trades", "Win rate", "Average trade P&L", "Profit factor",
            "Pct of days in a position", "Average holding days", "Max holding days",
            "Current position",
        ],
        "Value": [
            oos_df["daily_pnl"].sum(), oos_df["drawdown"].min(), sharpe,
            len(closed), win_rate, avg_trade, profit_factor,
            pct_in_position, avg_holding, max_holding,
            float(oos_df["position"].iloc[-1]),
        ],
    })


if __name__ == "__main__":
    from load_data import load_daily_dataset
    from model import apply_fair_value_model
    from strategy import run_backtest

    df = load_daily_dataset()
    df, fv_model = apply_fair_value_model(df)
    df, trades = run_backtest(df)

    print("--- Fair value diagnostics ---")
    print(fair_value_diagnostics(df).to_string(index=False))

    print("\n--- Residual persistence (training) ---")
    print(residual_persistence(df))

    print("\n--- Z-score threshold sensitivity (out-of-sample) ---")
    print(zscore_threshold_sensitivity(df).to_string(index=False))

    print("\n--- Trade / P&L summary ---")
    oos = df.loc[df["date"] > config.TRAIN_END]
    print(trade_summary(trades, oos).to_string(index=False))
