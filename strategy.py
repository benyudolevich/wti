"""
Trading strategy: entries at |zscore| >= ENTRY_Z, exits on partial
convergence / stop-loss / time-stop / GRI-regime-blocked re-entry. One
spread position at a time, daily close-to-close P&L.

Four risk-management mechanisms layered on top of the original bare
mean-reversion signal (see config.py for the calibration rationale on each):
  - EXIT_Z: partial-convergence exit instead of waiting for an exact
    zero-crossing, so capital frees up for new entries sooner.
  - STOP_Z_ABS: hard stop-loss on adverse z-score excursion, caps the
    "ride an unbounded move" failure mode directly.
  - MAX_HOLDING_DAYS: time stop, calibrated off the weekly-resampled
    residual half-life rather than guessed.
  - CRISIS_PROB_HARD_BLOCK / continuous sizing: rather than a binary entry
    block, position size at entry is scaled by (1 - crisis_prob), frozen
    for the life of the trade. A pure binary block (crisis_prob > 0.5)
    turned out to suppress 93.6% of the entire out-of-sample window, since
    GRI has been structurally elevated across nearly all of it, not just
    during the acute March-June spike -- continuous sizing lets the
    strategy keep trading through "elevated but not extreme" periods while
    still shrinking hard exactly when the model is confident a live crisis
    is underway. Still fully blocked above CRISIS_PROB_HARD_BLOCK as a
    backstop for the most extreme readings.
"""

import numpy as np
import pandas as pd

import config


def generate_positions(df):
    """df must already have a 'crisis_prob' column (see regime.attach_regime)."""
    df = df.copy()
    n = len(df)
    position = np.zeros(n, dtype=float)
    entry_index = np.full(n, -1, dtype=int)
    entry_zscore = np.full(n, np.nan)
    entry_direction = np.zeros(n, dtype=int)
    exit_reason = np.full(n, "", dtype=object)

    for i in range(1, n):
        date = df.at[i, "date"]
        zscore = df.at[i, "zscore"]
        crisis_prob = df.at[i, "crisis_prob"]

        if date <= config.TRAIN_END:
            position[i] = 0
            continue

        if not np.isfinite(zscore):
            position[i] = 0
            continue

        previous_position = position[i - 1]
        previous_direction = entry_direction[i - 1]

        if previous_position == 0:
            hard_blocked = np.isfinite(crisis_prob) and crisis_prob > config.CRISIS_PROB_HARD_BLOCK
            size_fraction = 1 - crisis_prob if np.isfinite(crisis_prob) else 1.0

            direction = 0
            if not hard_blocked:
                if zscore <= -config.ENTRY_Z:
                    direction = 1
                elif zscore >= config.ENTRY_Z:
                    direction = -1

            if direction != 0:
                position[i] = direction * size_fraction
                entry_index[i] = i
                entry_zscore[i] = zscore
                entry_direction[i] = direction
            else:
                position[i] = 0

        else:
            prev_entry_index = entry_index[i - 1]
            prev_entry_zscore = entry_zscore[i - 1]
            days_held = i - prev_entry_index

            converged = (
                (previous_direction == 1 and zscore >= -config.EXIT_Z)
                or (previous_direction == -1 and zscore <= config.EXIT_Z)
            )
            stopped_out = (
                (previous_direction == 1 and zscore <= -config.STOP_Z_ABS)
                or (previous_direction == -1 and zscore >= config.STOP_Z_ABS)
            )
            timed_out = days_held >= config.MAX_HOLDING_DAYS

            if converged or stopped_out or timed_out:
                position[i] = 0
                exit_reason[i] = "stop_loss" if stopped_out else ("time_stop" if timed_out else "converged")
            else:
                position[i] = previous_position
                entry_index[i] = prev_entry_index
                entry_zscore[i] = prev_entry_zscore
                entry_direction[i] = previous_direction

    df["position"] = position
    df["exit_reason"] = exit_reason
    df["previous_position"] = df["position"].shift(1).fillna(0)
    df["spread_change"] = df["spread"].diff()

    df["gross_pnl"] = df["previous_position"] * df["spread_change"] * config.CONTRACT_SIZE_BBL
    df["position_change"] = (df["position"] - df["previous_position"]).abs()
    df["transaction_cost"] = df["position_change"] * config.COST_PER_POSITION_CHANGE
    df["daily_pnl"] = df["gross_pnl"] - df["transaction_cost"]

    df.loc[df["date"] <= config.TRAIN_END, "daily_pnl"] = 0.0
    df["daily_pnl"] = df["daily_pnl"].fillna(0.0)
    df["cumulative_pnl"] = df["daily_pnl"].cumsum()
    df["equity_peak"] = df["cumulative_pnl"].cummax()
    df["drawdown"] = df["cumulative_pnl"] - df["equity_peak"]

    # Position is now a signed SIZE FRACTION (e.g. 0.3, -0.05), not exactly
    # +-1, since entries are scaled by (1 - crisis_prob) -- compare sign and
    # nonzero-ness rather than exact equality.
    df["action"] = ""
    df.loc[(df["previous_position"] == 0) & (df["position"] > 0), "action"] = "ENTER LONG"
    df.loc[(df["previous_position"] == 0) & (df["position"] < 0), "action"] = "ENTER SHORT"
    df.loc[(df["previous_position"] > 0) & (df["position"] == 0), "action"] = "EXIT LONG"
    df.loc[(df["previous_position"] < 0) & (df["position"] == 0), "action"] = "EXIT SHORT"

    return df


def build_trade_log(df):
    trades = []
    open_trade = None

    for i, row in df.iterrows():
        action = row["action"]

        if action in ("ENTER LONG", "ENTER SHORT"):
            open_trade = {
                "entry_index": i,
                "entry_date": row["date"],
                "direction": "LONG" if row["position"] > 0 else "SHORT",
                "entry_size_fraction": abs(row["position"]),
                "entry_spread": row["spread"],
                "entry_fair_value": row["fair_value"],
                "entry_zscore": row["zscore"],
            }

        elif action in ("EXIT LONG", "EXIT SHORT") and open_trade is not None:
            entry_index = open_trade["entry_index"]
            trade_pnl = df.loc[entry_index:i, "daily_pnl"].sum()
            trades.append({
                **open_trade,
                "exit_index": i,
                "exit_date": row["date"],
                "exit_spread": row["spread"],
                "exit_fair_value": row["fair_value"],
                "exit_zscore": row["zscore"],
                "holding_days": i - entry_index,
                "trade_pnl": trade_pnl,
                "status": "CLOSED",
                "exit_reason": row["exit_reason"],
            })
            open_trade = None

    if open_trade is not None:
        i = df.index[-1]
        entry_index = open_trade["entry_index"]
        trade_pnl = df.loc[entry_index:i, "daily_pnl"].sum()
        trades.append({
            **open_trade,
            "exit_index": np.nan,
            "exit_date": pd.NaT,
            "exit_spread": np.nan,
            "exit_fair_value": np.nan,
            "exit_zscore": np.nan,
            "holding_days": i - entry_index,
            "trade_pnl": trade_pnl,
            "status": "OPEN",
            "exit_reason": "",
        })

    return pd.DataFrame(trades)


def run_backtest(df):
    df = generate_positions(df)
    trades = build_trade_log(df)
    return df, trades


if __name__ == "__main__":
    from load_data import load_daily_dataset
    from model import apply_fair_value_model
    from regime import attach_regime

    df = load_daily_dataset()
    df, fv_model = apply_fair_value_model(df)
    df = attach_regime(df)
    df, trades = run_backtest(df)

    oos = df.loc[df["date"] > config.TRAIN_END]
    print(f"Total P&L: {oos['daily_pnl'].sum():.0f}")
    print(f"Max drawdown: {oos['drawdown'].min():.0f}")
    print(f"Closed trades: {(trades['status'] == 'CLOSED').sum() if not trades.empty else 0}")
    print(trades)
