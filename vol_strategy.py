"""
Volatility-timing strategy: FLAT / LONG_VOL state machine on top of the
validated TVTP vol-regime forecast, entry only.

Structure (confirmed design, 2026-07-16):
  - Two states only: FLAT and LONG_VOL. No direction ambiguity like the
    spread strategy -- this is purely "do we want vol exposure right now."
  - Entry: crisis_prob_fwd10 >= 0.50 (the forecast signal, validated
    separately -- see report Part II).
  - Exit: a level-based bracket on OVX itself, anchored to the OVX level
    AT ENTRY (not a trailing peak) -- take profit if OVX rises
    VOL_STRATEGY_TAKE_PROFIT_PCT above entry, stop loss if OVX falls
    VOL_STRATEGY_STOP_LOSS_PCT below entry. No forecast dependency once in
    the trade at all.
  - Binary sizing (fixed size in/out) -- no continuous scaling yet, since
    there's no payoff model attached to calibrate a sizing dial against.

Rationale for the exit redesign: the original all-forecast exit
(crisis_prob_fwd5 dropping below a threshold) waited for the model to
confirm the crisis regime was ending, which happens well after the actual
peak -- trade 1 entered at OVX=55.26, peaked at 71.56, but didn't exit
until OVX had round-tripped all the way to 34.38, BELOW entry, giving back
the entire gain. A level-based bracket locks in a defined outcome (take
profit or stop loss) without depending on the forecast to decay in time.

No payoff model is attached (deliberately deferred -- see report Part IV).

Trading is restricted to strictly after TRAIN_END, consistent with the
rest of the project's no-lookahead discipline and because the historical
validation showed pre-cutoff episodes are an in-sample fit check, not a
real test.
"""

import numpy as np
import pandas as pd

import config


def generate_vol_positions(df):
    """df must already have crisis_prob_fwd{h} columns (see vol_regime.attach_vol_regime)."""
    df = df.copy()
    n = len(df)
    position = np.zeros(n, dtype=int)
    entry_ovx = np.full(n, np.nan)
    exit_reason = np.full(n, "", dtype=object)

    entry_col = f"crisis_prob_fwd{config.VOL_STRATEGY_ENTRY_HORIZON}"

    for i in range(1, n):
        date = df.at[i, "date"]

        if date <= config.TRAIN_END:
            position[i] = 0
            continue

        entry_signal = df.at[i, entry_col]
        ovx_today = df.at[i, "ovx"]
        previous_position = position[i - 1]

        if previous_position == 0:
            if np.isfinite(entry_signal) and entry_signal >= config.VOL_STRATEGY_ENTRY_THRESHOLD:
                position[i] = 1
                entry_ovx[i] = ovx_today
            else:
                position[i] = 0
        else:
            prev_entry_ovx = entry_ovx[i - 1]
            take_profit_level = prev_entry_ovx * (1 + config.VOL_STRATEGY_TAKE_PROFIT_PCT)
            stop_loss_level = prev_entry_ovx * (1 - config.VOL_STRATEGY_STOP_LOSS_PCT)

            hit_take_profit = np.isfinite(ovx_today) and ovx_today >= take_profit_level
            hit_stop_loss = np.isfinite(ovx_today) and ovx_today <= stop_loss_level

            if hit_take_profit or hit_stop_loss:
                position[i] = 0
                exit_reason[i] = "take_profit" if hit_take_profit else "stop_loss"
            else:
                position[i] = 1
                entry_ovx[i] = prev_entry_ovx

    df["vol_position"] = position
    df["vol_exit_reason"] = exit_reason
    df["vol_previous_position"] = df["vol_position"].shift(1).fillna(0)

    df["vol_action"] = ""
    df.loc[(df["vol_previous_position"] == 0) & (df["vol_position"] == 1), "vol_action"] = "ENTER LONG_VOL"
    df.loc[(df["vol_previous_position"] == 1) & (df["vol_position"] == 0), "vol_action"] = "EXIT LONG_VOL"

    return df


def _peak_realized_vol_over_slice(holding_slice):
    # Peak of the trailing realized_vol_20d series during the holding
    # period, NOT its average -- an average over a long hold dilutes a
    # brief, real spike into an unremarkable number (confirmed empirically:
    # trade 1's average realized vol over its 66-day hold was ~flat at
    # 42.1 vs an entry level of 42.2, masking a genuine peak of 55.5 that
    # occurred partway through, a +32% expansion).
    return holding_slice["realized_vol_20d"].max()


def build_vol_trade_log(df):
    trades = []
    open_trade = None

    for i, row in df.iterrows():
        action = row["vol_action"]

        if action == "ENTER LONG_VOL":
            open_trade = {
                "entry_index": i,
                "entry_date": row["date"],
                "entry_ovx": row["ovx"],
                "entry_realized_vol": row["realized_vol_20d"],
                "entry_fwd10": row[f"crisis_prob_fwd{config.VOL_STRATEGY_ENTRY_HORIZON}"],
            }

        elif action == "EXIT LONG_VOL" and open_trade is not None:
            entry_index = open_trade["entry_index"]
            holding_slice = df.loc[entry_index:i]
            realized_vol_during = _peak_realized_vol_over_slice(holding_slice)

            trades.append({
                **open_trade,
                "exit_index": i,
                "exit_date": row["date"],
                "exit_ovx": row["ovx"],
                "exit_reason": row["vol_exit_reason"],
                "ovx_pct_move": row["ovx"] / open_trade["entry_ovx"] - 1,
                "peak_ovx_during": holding_slice["ovx"].max(),
                "holding_days": i - entry_index,
                "peak_realized_vol_during": realized_vol_during,
                "vol_expansion_ratio": (
                    realized_vol_during / open_trade["entry_realized_vol"]
                    if open_trade["entry_realized_vol"] else np.nan
                ),
                "status": "CLOSED",
            })
            open_trade = None

    if open_trade is not None:
        i = df.index[-1]
        entry_index = open_trade["entry_index"]
        holding_slice = df.loc[entry_index:i]
        realized_vol_during = _peak_realized_vol_over_slice(holding_slice)
        current_ovx = df.at[i, "ovx"]
        trades.append({
            **open_trade,
            "exit_index": np.nan,
            "exit_date": pd.NaT,
            "exit_ovx": np.nan,
            "exit_reason": "",
            "ovx_pct_move": current_ovx / open_trade["entry_ovx"] - 1 if open_trade["entry_ovx"] else np.nan,
            "peak_ovx_during": holding_slice["ovx"].max(),
            "holding_days": i - entry_index,
            "peak_realized_vol_during": realized_vol_during,
            "vol_expansion_ratio": (
                realized_vol_during / open_trade["entry_realized_vol"]
                if open_trade["entry_realized_vol"] else np.nan
            ),
            "status": "OPEN",
        })

    return pd.DataFrame(trades)


def run_vol_backtest(df):
    df = generate_vol_positions(df)
    trades = build_vol_trade_log(df)
    return df, trades


def vol_strategy_summary(trades):
    closed = trades.loc[trades["status"] == "CLOSED"] if not trades.empty else pd.DataFrame()

    if len(closed) > 0:
        avg_holding = closed["holding_days"].mean()
        median_holding = closed["holding_days"].median()
        take_profit_rate = (closed["exit_reason"] == "take_profit").mean()
        stop_loss_rate = (closed["exit_reason"] == "stop_loss").mean()
        avg_ovx_pct_move = closed["ovx_pct_move"].mean()
        avg_ovx_move_to_peak = (closed["peak_ovx_during"] - closed["entry_ovx"]).mean()
    else:
        avg_holding = median_holding = take_profit_rate = stop_loss_rate = np.nan
        avg_ovx_pct_move = avg_ovx_move_to_peak = np.nan

    return pd.DataFrame({
        "Metric": [
            "Total trades", "Closed trades", "Average holding days",
            "Median holding days", "Pct closed via take-profit",
            "Pct closed via stop-loss", "Average OVX pct move at exit",
            "Average OVX move (entry to peak)",
        ],
        "Value": [
            len(trades), len(closed), avg_holding,
            median_holding, take_profit_rate, stop_loss_rate,
            avg_ovx_pct_move, avg_ovx_move_to_peak,
        ],
    })


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    from load_data import load_daily_dataset
    from vol_regime import attach_vol_regime

    df = load_daily_dataset()
    df["log_ovx"] = np.log(df["ovx"])
    df, train_result, crisis_regime = attach_vol_regime(df)
    df, trades = run_vol_backtest(df)

    print()
    print(vol_strategy_summary(trades).to_string(index=False))
    print()
    if not trades.empty:
        cols = [
            "entry_date", "exit_date", "entry_ovx", "exit_ovx", "peak_ovx_during",
            "holding_days", "entry_fwd10", "ovx_pct_move", "exit_reason", "status",
        ]
        print(trades[cols].to_string(index=False))