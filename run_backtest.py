"""
Entry point: load data, fit the fair-value model, run the backtest, print
diagnostics, and generate charts.

    py -3 run_backtest.py
"""

import config
from load_data import load_daily_dataset
from model import apply_fair_value_model
from regime import attach_regime
from strategy import run_backtest
import diagnostics
import charts


def main():
    df = load_daily_dataset()
    print(f"Loaded {len(df):,} daily rows, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")

    df, fv_model = apply_fair_value_model(df)
    print("\n=== Fair-value model ===")
    print(fv_model.ols_result.summary().tables[1])

    print("\n=== Regime model ===")
    df = attach_regime(df)

    print("\n=== Fair-value diagnostics ===")
    print(diagnostics.fair_value_diagnostics(df).to_string(index=False))

    print("\n=== Residual persistence (training) ===")
    print(diagnostics.residual_persistence(df))

    df, trades = run_backtest(df)

    print("\n=== Z-score threshold sensitivity (out-of-sample) ===")
    print(diagnostics.zscore_threshold_sensitivity(df).to_string(index=False))

    print("\n=== Trade / P&L summary ===")
    oos = df.loc[df["date"] > config.TRAIN_END]
    print(diagnostics.trade_summary(trades, oos).to_string(index=False))

    print("\n=== Trade log ===")
    if not trades.empty:
        print(trades[[
            "entry_date", "exit_date", "direction", "entry_spread", "exit_spread",
            "entry_zscore", "exit_zscore", "holding_days", "trade_pnl", "status",
        ]].to_string(index=False))

    print("\n=== Charts ===")
    for path in charts.generate_all(df, fv_model):
        print(f"Saved {path}")

    return df, trades, fv_model


if __name__ == "__main__":
    main()
