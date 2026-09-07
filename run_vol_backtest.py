"""Run the forward-looking volatility-regime timing research path."""

import argparse
import numpy as np

import config
from load_data import load_daily_dataset
from vol_regime import attach_vol_regime
from vol_strategy import run_vol_backtest, vol_strategy_summary


def main(driver_col=None):
    driver_col = driver_col or config.VOL_REGIME_DRIVER

    df = load_daily_dataset()
    if driver_col not in df.columns:
        raise ValueError(
            f"Unknown driver {driver_col!r}. Available columns include "
            "'log_gri' and 'log_gpr_oil'."
        )

    df["log_ovx"] = np.log(df["ovx"])
    df, _, _ = attach_vol_regime(df, driver_col=driver_col)
    df, trades = run_vol_backtest(df)

    print(f"TVTP driver: {driver_col}")
    print()
    print(vol_strategy_summary(trades).to_string(index=False))

    if not trades.empty:
        print()
        print(trades.to_string(index=False))

    return df, trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--driver",
        default=config.VOL_REGIME_DRIVER,
        choices=["log_gpr_oil", "log_gri", "log_gpr_oil_me"],
        help="TVTP transition driver; defaults to the configured oil-specific index.",
    )
    args = parser.parse_args()
    main(args.driver)
