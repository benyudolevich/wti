"""
OVX momentum/phase classifier -- separate from and additional to the
existing level-based regime model (vol_regime.py).

The level model answers "are we in a high-vol era." This answers "within
whatever era we're in, is vol currently accelerating, decelerating, or
stable" -- the dimension the level model structurally cannot see, since a
level-only classifier reads "OVX=60 while climbing" identically to
"OVX=60 while falling."

Momentum measure: trailing 5-day percentage change of a SMOOTHED OVX
series (3-day trailing average), not raw OVX -- the first version used raw
OVX and was too choppy to be useful (flickered between all 5 phases almost
daily, even during flat/boring stretches). Smoothing the input first is
the standard fix (the same reason MACD is built on EMAs, not raw price).
Thresholds for the 5 phase buckets (rapidly_decreasing / decreasing /
stable / increasing / rapidly_increasing) are the 10th/25th/75th/90th
percentile of this measure's TRAINING-period distribution (frozen, no
lookahead) -- same calibration discipline used elsewhere in this project.

Not yet wired into any entry/exit decision -- this is a standalone
validation step first.
"""

import numpy as np
import pandas as pd

import config

OVX_SMOOTHING_WINDOW_DAYS = 3
MOMENTUM_WINDOW_DAYS = 5


def compute_vol_momentum(df):
    df = df.copy()
    df["ovx_smooth"] = df["ovx"].rolling(OVX_SMOOTHING_WINDOW_DAYS, min_periods=1).mean()
    df["ovx_chg_5d"] = df["ovx_smooth"].pct_change(MOMENTUM_WINDOW_DAYS)

    train_chg = df.loc[df["date"] <= config.TRAIN_END, "ovx_chg_5d"].dropna()
    p10, p25, p75, p90 = train_chg.quantile([0.10, 0.25, 0.75, 0.90])

    def classify(x):
        if not np.isfinite(x):
            return ""
        if x <= p10:
            return "rapidly_decreasing"
        if x <= p25:
            return "decreasing"
        if x < p75:
            return "stable"
        if x < p90:
            return "increasing"
        return "rapidly_increasing"

    df["vol_phase"] = df["ovx_chg_5d"].map(classify)

    thresholds = {"p10": p10, "p25": p25, "p75": p75, "p90": p90}
    return df, thresholds


if __name__ == "__main__":
    from load_data import load_daily_dataset

    df = load_daily_dataset()
    df, thresholds = compute_vol_momentum(df)

    print(f"Training-period 5-day OVX %% change thresholds: {thresholds}")
    print()
    print("Phase distribution (full history):")
    print(df["vol_phase"].value_counts())
    print()
    print("Phase distribution (out-of-sample only):")
    print(df.loc[df["date"] > config.TRAIN_END, "vol_phase"].value_counts())