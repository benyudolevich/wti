"""
Validate the 2-state TVTP vol-regime model against every historical
high-OVX episode, not just the 2026 one.

Important methodological point: episodes before TRAIN_END (2025-01-13)
were part of the TRAINING data, so strong performance there mainly checks
in-sample fit quality (does it recover episodes it was fit on), not
generalization. Only episodes after TRAIN_END are a genuine out-of-sample
test. Both are reported, clearly separated.

The key test of the "forward-looking" claim: for each episode onset, what
did the h-day-ahead FORECAST (crisis_prob_fwd{h}), computed h days BEFORE
onset, actually show? That is the forecast a trader would have had in hand
at the time -- not the same-day classification.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from load_data import load_daily_dataset
from vol_regime import attach_vol_regime, FORECAST_HORIZONS
import config

df = load_daily_dataset()
df["log_ovx"] = np.log(df["ovx"])
df, train_result, crisis_regime = attach_vol_regime(df)
df = df.reset_index(drop=True)

high = df["ovx"] > 60
episode_id = (~high).cumsum()
episodes = (
    df.loc[high]
    .groupby(episode_id[high])
    .agg(start=("date", "min"), end=("date", "max"), peak_ovx=("ovx", "max"))
    .reset_index(drop=True)
)

results = []
for _, ep in episodes.iterrows():
    onset_idx = df.index[df["date"] == ep["start"]][0]
    in_sample = ep["start"] <= config.TRAIN_END

    row = {
        "start": ep["start"].date(),
        "peak_ovx": ep["peak_ovx"],
        "in_sample": in_sample,
        "prob_now_at_onset": df.at[onset_idx, "crisis_prob_now"],
    }

    # The actual test: h-day-ahead FORECAST computed h days before onset.
    # This is what a trader would have seen in advance, using only
    # information available at that earlier date.
    for h in FORECAST_HORIZONS:
        lead_idx = onset_idx - h
        row[f"fwd{h}_forecast_{h}d_before"] = (
            df.at[lead_idx, f"crisis_prob_fwd{h}"] if lead_idx >= 0 else np.nan
        )

    episode_mask = (df["date"] >= ep["start"]) & (df["date"] <= ep["end"])
    row["peak_prob_during"] = df.loc[episode_mask, "crisis_prob_now"].max()

    results.append(row)

results_df = pd.DataFrame(results)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

print("=== ALL EPISODES: forward forecast computed h days before onset ===")
print(results_df.to_string(index=False))

print()
print("=== Baseline: average fwd{h} forecast during genuinely calm days (OVX < 35) ===")
calm_mask = df["ovx"] < 35
for h in FORECAST_HORIZONS:
    print(f"  fwd{h} baseline mean: {df.loc[calm_mask, f'crisis_prob_fwd{h}'].mean():.4f}")

print()
print("=== Mean forward-forecast lift vs calm baseline, by horizon (out-of-sample episodes only) ===")
oos = results_df.loc[~results_df["in_sample"]]
for h in FORECAST_HORIZONS:
    col = f"fwd{h}_forecast_{h}d_before"
    baseline = df.loc[calm_mask, f"crisis_prob_fwd{h}"].mean()
    print(f"  h={h:>2}: OOS episode mean={oos[col].mean():.3f}  (calm baseline={baseline:.4f})")

results_df.to_csv("vol_regime_validation.csv", index=False)
