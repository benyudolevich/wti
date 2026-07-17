"""
Regime detection: 2-state Markov-switching model on log(GRI), fit on
TRAINING data only, then applied causally (filtered, no-lookahead) across
the full sample.

Replaces the earlier ad hoc percentile-on-smoothed-GRI heuristic. That
heuristic had two problems this addresses directly:
  - A single noisy day of GRI dipping under a fixed threshold could reopen
    the gate mid-crisis (happened twice, 2026-04-21 and 2026-04-27, each
    immediately followed by a stop-loss).
  - The threshold and smoothing window were both hand-picked, not fit.

A Markov-switching model estimates a "calm" state (lower mean/lower
variance log-GRI) and a "crisis" state (higher mean/higher variance),
plus a transition matrix that captures how persistent each state tends to
be -- so regime persistence is a property learned from data, not a
manually chosen smoothing window.

No-lookahead discipline: the model's parameters (state means/variances,
transition probabilities) are estimated ONLY on the training window, then
frozen. Those frozen parameters are used to run the Hamilton FILTER (not
the Kim smoother) across the full train+test sample, which at each date
only uses information up to and including that date.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

import config


def fit_regime_model(df):
    train = df.loc[df["date"] <= config.TRAIN_END, "log_gri"].dropna()

    train_model = MarkovRegression(train.to_numpy(), k_regimes=config.REGIME_K_STATES, switching_variance=False)
    train_result = train_model.fit(search_reps=50)

    # Identify which state is "crisis" (highest mean log(GRI)) by its
    # filtered-probability-weighted mean, rather than assuming a fixed
    # parameter order.
    regime_means = [
        np.average(train.to_numpy(), weights=train_result.filtered_marginal_probabilities[:, s])
        for s in range(config.REGIME_K_STATES)
    ]
    crisis_regime = int(np.argmax(regime_means))

    return train_model, train_result, crisis_regime, regime_means


def compute_crisis_probability(df, train_result, crisis_regime):
    """Causal (filtered-only) crisis-regime probability across the full sample,
    using the frozen training-period parameters -- no re-estimation on test data."""
    valid = df["log_gri"].notna()
    full_model = MarkovRegression(
        df.loc[valid, "log_gri"].to_numpy(), k_regimes=config.REGIME_K_STATES, switching_variance=False
    )
    filtered = full_model.filter(train_result.params)
    crisis_prob = filtered.filtered_marginal_probabilities[:, crisis_regime]

    result = pd.Series(np.nan, index=df.index)
    result.loc[valid] = crisis_prob
    return result


def attach_regime(df):
    df = df.copy()
    train_model, train_result, crisis_regime, regime_means = fit_regime_model(df)
    df["crisis_prob"] = compute_crisis_probability(df, train_result, crisis_regime)

    ordered = sorted(np.exp(regime_means))
    print(f"Regime means (GRI), low to high: {[f'{m:.1f}' for m in ordered]}")
    print(f"Training transition matrix:\n{train_result.regime_transition[:, :, 0]}")

    p_stay_crisis = train_result.regime_transition[crisis_regime, crisis_regime, 0]
    expected_duration = 1 / (1 - p_stay_crisis)
    print(f"Estimated crisis-regime persistence: P(stay)={p_stay_crisis:.3f}, expected duration={expected_duration:.1f} days")

    return df


if __name__ == "__main__":
    from load_data import load_daily_dataset

    df = load_daily_dataset()
    df = attach_regime(df)

    recent = df.loc[df["date"] >= "2026-01-01", ["date", "gri", "crisis_prob"]]
    print()
    print(recent.iloc[::5].to_string(index=False))