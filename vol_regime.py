"""
Forward-looking volatility regime model.

Unlike regime.py (which classifies the CURRENT state from GRI alone with a
fixed transition matrix), this fits a Markov-switching model on log(OVX) --
the actual volatility series we care about -- where the TRANSITION
PROBABILITIES themselves are a function of the configured geopolitical-risk driver (time-varying transition
probabilities, TVTP). That means when GRI rises, the model's own estimated
probability of transitioning into the high-vol state rises too, *before*
OVX has necessarily moved -- which is what lets this produce a genuine
h-day-ahead forecast rather than just a same-day classification.

No-lookahead discipline, same as regime.py: parameters (state means,
variance, and the TVTP logit coefficients) are estimated on the TRAINING
window only, then frozen and applied via the causal Hamilton filter (not
the Kim smoother) across the full sample.

Forecasting mechanism: given the filtered state distribution at time t and
the transition matrix implied by GRI_t (frozen-covariate assumption -- we
don't know future GRI, so we project forward assuming today's GRI level
persists), the h-day-ahead state distribution is TM^h @ state_t (Chapman-
Kolmogorov). This is what "5 days before a regime shift" cashes out to
mechanically.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

import config
import model_cache

FORECAST_HORIZONS = (3, 5, 10, 15)


def _build_exog_tvtp(df, driver_col, driver_center, driver_scale):
    return np.column_stack([
        np.ones(len(df)),
        (df[driver_col].to_numpy() - driver_center) / driver_scale,
    ])


def fit_vol_regime_model(df, driver_col=None):
    driver_col = driver_col or config.VOL_REGIME_DRIVER
    train = df.loc[df["date"] <= config.TRAIN_END, ["log_ovx", driver_col]].dropna()
    driver_center = train[driver_col].mean()
    driver_scale = train[driver_col].std()

    fp = model_cache.fingerprint(
        config.TRAIN_END, config.VOL_REGIME_K_STATES, driver_col,
        model_cache.hash_series(train["log_ovx"]), model_cache.hash_series(train[driver_col]),
    )

    def _fit():
        exog_tvtp_train = _build_exog_tvtp(train, driver_col, driver_center, driver_scale)
        train_model = MarkovRegression(
            train["log_ovx"].to_numpy(),
            k_regimes=config.VOL_REGIME_K_STATES,
            exog_tvtp=exog_tvtp_train,
            switching_variance=False,
        )
        train_result = train_model.fit(search_reps=50, maxiter=500)
        if not train_result.mle_retvals.get("converged", False):
            train_result = train_model.fit(
                start_params=train_result.params, method="bfgs", maxiter=1000, disp=False
            )

        regime_means = [
            np.average(train["log_ovx"].to_numpy(), weights=train_result.filtered_marginal_probabilities[:, s])
            for s in range(config.VOL_REGIME_K_STATES)
        ]
        crisis_regime = int(np.argmax(regime_means))

        return train_model, train_result, crisis_regime, regime_means

    train_model, train_result, crisis_regime, regime_means = model_cache.load_or_fit(
        f"vol_regime_tvtp_{driver_col}", fp, _fit
    )

    return train_model, train_result, crisis_regime, regime_means, driver_center, driver_scale


def compute_filtered_and_forecasts(df, train_result, crisis_regime, driver_col, driver_center, driver_scale):
    valid = df[["log_ovx", driver_col]].notna().all(axis=1)
    valid_df = df.loc[valid]
    exog_tvtp_full = _build_exog_tvtp(valid_df, driver_col, driver_center, driver_scale)

    full_model = MarkovRegression(
        valid_df["log_ovx"].to_numpy(),
        k_regimes=config.VOL_REGIME_K_STATES,
        exog_tvtp=exog_tvtp_full,
        switching_variance=False,
    )
    filtered = full_model.filter(train_result.params)
    state_probs = filtered.filtered_marginal_probabilities  # (nobs, k_regimes)

    # Transition matrix implied at each date by that date's driver level.
    transition_matrices = full_model.regime_transition_matrix(train_result.params, exog_tvtp=exog_tvtp_full)
    # shape (k_regimes, k_regimes, nobs); columns sum to 1 (Hamilton convention:
    # tm[:, :, t] @ state_vector_{t-1} = state_vector_t)

    nobs = state_probs.shape[0]
    out = {f"crisis_prob_fwd{h}": np.full(nobs, np.nan) for h in FORECAST_HORIZONS}
    out["crisis_prob_now"] = state_probs[:, crisis_regime]

    for t in range(nobs):
        tm_t = transition_matrices[:, :, t]
        state_vec = state_probs[t, :]
        for h in FORECAST_HORIZONS:
            tm_h = np.linalg.matrix_power(tm_t, h)
            forecast_vec = tm_h @ state_vec
            out[f"crisis_prob_fwd{h}"][t] = forecast_vec[crisis_regime]

    result = pd.DataFrame(out, index=valid_df.index).reindex(df.index)
    return result


def attach_vol_regime(df, driver_col=None):
    driver_col = driver_col or config.VOL_REGIME_DRIVER
    df = df.copy()
    train_model, train_result, crisis_regime, regime_means, driver_center, driver_scale = fit_vol_regime_model(
        df, driver_col
    )

    ordered = sorted(np.exp(regime_means))
    print(f"Vol-regime means (OVX), low to high: {[f'{m:.1f}' for m in ordered]}")
    print(f"Converged: {train_result.mle_retvals.get('converged', 'n/a')}, log-likelihood: {train_result.llf:.2f}")

    forecasts = compute_filtered_and_forecasts(df, train_result, crisis_regime, driver_col, driver_center, driver_scale)
    df = pd.concat([df, forecasts], axis=1)

    return df, train_result, crisis_regime


if __name__ == "__main__":
    from load_data import load_daily_dataset

    df = load_daily_dataset()
    df["log_ovx"] = np.log(df["ovx"])
    df, train_result, crisis_regime = attach_vol_regime(df)

    cols = ["date", "ovx", "gri", "crisis_prob_now"] + [f"crisis_prob_fwd{h}" for h in FORECAST_HORIZONS]
    recent = df.loc[df["date"] >= "2026-01-01", cols]
    print()
    print(recent.iloc[::5].to_string(index=False))