"""
Joint 2D Gaussian HMM on OVX level and OVX momentum -- states are LEARNED
(fit via Baum-Welch EM), not hand-defined percentile buckets like the
earlier vol_momentum.py attempt.

Why hand-rolled instead of hmmlearn: hmmlearn's wheel requires a C++
compiler this machine doesn't have (Python 3.14 is too new for a prebuilt
wheel), and a plain multivariate Gaussian HMM is a standard, well-defined
algorithm anyway. Hand-rolling it also gives direct control over the
forward-only (causal) filter needed for no-lookahead application -- the
same reason vol_regime.py uses statsmodels' .filter() instead of .smooth().

Features (2D, both from the smoothed OVX series to reduce noise, same
smoothing as vol_momentum.py):
  - log(OVX), smoothed (3-day trailing mean) -- the LEVEL dimension.
  - 5-day log-change of that smoothed series -- the MOMENTUM dimension.
Both standardized using TRAINING-period mean/std (frozen, no lookahead)
before fitting, since raw level (~15-300) and momentum (~-0.3 to +0.5) are
on very different scales.

No-lookahead discipline: EM fitting uses training data only. Frozen fitted
parameters (means, covariances, transition matrix, initial distribution)
are then used to run the forward algorithm (not forward-backward) across
the full sample, so each day's state probability uses only information up
to and including that day.
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn.cluster import KMeans

import config
import model_cache

N_STATES = 6
OVX_SMOOTHING_WINDOW_DAYS = 3
MOMENTUM_WINDOW_DAYS = 5
COV_REGULARIZATION = 1e-3
N_RESTARTS = 10
MAX_ITER = 200
TOL = 1e-4


def build_features(df):
    df = df.copy()
    df["ovx_smooth"] = df["ovx"].rolling(OVX_SMOOTHING_WINDOW_DAYS, min_periods=1).mean()
    df["log_ovx_smooth"] = np.log(df["ovx_smooth"])
    df["ovx_momentum"] = df["log_ovx_smooth"] - df["log_ovx_smooth"].shift(MOMENTUM_WINDOW_DAYS)
    return df


def _log_emission(X, means, covs):
    n_obs = X.shape[0]
    n_states = means.shape[0]
    log_b = np.zeros((n_obs, n_states))
    for s in range(n_states):
        cov = covs[s] + np.eye(X.shape[1]) * COV_REGULARIZATION
        log_b[:, s] = multivariate_normal(mean=means[s], cov=cov).logpdf(X)
    return log_b


def _forward_backward(log_b, log_A, log_pi):
    n_obs, n_states = log_b.shape
    log_alpha = np.zeros((n_obs, n_states))
    log_beta = np.zeros((n_obs, n_states))

    log_alpha[0] = log_pi + log_b[0]
    for t in range(1, n_obs):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_b[t]

    log_beta[-1] = 0.0
    for t in range(n_obs - 2, -1, -1):
        log_beta[t] = logsumexp(log_A + log_b[t + 1] + log_beta[t + 1], axis=1)

    loglik = logsumexp(log_alpha[-1])

    log_gamma = log_alpha + log_beta - loglik

    log_xi = np.zeros((n_obs - 1, n_states, n_states))
    for t in range(n_obs - 1):
        log_xi[t] = (
            log_alpha[t][:, None] + log_A + log_b[t + 1][None, :] + log_beta[t + 1][None, :] - loglik
        )

    return log_gamma, log_xi, loglik


def _em_fit(X, n_states, seed):
    n_obs, n_dim = X.shape
    rng = np.random.RandomState(seed)

    kmeans = KMeans(n_clusters=n_states, n_init=4, random_state=seed).fit(X)
    means = kmeans.cluster_centers_.copy()
    covs = np.array([
        np.cov(X[kmeans.labels_ == s].T) + np.eye(n_dim) * COV_REGULARIZATION
        if (kmeans.labels_ == s).sum() > n_dim else np.eye(n_dim)
        for s in range(n_states)
    ])

    # Persistent prior: high self-transition probability, rest spread uniformly.
    A = np.full((n_states, n_states), 0.05 / (n_states - 1))
    np.fill_diagonal(A, 0.95)
    log_A = np.log(A)
    pi = np.full(n_states, 1 / n_states)
    log_pi = np.log(pi)

    prev_loglik = -np.inf
    for iteration in range(MAX_ITER):
        log_b = _log_emission(X, means, covs)
        log_gamma, log_xi, loglik = _forward_backward(log_b, log_A, log_pi)

        if abs(loglik - prev_loglik) < TOL:
            break
        prev_loglik = loglik

        gamma = np.exp(log_gamma)
        xi_sum = np.exp(logsumexp(log_xi, axis=0))

        # M-step
        log_pi = log_gamma[0] - logsumexp(log_gamma[0])
        A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        log_A = np.log(np.clip(A, 1e-10, None))

        for s in range(n_states):
            w = gamma[:, s]
            w_sum = w.sum()
            means[s] = (w[:, None] * X).sum(axis=0) / w_sum
            diff = X - means[s]
            covs[s] = (w[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0) / w_sum
            covs[s] += np.eye(n_dim) * COV_REGULARIZATION

    return {"means": means, "covs": covs, "A": np.exp(log_A), "pi": np.exp(log_pi), "loglik": prev_loglik}


def fit_joint_hmm(df):
    train = df.loc[df["date"] <= config.TRAIN_END, ["log_ovx_smooth", "ovx_momentum"]].dropna()

    feat_mean = train.mean()
    feat_std = train.std()

    fp = model_cache.fingerprint(
        config.TRAIN_END, N_STATES, OVX_SMOOTHING_WINDOW_DAYS, MOMENTUM_WINDOW_DAYS,
        COV_REGULARIZATION, N_RESTARTS, MAX_ITER, TOL,
        model_cache.hash_series(train["log_ovx_smooth"]), model_cache.hash_series(train["ovx_momentum"]),
    )

    def _fit():
        X_train = ((train - feat_mean) / feat_std).to_numpy()
        best = None
        for seed in range(N_RESTARTS):
            result = _em_fit(X_train, N_STATES, seed)
            if best is None or result["loglik"] > best["loglik"]:
                best = result
        return best

    best = model_cache.load_or_fit("vol_joint_hmm", fp, _fit)

    return best, feat_mean, feat_std


def forward_filter_full_sample(df, params, feat_mean, feat_std):
    """Causal (forward-only) filtered state probabilities across the full
    sample, using FROZEN training-fit parameters. Not the smoother."""
    valid = df[["log_ovx_smooth", "ovx_momentum"]].notna().all(axis=1)
    X_full = ((df.loc[valid, ["log_ovx_smooth", "ovx_momentum"]] - feat_mean) / feat_std).to_numpy()

    log_A = np.log(params["A"])
    log_pi = np.log(params["pi"])
    log_b = _log_emission(X_full, params["means"], params["covs"])

    n_obs, n_states = log_b.shape
    log_alpha = np.zeros((n_obs, n_states))
    log_alpha[0] = log_pi + log_b[0]
    log_alpha[0] -= logsumexp(log_alpha[0])
    for t in range(1, n_obs):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_b[t]
        log_alpha[t] -= logsumexp(log_alpha[t])

    filtered = np.exp(log_alpha)
    result = pd.DataFrame(filtered, index=df.loc[valid].index, columns=[f"state_{s}" for s in range(n_states)])
    return result.reindex(df.index)


def describe_states(params, feat_mean, feat_std):
    rows = []
    for s in range(len(params["means"])):
        real_log_ovx = params["means"][s][0] * feat_std["log_ovx_smooth"] + feat_mean["log_ovx_smooth"]
        real_momentum = params["means"][s][1] * feat_std["ovx_momentum"] + feat_mean["ovx_momentum"]
        rows.append({
            "state": s,
            "ovx_level": np.exp(real_log_ovx),
            "momentum_5d_log_chg": real_momentum,
            "self_persistence": params["A"][s, s],
        })
    return pd.DataFrame(rows).sort_values("ovx_level").reset_index(drop=True)


def attach_joint_regime(df):
    df = build_features(df)
    params, feat_mean, feat_std = fit_joint_hmm(df)
    filtered = forward_filter_full_sample(df, params, feat_mean, feat_std)
    df = pd.concat([df, filtered], axis=1)

    state_cols = [c for c in filtered.columns]
    valid_rows = df[state_cols].notna().all(axis=1)
    df["joint_state"] = pd.Series(np.nan, index=df.index, dtype=object)
    df.loc[valid_rows, "joint_state"] = df.loc[valid_rows, state_cols].idxmax(axis=1)

    return df, params, feat_mean, feat_std


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    from load_data import load_daily_dataset

    df = load_daily_dataset()
    df, params, feat_mean, feat_std = attach_joint_regime(df)

    print(f"Training log-likelihood: {params['loglik']:.2f}")
    print()
    print("Discovered states (sorted by OVX level):")
    print(describe_states(params, feat_mean, feat_std).to_string(index=False))
    print()
    print("Transition matrix:")
    print(np.round(params["A"], 3))