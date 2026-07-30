"""
Fair-value model: natural cubic spline in Cushing utilization + natural
cubic spline in log(GRI), fit by OLS on the training period only.

Chosen over the earlier hyperbolic-in-utilization / detrended-utilization
alternatives after head-to-head out-of-sample testing:
  - spline vs hyperbolic: no accuracy difference, spline is safer to
    extrapolate (bounded slope past the training range).
  - plain utilization vs detrended/standardized utilization: plain
    utilization won on every out-of-sample metric (detrending destroyed the
    ability to detect a sustained, slow-building squeeze).
  - adding OVX: hurts (its historical sign is dominated by demand-collapse
    vol episodes like 2020, wrong direction for a supply-fear scenario).
  - adding GRI (spline, not linear): helps on every out-of-sample metric,
    though it still understates true event magnitude during the most
    extreme geopolitical episodes -- treat that residual gap as something
    for the strategy's risk management to handle, not the fair-value curve.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

import config

# constraints="center" re-parameterizes each spline basis to be orthogonal
# to the constant. Without it, patsy's automatic Intercept plus two cr()
# terms (each of which already spans a constant direction on its own) made
# the design matrix exactly rank-deficient (rank 7 of 9 columns, condition
# number ~1.1e16) -- discovered while building vol_fair_value.py, where the
# same pattern with only one spline term caused OLS coefficients to blow up
# to ~1e11 (a near-singular matrix being inverted rather than correctly
# recognized as singular). Here the un-centered version happened to still
# produce finite, reasonable-looking coefficients (apparently on the right
# side of numpy.linalg.pinv's truncation threshold), but the underlying
# rank-deficiency was the same latent bug -- coefficients not uniquely
# identified, and fragile to any change in matrix conditioning (new data,
# different LAPACK/platform). Centering removes the redundancy outright:
# rank 9 of 9, condition number ~33.
FORMULA = (
    f"cr(utilization, df={config.UTILIZATION_SPLINE_DF}, constraints='center') "
    f"+ cr(log_gri, df={config.GRI_SPLINE_DF}, constraints='center')"
)
REQUIRED_COLUMNS = ["utilization", "log_gri", "spread"]


class FairValueModel:
    def __init__(self, design_info, ols_result, residual_sigma):
        self.design_info = design_info
        self.ols_result = ols_result
        self.residual_sigma = residual_sigma

    def predict(self, df):
        valid = df[REQUIRED_COLUMNS].notna().all(axis=1)
        X = dmatrix(self.design_info, df.loc[valid], return_type="dataframe")
        fair_value = pd.Series(
            X.to_numpy() @ self.ols_result.params, index=df.loc[valid].index
        ).reindex(df.index)
        return fair_value


def fit_fair_value_model(df):
    train_mask = (df["date"] <= config.TRAIN_END) & df[REQUIRED_COLUMNS].notna().all(axis=1)
    train = df.loc[train_mask]

    if len(train) < 30:
        raise ValueError("Too few valid training observations.")

    X_train = dmatrix(FORMULA, train, return_type="dataframe")
    ols_result = sm.OLS(train["spread"].to_numpy(), X_train.to_numpy()).fit()

    model = FairValueModel(X_train.design_info, ols_result, residual_sigma=None)
    fair_value = model.predict(df)
    residual = df["spread"] - fair_value

    residual_sigma = residual.loc[train_mask].std(ddof=1)
    model.residual_sigma = residual_sigma

    return model, fair_value, residual


def apply_fair_value_model(df):
    """Fit on df's training window and attach fair_value/residual/zscore columns."""
    model, fair_value, residual = fit_fair_value_model(df)
    df = df.copy()
    df["fair_value"] = fair_value
    df["residual"] = residual
    df["zscore"] = residual / model.residual_sigma
    return df, model


if __name__ == "__main__":
    from load_data import load_daily_dataset

    df = load_daily_dataset()
    df, model = apply_fair_value_model(df)

    print(model.ols_result.summary().tables[1])
    print(f"\nTraining residual sigma: {model.residual_sigma:.4f}")

    train_mask = df["date"] <= config.TRAIN_END
    test_mask = ~train_mask
    for label, mask in [("Train", train_mask), ("Test", test_mask)]:
        r = df.loc[mask & df["residual"].notna(), "residual"]
        print(f"{label:6s} RMSE={np.sqrt(np.mean(r**2)):.3f}  MAE={r.abs().mean():.3f}  bias={r.mean():+.3f}  n={len(r)}")
