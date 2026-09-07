# WTI Volatility and Inventory-Regime Research

Independent quantitative research on WTI crude-oil calendar spreads and volatility regimes.

This repository began as an inventory-squeeze model for the WTI front-two futures spread ('CL1 - CL2') and expanded into a separate, forward-looking volatility-regime research track. The emphasis is on causal data alignment, out-of-sample evaluation, and documenting where the evidence is weak.

> **Status:** research prototype, not an executable trading system or investment recommendation. The volatility track currently evaluates signal timing; it does not contain a historical WTI options-chain or options-P&L model.

## Research questions

1. Does Cushing inventory utilization provide a stable fair-value anchor for the WTI 'CL1 - CL2' calendar spread?
2. Does geopolitical-risk information improve the fair-value or risk-management model?
3. Can geopolitical-risk data improve forecasts of transitions into high crude-oil implied volatility?

## Current implementation

- **Data pipeline:** Reads WTI futures tenors, Cushing inventory, OVX, GRI, Brent, and the oil-specific 'GPR_OIL' index. Inventory is aligned using estimated release dates and backward as-of joins so observations are not used before becoming available.
- **Fair-value model:** Fits a training-period natural-cubic-spline model using Cushing utilization and 'log(GRI)', then evaluates residuals out of sample.
- **Spread strategy:** Uses residual z-score entries with partial-convergence exits, hard stops, time stops, and regime-aware sizing.
- **Volatility regime model:** Fits a two-state Markov-switching model to 'log(OVX)' with time-varying transition probabilities driven by 'GPR_OIL' by default. It produces 3-, 5-, 10-, and 15-trading-day-ahead regime forecasts using causal filtering.
- **Volatility timing rule:** Enters a binary 'LONG_VOL' state when the 10-day-ahead crisis probability reaches the configured threshold and exits using an entry-anchored OVX take-profit/stop-loss bracket. This is a timing diagnostic, not an options P&L backtest.
- **Research report:** 'report/main.tex' contains the model evolution, validation results, limitations, and open questions.

## Main conclusions

The project does not establish a robust standalone trading edge.

- The inventory relationship is useful as a structural research anchor, but the initial mean-reversion strategy was vulnerable to extended geopolitical moves.
- The early aggregate volatility-validation result overstated confidence because one sustained event was counted as multiple episodes and recent 'GPR_OIL' observations were stale.
- After consolidating episodes, the genuinely distinct out-of-sample sample is small: one clean lead-time result, one failure to discriminate timing, and one episode that was not testable because of stale input data.
- Further work is required before attaching a realistic options payoff model or claiming economic profitability.

## Repository map

| Path | Purpose |
| --- | --- |
| 'load_data.py' | Data cleaning, Excel-date handling, release-date alignment, and feature construction |
| 'model.py' | Spline fair-value model |
| 'regime.py' | GRI-only Markov regime classifier used by the spread strategy |
| 'vol_regime.py' | Forward-looking TVTP volatility-regime model |
| 'vol_strategy.py' | Timing-only long-volatility state machine |
| 'vol_joint_regime.py' | Exploratory learned level-plus-momentum HMM |
| 'strategy.py' | Calendar-spread positions, P&L, and trade log |
| 'diagnostics.py' | Fit, persistence, threshold, and trade diagnostics |
| 'run_backtest.py' | Entry point for the inventory/spread research path |
| 'run_vol_backtest.py' | Entry point for the volatility-regime timing path |
| 'validate_vol_regime.py' | Episode-level validation of forward forecasts |
| 'report/main.tex' | Technical write-up |
| 'Misc.ipynb' | Historical exploratory notebook for the original baseline |
| 'WTI_inventory_squeeze_handoff.md' | Historical design handoff; see the current modules above for the implemented pipeline |

## Reproducing the research

Use Python 3.10 or later in a clean virtual environment:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run the two main research paths:

    python run_backtest.py
    python run_vol_backtest.py

Validate the forward-looking volatility forecasts. The default driver is 'log_gpr_oil'; pass 'log_gri' to compare the older index:

    python validate_vol_regime.py
    python validate_vol_regime.py log_gri

Generate the volatility charts separately:

    python vol_charts.py

Long-running regime fits use the cache in 'cache/', which is intentionally ignored by Git and rebuilt when the training data or relevant configuration changes.

## Data and reproducibility notes

'Misc.xlsx' and 'ai_gpr_data_daily.csv' are research inputs, not generated outputs. Confirm the applicable source-data redistribution terms before making this repository public. The checked-in data and results are time-bounded; do not describe them as live or current without refreshing the inputs.

The training cutoff is centralized in 'config.py' as 'TRAIN_END'. Model parameters are fit only through that date and then frozen for out-of-sample filtering/backtesting.

## Limitations

- The spread uses generic Bloomberg-style 'CL1'/'CL2' series rather than a fully specified executable contract-pair and roll schedule.
- Inventory release dates are estimated from week-ending dates, with an override mechanism for known holiday exceptions.
- The volatility strategy has no historical option chain, implied-volatility surface, bid/ask, transaction costs, or realistic payoff model.
- The genuine out-of-sample episode count is too small to support a strong performance claim.
- The research code is modular and reproducible, but it is not production trading infrastructure.
