# WTI Inventory-Squeeze Model — Claude Code Handoff

## Project goal

Build and evaluate a realistic WTI calendar-spread strategy based on the nonlinear relationship between:

- Cushing crude inventory utilization
- The nearby WTI spread: `CL1 - CL2`

The original research replicated an inventory “squeeze” curve: spreads tend to move sharply into backwardation at very low inventories and sharply into contango at very high inventories.

Ignore the previous geopolitical-impact-index idea entirely.

The end goal is **not merely to fit the historical curve**. It is to determine whether inventory-based fair value can support a credible, out-of-sample trading strategy with useful diagnostics and charts.

---

## Current files

### `misc.xlsx`

Located in the same folder as the notebook/code.

The workbook contains one sheet with data from roughly `2016-01-01` through the present, sorted in ascending date order.

Current column structure:

| Excel column | Meaning | Frequency |
|---|---|---|
| A | CL1 date | Daily trading days |
| B | CL1 settlement price | Daily |
| C | CL2 date | Daily trading days |
| D | CL2 settlement price | Daily |
| E | Spread = CL1 - CL2 | Daily |
| F | Cushing inventory observation date | Weekly |
| G | Cushing inventory, thousand barrels | Weekly |

Important:

- Columns A–E are daily.
- Columns F–G are weekly and shorter than the daily series.
- Inventory values must be aligned to daily oil dates using a backward/as-of join and then forward-filled.
- No inventory value may be used before it was publicly available.

---

## Inventory timing

The dates in column F appear to be EIA week-ending dates, usually Fridays, not publication dates.

For no-lookahead alignment:

1. Convert each inventory observation date into its release date.
2. Normal assumption: release occurs three U.S. business days later, usually Wednesday.
3. Holiday weeks may require manual release-date overrides.
4. For each daily oil date, use the latest inventory observation whose release date is less than or equal to that oil date.

Do not join weekly inventory directly on its week-ending date.

---

## Fixed assumptions

```python
TRAIN_END = "2025-01-13"
CAPACITY_KBBL = 78_500.0
CONTRACT_SIZE_BBL = 1_000.0
```

Inventory utilization:

```python
utilization = inventory_kbbl / 78_500
```

The period through `2025-01-13` is the training sample.

The period after `2025-01-13` is the out-of-sample test.

---

## Baseline squeeze model

Current baseline functional form:

```text
fair_value_spread
    = beta_0
    + beta_1 / utilization
    + beta_2 / (1 - utilization)
```

Where:

- `1 / utilization` captures the low-inventory squeeze
- `1 / (1 - utilization)` captures the full-storage squeeze
- Dependent variable is `CL1 - CL2`

Fit coefficients using only the training sample.

Then calculate:

```python
residual = actual_spread - fair_value_spread
zscore = residual / training_residual_std
```

The original notebook/report primarily established this structural relationship. It did not prove that residuals are short-horizon mean reverting.

---

## Current strategy attempt

The first strategy tested was:

- Enter long spread when `zscore <= -1.5`
- Enter short spread when `zscore >= +1.5`
- Exit only when residual crosses zero
- One spread position at a time
- Daily close-to-close P&L
- Signal formed at today’s settlement; position earns P&L beginning next trading day
- No transaction costs initially

Position convention:

```text
+1 = long CL1, short CL2
-1 = short CL1, long CL2
 0 = flat
```

Daily gross P&L:

```python
daily_pnl = previous_day_position * spread.diff() * 1_000
```

This produced only about two trades over roughly 1.5 years.

That result is likely due to model/strategy design rather than a coding bug.

---

## Why the current strategy is too inactive

Likely causes:

1. Training residual volatility includes crisis periods such as 2020 and 2022, making `1.5 * sigma` a very wide entry threshold.
2. Residuals can remain on one side of zero for months, so zero-crossing exits create very long holds.
3. Extreme residual days cluster into one episode, not many independent trades.
4. Inventory utilization alone is a structural anchor, not necessarily a complete tradable fair-value model.
5. Daily spread observations are matched against a weekly fundamental input, inflating unexplained daily residual noise.
6. `CL1` and `CL2` are generic Bloomberg futures, so the underlying contract pair changes around roll dates.
7. The model currently ignores days to expiry, curve seasonality, and prompt-contract roll mechanics.

Do not simply lower the threshold to manufacture trades without first diagnosing these issues.

---

## Required diagnostics

Implement and display:

- Training residual standard deviation
- Out-of-sample residual standard deviation
- Mean out-of-sample residual and z-score
- Minimum and maximum z-score
- Number of days beyond each threshold
- Number of distinct threshold-breach episodes
- Percentage of days in a position
- Average and maximum holding period
- Residual AR(1) coefficient
- Approximate residual half-life
- Trade count, win rate, average trade P&L
- Total P&L, Sharpe, and maximum drawdown
- Sensitivity by entry and exit threshold

Also inspect whether the out-of-sample residual distribution is structurally shifted away from zero.

---

## Better strategy-development path

Prioritize the following in order.

### 1. Separate model validation from trading

First test whether the frozen pre-2025 squeeze curve continues to explain post-2025 spread levels.

Report:

- Out-of-sample RMSE
- MAE
- Correlation
- Residual bias
- Stability of fitted coefficients through rolling/expanding regressions

### 2. Re-estimate at weekly frequency

Create a weekly modeling dataset using one spread observation per inventory release.

Test alternatives such as:

- Settlement on release day
- Settlement immediately after release
- Weekly average spread after release

This avoids counting the same weekly inventory value five times against daily spread noise.

### 3. Improve exits

Test partial-convergence exits rather than requiring a zero crossing.

Examples:

```text
Entry: |z| >= 1.25 or 1.50
Exit:  |z| <= 0.25 or 0.50
```

Also test:

- Maximum holding period, e.g. 10, 20, or 30 trading days
- Stop-loss threshold, e.g. entry z-score plus another 1.0 sigma adverse move
- Time-based exit before the generic roll

### 4. Add contract-timing information

At minimum, add:

- Days to CL1 expiry
- Days to generic roll
- Calendar month / seasonality
- Interaction between utilization and days to expiry

A better specification may be:

```text
spread ~ squeeze_terms
       + days_to_expiry
       + utilization × days_to_expiry
       + month fixed effects
```

### 5. Move from generic to explicit contracts

Ultimately, use explicit monthly WTI contracts rather than only `CL1` and `CL2`.

Track a fixed pair such as:

```text
Feb-Mar
Mar-Apr
Apr-May
```

Define:

- Exact entry contract pair
- Exact exit date
- Roll convention
- Liquidity filter
- Transaction costs

Generic `CL1-CL2` is acceptable for exploratory structural analysis but imperfect for executable P&L.

---

## Required charts

Generate clean charts for:

1. Actual spread versus Cushing utilization, with fitted squeeze curve
2. Training versus out-of-sample scatter
3. Actual spread versus modeled fair value over time
4. Residual over time
5. Z-score with entry and exit bands
6. Position over time
7. Cumulative P&L
8. Drawdown
9. Trade-by-trade P&L
10. Sensitivity heatmap for entry/exit thresholds
11. Residual distribution: training versus out of sample
12. Residual persistence / autocorrelation

Charts should use `matplotlib`, one chart per figure, no seaborn.

---

## Preferred code structure

Keep the project modular rather than one giant notebook cell.

Suggested files:

```text
load_data.py
model.py
strategy.py
diagnostics.py
charts.py
run_backtest.py
```

Suggested responsibilities:

- `load_data.py`: read Excel, clean columns, release-date logic, no-lookahead as-of join
- `model.py`: fit squeeze model, calculate fair value and residuals
- `strategy.py`: generate positions, P&L, and trade log
- `diagnostics.py`: validation metrics, residual tests, threshold sweeps
- `charts.py`: all plots
- `run_backtest.py`: configuration and execution

Keep assumptions centralized in a config section.

---

## Deliverables

Produce:

1. A reproducible baseline backtest
2. A diagnostic report explaining why trade count is low
3. A weekly-frequency model comparison
4. Entry/exit sensitivity results
5. Improved strategy variants
6. Exported daily results and trade log to Excel or CSV
7. A concise conclusion on whether inventory squeeze is:
   - useful as a structural explanatory model,
   - useful as a regime filter,
   - or strong enough to generate standalone trades

The main question is:

> Does Cushing utilization provide a stable enough fair-value anchor for WTI calendar spreads to improve trade timing out of sample, after properly accounting for release timing, expiry, roll mechanics, and residual persistence?
