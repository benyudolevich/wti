"""
Centralized assumptions for the WTI inventory-squeeze model.

Change values here rather than inside load_data/model/strategy so every
module stays in sync.
"""

from pathlib import Path
import pandas as pd

FILE = Path("Misc.xlsx")
SHEET_NAME = 0

# Everything on/before this date is training data; everything after is the
# out-of-sample backtest.
TRAIN_END = pd.Timestamp("2025-01-13")

# Cushing working storage capacity used to compute utilization. Known
# limitation: this is held constant across the full 2010-2026 history even
# though actual capacity has changed over that period (see handoff notes).
# Plain utilization still outperformed a detrended/capacity-invariant
# alternative in backtesting, so it remains the working choice for now.
CAPACITY_KBBL = 78_500.0

# One NYMEX WTI contract represents 1,000 barrels.
CONTRACT_SIZE_BBL = 1_000.0

# Charged whenever position changes by one unit. Zero for now; set above
# zero to stress-test the strategy against transaction costs.
COST_PER_POSITION_CHANGE = 0.0

# Natural cubic spline degrees of freedom for each fair-value term.
UTILIZATION_SPLINE_DF = 4
GRI_SPLINE_DF = 4

# Entry/exit thresholds.
# ENTRY_Z lowered from 1.0 -> 0.75 to generate more signals in calm regimes
# (13 -> 18 out-of-sample breach episodes per the threshold-sensitivity scan).
ENTRY_Z = 0.75

# Partial-convergence exit: exit once |zscore| has shrunk back to this band,
# rather than waiting for an exact zero-crossing of the residual. Shortens
# average holding period so the strategy frees up to re-enter sooner.
EXIT_Z = 0.3

# Hard stop-loss: exit if the z-score moves this far in absolute terms
# against the position, regardless of entry level. This is what was missing
# when the March 2026 SHORT rode the war-fear spike to a -$11,400 drawdown.
STOP_Z_ABS = 3.0

# Time stop: force an exit after this many trading days regardless of
# convergence. Calibrated at ~2x the weekly-resampled residual half-life
# (~20 trading days), which reflects the slower structural convergence
# horizon rather than the near-meaningless <1-day daily AR(1) half-life.
MAX_HOLDING_DAYS = 40

# Number of Markov-switching states for the GRI regime model. 2 states
# (calm/crisis) turned out to classify 93.6% of the entire 2025-2026
# out-of-sample period as "crisis" -- it was picking up a general baseline
# shift in GRI relative to the 2010-2025 training average, not the specific
# acute escalation. 3 states (calm/elevated/crisis) lets the middle state
# absorb that baseline shift so the top state can isolate genuine spikes.
REGIME_K_STATES = 3

# Number of states for the forward-looking OVX/GRI vol-regime model
# (vol_regime.py). Neither 4 nor 3 states converged cleanly -- trying 2
# (calm/crisis) first as the simplest possible baseline to confirm the
# optimizer can converge at all on this TVTP specification before adding
# back state-count complexity.
VOL_REGIME_K_STATES = 2

# Default driver for the TVTP volatility-regime model. GRI remains available
# for comparison, but the current research path uses the oil-specific index.
VOL_REGIME_DRIVER = "log_gpr_oil"

# Regime-based POSITION SIZING rather than a binary entry block. A binary
# block turned out to suppress ~93.6% of the entire out-of-sample window,
# because GRI has been structurally elevated across nearly all of it, not
# just during the acute March-June spike -- a hard gate can't tell "more
# tense era" from "active war scare." Continuous sizing lets the strategy
# keep trading through the elevated-but-not-extreme periods (satisfying
# "more signals") while automatically shrinking size as crisis probability
# rises (satisfying "regime aware"), rather than forcing an all-or-nothing
# choice between the two.
#
# size_fraction = 1 - crisis_prob, i.e. full size when calm, near-zero size
# when the model is highly confident a crisis is live. Still fully blocked
# above CRISIS_PROB_HARD_BLOCK as a backstop for the most extreme readings.
CRISIS_PROB_HARD_BLOCK = 0.97

# Inventory release-date timing.
INVENTORY_DATES_ARE_RELEASE_DATES = False
RELEASE_DATE_OVERRIDES = {}

# -----------------------------------------------------------------------
# Volatility-timing strategy (vol_strategy.py)
# -----------------------------------------------------------------------
# Two-state (FLAT / LONG_VOL) strategy on top of the validated TVTP
# vol-regime forecast for ENTRY only. Confirmed design (2026-07-16): the
# original all-forecast version (entry AND exit both on crisis_prob_fwdN)
# exited too late relative to the actual peak -- trade 1 entered at
# OVX=55.26, peaked at 71.56, but the forecast-based exit didn't fire until
# OVX had round-tripped all the way down to 34.38, BELOW entry, giving back
# the entire gain. Replaced the exit with a level-based bracket instead: no
# forecast dependency once in the trade at all.
VOL_STRATEGY_ENTRY_HORIZON = 10
VOL_STRATEGY_ENTRY_THRESHOLD = 0.50

# Exit bracket, both legs measured against the OVX level AT ENTRY (not a
# trailing peak): take profit if OVX rises this fraction above entry;
# stop loss if OVX falls this fraction below entry. Symmetric 25/25 as a
# starting point -- not yet swept.
VOL_STRATEGY_TAKE_PROFIT_PCT = 0.25
VOL_STRATEGY_STOP_LOSS_PCT = 0.25
