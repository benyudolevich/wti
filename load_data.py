"""
Load and clean the WTI CL1/CL2 + Cushing inventory + OVX + GRI workbook.

Produces one daily dataframe with no-lookahead-safe inventory alignment.
OVX and GRI are treated as contemporaneously observable (market price /
same-day news index) and matched via backward as-of join to cover any gaps
between their native calendar and the CL1/CL2 trading calendar.
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

import config
from contract_calendar import days_to_front_expiry


EXCEL_EPOCH = pd.Timestamp("1899-12-30")


def _parse_date_column(raw_col):
    """Handle columns with a mix of real datetimes and bare Excel serial
    numbers (observed in gri_date: the most recent ~1,100 rows are stored
    as raw serials like 46216 rather than formatted dates, which pandas'
    default parser silently misreads as 1970-01-01-plus-nanoseconds).

    Checks Python types directly rather than pd.to_numeric, which will
    happily (and wrongly) coerce real datetime objects into huge integers
    too.
    """
    is_bare_number = raw_col.map(lambda x: isinstance(x, (int, float, np.integer, np.floating)))
    parsed = pd.to_datetime(raw_col.where(~is_bare_number), errors="coerce")
    serials = pd.to_numeric(raw_col.where(is_bare_number), errors="coerce")
    parsed = parsed.where(~is_bare_number, EXCEL_EPOCH + pd.to_timedelta(serials, unit="D"))
    # Force a single, consistent datetime64 resolution: pandas can otherwise
    # infer different resolutions (e.g. "us" vs "ns") for different columns
    # depending on the mix of real datetime objects vs bare serials in the
    # source data, which then makes merge_asof fail with a dtype mismatch.
    return parsed.astype("datetime64[ns]")


def _clean_pair(raw, date_col, value_col, name):
    d = raw[[date_col, value_col]].copy()
    d["date"] = _parse_date_column(d[date_col])
    d[name] = pd.to_numeric(d[value_col], errors="coerce")
    return (
        d[["date", name]]
        .dropna()
        .drop_duplicates(subset="date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _build_release_dates(inventory):
    inventory = inventory.rename(columns={"date": "inventory_date"})

    if config.INVENTORY_DATES_ARE_RELEASE_DATES:
        inventory["release_date"] = inventory["inventory_date"]
    else:
        us_business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        inventory["release_date"] = inventory["inventory_date"].map(
            lambda d: d + 3 * us_business_day
        )

    for inventory_date, release_date in config.RELEASE_DATE_OVERRIDES.items():
        inventory.loc[
            inventory["inventory_date"] == pd.Timestamp(inventory_date),
            "release_date",
        ] = pd.Timestamp(release_date)

    return (
        inventory.sort_values("release_date")
        .drop_duplicates(subset="release_date", keep="last")
        .reset_index(drop=True)
    )


def load_daily_dataset():
    if not config.FILE.exists():
        raise FileNotFoundError(f"Could not find {config.FILE.resolve()}")

    raw = pd.read_excel(config.FILE, sheet_name=config.SHEET_NAME)
    raw = raw.iloc[:, :20].copy()
    raw.columns = [
        "cl1_date", "cl1", "cl2_date", "cl2", "cl3_date", "cl3",
        "cl4_date", "cl4", "cl5_date", "cl5", "cl6_date", "cl6",
        "inv_date", "inventory", "ovx_date", "ovx", "gri_date", "gri",
        "brent_date", "brent",
    ]

    cl1 = _clean_pair(raw, "cl1_date", "cl1", "cl1")
    cl2 = _clean_pair(raw, "cl2_date", "cl2", "cl2")
    cl3 = _clean_pair(raw, "cl3_date", "cl3", "cl3")
    cl4 = _clean_pair(raw, "cl4_date", "cl4", "cl4")
    cl5 = _clean_pair(raw, "cl5_date", "cl5", "cl5")
    cl6 = _clean_pair(raw, "cl6_date", "cl6", "cl6")
    ovx = _clean_pair(raw, "ovx_date", "ovx", "ovx")
    gri = _clean_pair(raw, "gri_date", "gri", "gri")
    brent = _clean_pair(raw, "brent_date", "brent", "brent")
    inventory = _clean_pair(raw, "inv_date", "inventory", "inventory")
    inventory = _build_release_dates(inventory)

    # CL1 is the trading-calendar anchor. CL2 has full matching coverage
    # (inner join, as before). CL3-CL6 are LEFT-joined on exact date match
    # (not as-of): CL4/CL5 in particular are very sparsely populated in the
    # source data right now (94 and 1 non-null rows respectively, vs ~4930
    # for the others) -- an inner join across all six would collapse the
    # whole dataset to ~1 row. Left join preserves CL1/CL2 as the core
    # series and leaves the sparser tenors NaN wherever unavailable, rather
    # than fabricating a stale carry-forward price for a settlement value.
    prices = cl1.merge(cl2, on="date", how="inner", validate="one_to_one")
    for other in (cl3, cl4, cl5, cl6):
        prices = prices.merge(other, on="date", how="left", validate="one_to_one")
    prices["spread"] = prices["cl1"] - prices["cl2"]
    prices = prices.sort_values("date").reset_index(drop=True)

    df = pd.merge_asof(
        prices.sort_values("date"),
        inventory[["inventory_date", "release_date", "inventory"]].sort_values("release_date"),
        left_on="date",
        right_on="release_date",
        direction="backward",
        allow_exact_matches=True,
    )

    df = pd.merge_asof(df.sort_values("date"), ovx.sort_values("date"), on="date", direction="backward")
    df = pd.merge_asof(df.sort_values("date"), gri.sort_values("date"), on="date", direction="backward")
    df = pd.merge_asof(df.sort_values("date"), brent.sort_values("date"), on="date", direction="backward")

    # Oil-specific AI-GPR index (Iacoviello & Tong 2026) -- validated to beat
    # GRI head-to-head in the fair-value model. Previously loaded ad-hoc in
    # three separate scripts; now a core part of the pipeline.
    gpr = pd.read_csv("ai_gpr_data_daily.csv", parse_dates=["Date"]).rename(columns={"Date": "date"})
    gpr["date"] = gpr["date"].astype("datetime64[ns]")
    gpr = gpr[["date", "GPR_OIL", "GPR_OIL_MiddleEast"]].sort_values("date")
    df = pd.merge_asof(df.sort_values("date"), gpr, on="date", direction="backward")

    df["utilization"] = df["inventory"] / config.CAPACITY_KBBL
    df.loc[~df["utilization"].between(0.01, 0.99), "utilization"] = np.nan

    df["log_ovx"] = np.log(df["ovx"])
    df["log_gri"] = np.log(df["gri"])
    # log1p, not log: GPR_OIL has heavy zero-mass (~33% aggregate, ~53% for
    # the Middle-East-only series), unlike GRI which is never exactly zero.
    df["log_gpr_oil"] = np.log1p(df["GPR_OIL"])
    df["log_gpr_oil_me"] = np.log1p(df["GPR_OIL_MiddleEast"])

    # Annualized realized volatility of the front-month contract, for
    # comparing against OVX (implied) -- e.g. did realized vol actually
    # expand during a period the vol-regime model flagged as high risk.
    log_ret = np.log(df["cl1"] / df["cl1"].shift(1))
    df["realized_vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252) * 100

    # Curve convexity ("carry of the carry"): CV = S12 - S23 = CL1 - 2*CL2 + CL3.
    # Kept separate from the fair-value regression (model.py) since it's
    # constructed from the same traded prices we're modeling -- folding it
    # directly into the structural regression would risk circularity.
    # NaN wherever CL3 is unavailable (CL3 has ~79% coverage, not full).
    df["spread23"] = df["cl2"] - df["cl3"]
    df["convexity"] = df["spread"] - df["spread23"]

    # Roll-pressure proxy: business days to front-month expiry, from the
    # documented CME WTI calendar rule (see contract_calendar.py).
    df["days_to_expiry"] = days_to_front_expiry(df["date"])

    # OVX acceleration: smoothed 5-day log-change of a 3-day trailing
    # average of OVX. Continuous, NOT bucketed into percentile phases --
    # the earlier vol_momentum.py attempt classified this into hand-defined
    # buckets and was found to be a rule, not a learned signal; here it's
    # just a raw candidate covariate for the gate to weigh on its own.
    # Smoothing the input first (not the raw OVX) is still needed: the
    # un-smoothed version was too choppy to be usable even as a continuous
    # feature (confirmed empirically when vol_momentum.py was built).
    ovx_smooth = df["ovx"].rolling(3, min_periods=1).mean()
    df["ovx_accel"] = np.log(ovx_smooth) - np.log(ovx_smooth.shift(5))

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_daily_dataset()
    print(f"Loaded {len(df):,} daily rows, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    print(df.tail(5))
