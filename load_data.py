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
    return parsed


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
    raw = raw.iloc[:, :11].copy()
    raw.columns = [
        "cl1_date", "cl1", "cl2_date", "cl2", "spread_excel",
        "inv_date", "inventory", "ovx_date", "ovx", "gri_date", "gri",
    ]

    cl1 = _clean_pair(raw, "cl1_date", "cl1", "cl1")
    cl2 = _clean_pair(raw, "cl2_date", "cl2", "cl2")
    ovx = _clean_pair(raw, "ovx_date", "ovx", "ovx")
    gri = _clean_pair(raw, "gri_date", "gri", "gri")
    inventory = _clean_pair(raw, "inv_date", "inventory", "inventory")
    inventory = _build_release_dates(inventory)

    prices = cl1.merge(cl2, on="date", how="inner", validate="one_to_one")
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

    df["utilization"] = df["inventory"] / config.CAPACITY_KBBL
    df.loc[~df["utilization"].between(0.01, 0.99), "utilization"] = np.nan

    df["log_ovx"] = np.log(df["ovx"])
    df["log_gri"] = np.log(df["gri"])

    # Annualized realized volatility of the front-month contract, for
    # comparing against OVX (implied) -- e.g. did realized vol actually
    # expand during a period the vol-regime model flagged as high risk.
    log_ret = np.log(df["cl1"] / df["cl1"].shift(1))
    df["realized_vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252) * 100

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_daily_dataset()
    print(f"Loaded {len(df):,} daily rows, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    print(df.tail(5))
