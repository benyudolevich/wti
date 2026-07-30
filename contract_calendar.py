"""
WTI (NYMEX/CME "CL") contract expiry calendar.

Standard, publicly documented CME rule: trading terminates on the 3rd
business day prior to the 25th calendar day of the month preceding the
delivery month; if the 25th is not itself a business day, roll back to the
business day immediately preceding it first.

Used to compute a genuine "days to front-month expiry" feature without
needing to reverse-engineer Bloomberg's specific generic-series roll
convention from price data (checked -- no clean signature recoverable from
CL1/CL2 price jumps, especially with real 2026 volatility in the way).
Bloomberg's generic front-month series typically rolls at or near actual
contract expiry, so this is a close, and fully verifiable, proxy.

Does NOT cover S&P GSCI / Bloomberg Commodity Index roll windows -- those
are third-party index methodologies, not an exchange rule, and are out of
scope for now per the "check CL1-CL6 Bloomberg expiry roll schedule"
decision.
"""

import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def wti_expiry(delivery_year, delivery_month):
    if delivery_month == 1:
        ref_year, ref_month = delivery_year - 1, 12
    else:
        ref_year, ref_month = delivery_year, delivery_month - 1
    twenty_fifth = pd.Timestamp(ref_year, ref_month, 25)
    reference_day = US_BUSINESS_DAY.rollback(twenty_fifth)
    return reference_day - 3 * US_BUSINESS_DAY


def days_to_front_expiry(dates):
    """For each date, business days until the currently-active front-month
    contract's expiry (the nearest WTI expiry that hasn't happened yet)."""
    dates = pd.DatetimeIndex(dates)
    result = pd.Series(index=dates, dtype=float)

    for date in dates.unique():
        # Check delivery months from date's own month out a few months ahead;
        # the front contract is whichever has the nearest expiry >= date.
        candidates = []
        for offset in range(0, 4):
            month = date.month + offset
            year = date.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            candidates.append(wti_expiry(year, month))
        upcoming = [e for e in candidates if e >= date]
        expiry = min(upcoming) if upcoming else min(candidates)
        result.loc[date] = len(pd.bdate_range(date, expiry, freq=US_BUSINESS_DAY)) - 1

    return result.reindex(dates).to_numpy()


if __name__ == "__main__":
    test_dates = pd.to_datetime(["2026-07-01", "2026-07-15", "2026-07-20", "2026-07-22", "2026-08-01"])
    for d, dte in zip(test_dates, days_to_front_expiry(test_dates)):
        print(f"{d.date()}: {dte:.0f} business days to front expiry")
