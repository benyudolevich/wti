import pandas as pd

from load_data import EXCEL_EPOCH, _build_release_dates, _parse_date_column


def test_excel_serial_dates_are_converted_from_excel_epoch():
    serial = 46216
    parsed = _parse_date_column(pd.Series([serial])).iloc[0]

    assert parsed == EXCEL_EPOCH + pd.Timedelta(days=serial)
    assert parsed.year >= 2025


def test_inventory_release_date_is_after_week_ending_observation():
    inventory_date = pd.Timestamp("2025-01-17")  # Friday week-ending date
    inventory = pd.DataFrame(
        {"date": [inventory_date], "inventory": [20_000.0]}
    )

    released = _build_release_dates(inventory)

    assert released.loc[0, "release_date"] == pd.Timestamp("2025-01-22")
    assert released.loc[0, "release_date"] > inventory_date
