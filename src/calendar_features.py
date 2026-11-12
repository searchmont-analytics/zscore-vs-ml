"""Business calendar for an Ontario gaming operator.

Used two ways:
  * by the data generator, to shape "true" demand (weekends, holidays, events);
  * by the ML detector, as *generic* features (is_holiday, days_to_holiday, ...).

The detector never sees the generator's multipliers -- only the same calendar
facts an analyst would know (what day it is, whether it is a long weekend, how
big the lottery jackpot is). It has to learn the size of each effect from data.
"""
from __future__ import annotations

import datetime as dt

import holidays
import numpy as np
import pandas as pd

# Ontario statutory holidays (+ Civic Holiday, which most businesses observe).
_DROP = {"Easter Monday", "Remembrance Day"}


def ontario_holidays(years) -> dict[dt.date, str]:
    h = holidays.Canada(subdiv="ON", years=years, categories=("public", "optional"))
    return {d: n for d, n in h.items() if n not in _DROP}


def super_bowl_sunday(year: int) -> dt.date:
    """Second Sunday of February (current NFL schedule)."""
    d = dt.date(year, 2, 1)
    first_sun = d + dt.timedelta(days=(6 - d.weekday()) % 7)
    return first_sun + dt.timedelta(days=7)


def grey_cup_sunday(year: int) -> dt.date:
    """Approximation: third Sunday of November."""
    d = dt.date(year, 11, 1)
    first_sun = d + dt.timedelta(days=(6 - d.weekday()) % 7)
    return first_sun + dt.timedelta(days=14)


def special_events(years) -> dict[dt.date, str]:
    ev = {}
    for y in years:
        ev[dt.date(y, 12, 24)] = "Christmas Eve"
        ev[dt.date(y, 12, 31)] = "New Year's Eve"
        ev[super_bowl_sunday(y)] = "Super Bowl Sunday"
        ev[grey_cup_sunday(y)] = "Grey Cup Sunday"
    return ev


def build_calendar(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    years = sorted({d.year for d in dates} | {dates[-1].year + 1})
    hol = ontario_holidays(years)
    ev = special_events(years)
    hol_dates = np.array(sorted(hol), dtype="datetime64[D]")
    day_vals = dates.values.astype("datetime64[D]")

    cal = pd.DataFrame(index=dates)
    cal.index.name = "date"
    cal["dow"] = cal.index.dayofweek  # 0 = Mon
    cal["is_weekend"] = cal["dow"].isin([4, 5]).astype(int)  # Fri/Sat nights
    cal["holiday_name"] = [hol.get(d.date(), "") for d in dates]
    cal["event_name"] = [ev.get(d.date(), "") for d in dates]
    cal["is_holiday"] = (cal["holiday_name"] != "").astype(int)
    cal["is_event"] = (cal["event_name"] != "").astype(int)

    # Long-weekend eve: the Fri/Sat/Sun before a Monday holiday (or Thu before Good Friday).
    nxt = np.searchsorted(hol_dates, day_vals)
    prev = nxt - 1
    nxt_d = hol_dates[np.clip(nxt, 0, len(hol_dates) - 1)]
    prv_d = hol_dates[np.clip(prev, 0, len(hol_dates) - 1)]
    days_to = (nxt_d - day_vals).astype("timedelta64[D]").astype(int)
    days_since = (day_vals - prv_d).astype("timedelta64[D]").astype(int)
    cal["days_to_holiday"] = np.clip(days_to, 0, 14)
    cal["days_since_holiday"] = np.clip(days_since, 0, 14)
    cal["is_long_weekend"] = (
        ((cal["days_to_holiday"].between(1, 3)) & cal["dow"].isin([4, 5, 6]))
        | ((cal["days_to_holiday"] == 1) & (cal["dow"] == 3))
    ).astype(int)

    cal["month"] = cal.index.month
    cal["day_of_month"] = cal.index.day
    cal["day_of_year"] = cal.index.dayofyear
    last_day = cal.index.days_in_month
    cal["is_payday"] = ((cal["day_of_month"] == 15) | (cal["day_of_month"] >= last_day - 1)).astype(int)
    cal["is_xmas_week"] = ((cal["month"] == 12) & (cal["day_of_month"] >= 20)).astype(int)
    cal["is_draw_day"] = cal["dow"].isin([1, 4]).astype(int)  # Tue / Fri lottery draws
    return cal


def jackpot_series(cal: pd.DataFrame, seed: int = 7) -> pd.Series:
    """Simulated national lottery jackpot ($M) -- rolls over until won."""
    rng = np.random.default_rng(seed)
    jp, out = 10.0, []
    for is_draw in cal["is_draw_day"].values:
        out.append(jp)
        if is_draw:
            # Larger jackpots sell more tickets, so they are more likely to be won.
            p_win = 0.12 + 0.004 * jp
            jp = 10.0 if rng.random() < p_win else min(jp + rng.uniform(3, 8), 80.0)
    return pd.Series(out, index=cal.index, name="jackpot_musd")
