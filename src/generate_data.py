"""Generate synthetic data that behaves like a casino / iGaming operator's feeds.

Outputs (in data/):
  tombstone_feed.csv.gz   row-level daily customer master feed (new + updated customers)
  activity_daily.csv      daily play summary: record volume + total spend
  incidents.csv           ground truth -- the DQ incidents we deliberately injected

Normal behaviour has strong, *legitimate* seasonality: Fri/Sat peaks, long weekends,
New Year's Eve, Super Bowl Sunday, pay-days, summer, big lottery jackpots.
Incidents are the things a DQ monitor *should* catch: partial loads, duplicate loads,
unit errors, missing feeds, schema breaks that null out a column, slow drift.
"""
from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_features import build_calendar, jackpot_series

START, END = "2024-09-01", "2026-10-31"
SEED = 42
DATA = Path(__file__).resolve().parent.parent / "data"

# --------------------------------------------------------------------------------------
# "True" demand multipliers -- known only to the generator.
# --------------------------------------------------------------------------------------
DOW_MULT = {0: 0.86, 1: 0.84, 2: 0.88, 3: 0.97, 4: 1.28, 5: 1.45, 6: 1.12}
HOLIDAY_MULT = {
    "New Year's Day": 1.10, "Family Day": 1.30, "Good Friday": 1.25, "Victoria Day": 1.30,
    "Canada Day": 1.35, "Civic Holiday": 1.30, "Labour Day": 1.25, "Labor Day": 1.25,
    "Thanksgiving Day": 1.15, "Christmas Day": 0.62, "Boxing Day": 1.35,
}
EVENT_MULT = {"Christmas Eve": 0.80, "New Year's Eve": 1.55, "Super Bowl Sunday": 1.50, "Grey Cup Sunday": 1.20}


def demand_index(cal: pd.DataFrame, jackpot: pd.Series) -> pd.Series:
    t = np.arange(len(cal))
    m = np.array([DOW_MULT[d] for d in cal["dow"]])
    m *= np.array([HOLIDAY_MULT.get(h, 1.0) for h in cal["holiday_name"]])
    m *= np.array([EVENT_MULT.get(e, 1.0) for e in cal["event_name"]])
    m *= np.where(cal["is_long_weekend"] == 1, 1.15, 1.0)
    m *= np.where(cal["is_payday"] == 1, 1.07, 1.0)
    m *= 1 + 0.06 * np.sin(2 * np.pi * (cal["day_of_year"] - 100) / 365.25)  # summer high, Jan low
    m *= 1 + 0.10 * (t / 365)  # ~10 %/yr growth
    # Big jackpots pull extra traffic on draw days and the day before.
    jp_boost = 1 + 0.0035 * np.maximum(jackpot.values - 30, 0)
    m *= np.where((cal["is_draw_day"] == 1) | (cal["is_draw_day"].shift(-1, fill_value=0) == 1), jp_boost, 1.0)
    return pd.Series(m, index=cal.index)


# --------------------------------------------------------------------------------------
# Incidents (ground truth). Each one touches one or more metrics on specific days.
# --------------------------------------------------------------------------------------
INCIDENTS = [
    # date(s), metric(s), kind, description
    ("2025-02-15", "2025-02-15", ["activity_records", "daily_spend"], "partial_load", "Family Day long-weekend Saturday: 1 of 3 casino feeds failed to land (-35%)"),
    ("2025-03-11", "2025-03-11", ["activity_records"], "duplicate_load", "Tuesday batch loaded twice (x2 records, spend de-duplicated downstream)"),
    ("2025-05-19", "2025-05-19", ["daily_spend"], "partial_load", "Victoria Day: online channel spend missing (-30%)"),
    ("2025-06-03", "2025-06-05", ["email_null_pct"], "schema_break", "Upstream CRM renamed email field -> 30% nulls for 3 days"),
    ("2025-08-02", "2025-08-02", ["activity_records", "daily_spend"], "partial_load", "Civic Holiday Saturday: -30% (late file, never back-filled)"),
    ("2025-09-20", "2025-09-20", ["daily_spend"], "unit_error", "Spend reported in cents instead of dollars (x100)"),
    ("2025-11-07", "2025-11-07", ["duplicate_id_pct", "tombstone_records"], "duplicate_load", "Tombstone batch re-sent: duplicate customer IDs + 2x rows"),
    ("2025-12-31", "2025-12-31", ["activity_records", "daily_spend"], "partial_load", "New Year's Eve: casino feed truncated at 20:00 (-30%)"),
    ("2026-01-14", "2026-01-14", ["activity_records", "daily_spend", "tombstone_records"], "missing_feed", "All feeds missing (scheduler outage) -- near-zero rows"),
    ("2026-03-10", "2026-03-12", ["invalid_postal_pct"], "format_change", "New sign-up form stopped enforcing postal-code format"),
    ("2026-04-04", "2026-04-04", ["activity_records"], "partial_load", "Easter Saturday: -35% records"),
    ("2026-05-26", "2026-05-26", ["daily_spend"], "spike", "Tuesday spend +45% (adjustment entries posted twice)"),
    ("2026-07-06", "2026-07-19", ["daily_spend"], "drift", "Slow drift: a promo-credit mapping bug erodes spend ~2%/day for 2 weeks"),
    ("2026-08-22", "2026-08-22", ["tombstone_records"], "partial_load", "Saturday tombstone feed -55% (one region's CRM extract failed)"),
    ("2026-10-12", "2026-10-12", ["underage_pct"], "validation_gap", "Thanksgiving: DOB validation bypassed, under-19 records slip in"),
]


def incidents_frame() -> pd.DataFrame:
    rows = []
    for i, (s, e, metrics, kind, desc) in enumerate(INCIDENTS, 1):
        for d in pd.date_range(s, e):
            for m in metrics:
                rows.append({"incident_id": i, "date": d, "metric": m, "kind": kind, "description": desc})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Daily activity summary (record volume + spend)
# --------------------------------------------------------------------------------------
def make_activity(cal, demand, rng) -> pd.DataFrame:
    n = len(cal)
    base_records = 182_000
    base_spend = 9_400_000.0
    common = rng.normal(0, 0.035, n)  # shared day-level shock (weather, news)
    records = base_records * demand.values * np.exp(common + rng.normal(0, 0.025, n))
    avg_bet = 51.6 * (1 + 0.04 * cal["is_weekend"].values) * np.exp(rng.normal(0, 0.03, n))
    spend = records * avg_bet * np.exp(rng.normal(0, 0.02, n))
    df = pd.DataFrame({"activity_records": records.round().astype(int), "daily_spend": spend.round(2)}, index=cal.index)

    for i, (s, e, metrics, kind, _) in enumerate(INCIDENTS, 1):
        idx = pd.date_range(s, e)
        if kind == "drift" and "daily_spend" in metrics:
            df.loc[idx, "daily_spend"] *= (1 - 0.02) ** np.arange(1, len(idx) + 1)
            continue
        for m in metrics:
            if m not in df.columns:
                continue
            factor = {
                "partial_load": 0.65 if i in (1, 11) else 0.70,
                "duplicate_load": 2.0, "unit_error": 100.0, "missing_feed": 0.002, "spike": 1.45,
            }[kind]
            df.loc[idx, m] = (df.loc[idx, m] * factor).round(2 if m == "daily_spend" else 0)
    df["activity_records"] = df["activity_records"].astype(int)
    df.index.name = "date"
    return df


# --------------------------------------------------------------------------------------
# Row-level tombstone feed
# --------------------------------------------------------------------------------------
FIRST = ["Liam", "Olivia", "Noah", "Emma", "Ethan", "Ava", "Lucas", "Mia", "Aarav", "Priya", "Wei", "Mei",
         "Mohammed", "Fatima", "Jean", "Chloe", "Daniel", "Sofia", "Raj", "Anika", "Owen", "Grace"]
LAST = ["Smith", "Brown", "Tremblay", "Martin", "Roy", "Wilson", "Patel", "Singh", "Nguyen", "Chen", "Li",
        "MacDonald", "Gagnon", "Taylor", "Campbell", "Anderson", "Khan", "Ali", "Kim", "Lee", "Leblanc"]
FSA_LETTERS = "KLMNP"  # Ontario postal prefixes


def _postal(rng, n):
    L = np.array(list("ABCEGHJKLMNPRSTVWXYZ"))
    D = np.array(list(string.digits))
    p = (rng.choice(list(FSA_LETTERS), n).astype(object) + rng.choice(D, n) + rng.choice(L, n) + " "
         + rng.choice(D, n) + rng.choice(L, n) + rng.choice(D, n))
    return p


def make_tombstone(cal, demand, rng) -> pd.DataFrame:
    # Sign-ups/profile updates follow the same rhythm, a little damped.
    base = 1_150
    vol = base * demand.values ** 0.8 * np.exp(rng.normal(0, 0.05, len(cal)))
    vol = vol.round().astype(int)

    inc = incidents_frame()
    tomb_vol_factor = {"duplicate_load": 2.0, "missing_feed": 0.0, "partial_load": 0.45}
    for _, r in inc[inc.metric == "tombstone_records"].iterrows():
        k = cal.index.get_loc(r.date)
        vol[k] = int(vol[k] * tomb_vol_factor[r.kind])

    def rates_for(day, metric, normal):
        hit = inc[(inc.date == day) & (inc.metric == metric)]
        return normal if hit.empty else {"email_null_pct": 0.30, "invalid_postal_pct": 0.22,
                                         "duplicate_id_pct": 0.48, "underage_pct": 0.035}[metric]

    frames, next_id = [], 1_000_000
    for k, day in enumerate(cal.index):
        n = vol[k]
        if n == 0:
            continue
        is_new = rng.random(n) < 0.55
        cid = np.where(is_new, np.arange(next_id, next_id + n), rng.integers(100_000, next_id, n))
        next_id += n
        dup_rate = rates_for(day, "duplicate_id_pct", 0.002)
        dup_mask = rng.random(n) < dup_rate
        if dup_mask.any():
            cid[dup_mask] = rng.choice(cid[~dup_mask], dup_mask.sum())

        # DOB: adults 19-85, small % under 19 (should be rejected upstream)
        age = rng.uniform(19, 85, n)
        u_rate = rates_for(day, "underage_pct", 0.0008)
        under = rng.random(n) < u_rate
        age[under] = rng.uniform(14, 18.9, under.sum())
        dob = (day - pd.to_timedelta((age * 365.25).astype(int), unit="D")).strftime("%Y-%m-%d")

        fn = rng.choice(FIRST, n)
        ln = rng.choice(LAST, n)
        email = np.char.lower(np.char.add(np.char.add(fn.astype(str), "."), ln.astype(str))).astype(object) \
            + rng.integers(1, 999, n).astype(str).astype(object) + "@example.com"
        email[rng.random(n) < rates_for(day, "email_null_pct", 0.015)] = None

        postal = _postal(rng, n)
        bad = rng.random(n) < rates_for(day, "invalid_postal_pct", 0.006)
        postal[bad] = rng.choice(["", "N/A", "00000", "M5V2T6X", "12345"], bad.sum())

        phone = np.array([f"{a}-{b:03d}-{c:04d}" for a, b, c in zip(
            rng.choice(["416", "647", "905", "613", "519", "705", "289"], n),
            rng.integers(200, 999, n), rng.integers(0, 9999, n))], dtype=object)
        phone[rng.random(n) < 0.03] = None

        frames.append(pd.DataFrame({
            "load_date": day.strftime("%Y-%m-%d"), "customer_id": cid, "record_type": np.where(is_new, "NEW", "UPDATE"),
            "first_name": fn, "last_name": ln, "dob": dob, "email": email, "phone": phone,
            "postal_code": postal, "province": "ON",
        }))
    return pd.concat(frames, ignore_index=True)


def main():
    rng = np.random.default_rng(SEED)
    DATA.mkdir(exist_ok=True)
    cal = build_calendar(START, END)
    jp = jackpot_series(cal)
    demand = demand_index(cal, jp)

    act = make_activity(cal, demand, rng)
    act["jackpot_musd"] = jp.round(1)
    act.to_csv(DATA / "activity_daily.csv")

    tomb = make_tombstone(cal, demand, rng)
    tomb.to_csv(DATA / "tombstone_feed.csv.gz", index=False, compression="gzip")

    incidents_frame().to_csv(DATA / "incidents.csv", index=False)
    print(f"activity_daily.csv   {len(act):>9,} days")
    print(f"tombstone_feed.csv.gz {len(tomb):>9,} rows")
    print(f"incidents.csv        {len(INCIDENTS):>9} incidents")


if __name__ == "__main__":
    main()
