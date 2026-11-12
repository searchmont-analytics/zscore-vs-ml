"""Profile feeds into daily DQ metrics, run three detectors, score them against ground truth.

Detectors
---------
zscore_30d   Rolling 30-day baseline (mean/std of the prior 30 days), alert if |z| > 3.
             The classic approach -- and what most DQ tools ship by default.
dow_zscore   "Seasonal" z-score: compare to the same weekday over the prior 8 weeks.
             The usual first fix. It handles weekends but not holidays or jackpots.
ml_expected  Gradient-boosted model of the *expected* value for that day from calendar
             facts (weekday, holiday, long weekend, pay-day, events, jackpot) + recent level.
             Alerts on the residual (actual vs expected) scaled by its own recent error.
             Retrained weekly, walk-forward -- it never sees the future.
             Adds a 7-day "sustained shift" rule on residuals to catch slow drift.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from calendar_features import build_calendar

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "results"

VOLUME_METRICS = ["activity_records", "daily_spend", "tombstone_records"]
RATE_METRICS = ["email_null_pct", "invalid_postal_pct", "duplicate_id_pct", "underage_pct"]
METRICS = VOLUME_METRICS + RATE_METRICS

Z_WINDOW, Z_K = 30, 3.0
DOW_WEEKS = 8
ML_WARMUP, ML_RETRAIN_EVERY, ML_K = 90, 7, 3.5
DRIFT_WINDOW, DRIFT_K = 7, 3.0
POSTAL_RE = re.compile(r"^[A-Z]\d[A-Z] \d[A-Z]\d$")

CAL_FEATURES = ["dow", "is_weekend", "is_holiday", "is_event", "is_long_weekend", "days_to_holiday",
                "days_since_holiday", "month", "day_of_month", "is_payday", "is_xmas_week", "is_draw_day"]


# --------------------------------------------------------------------------------------
# 1. Profiling: row-level tombstone feed -> one row of DQ metrics per load date
# --------------------------------------------------------------------------------------
def profile_tombstone(path: Path, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    t = pd.read_csv(path, dtype={"postal_code": str}, keep_default_na=True)
    t["load_date"] = pd.to_datetime(t["load_date"])
    age = (t["load_date"] - pd.to_datetime(t["dob"])).dt.days / 365.25
    t["_email_null"] = t["email"].isna()
    t["_bad_postal"] = ~t["postal_code"].fillna("").str.match(POSTAL_RE)
    t["_dup_id"] = t.duplicated(["load_date", "customer_id"], keep=False)
    t["_underage"] = age < 19
    g = t.groupby("load_date")
    prof = pd.DataFrame({
        "tombstone_records": g.size(),
        "email_null_pct": g["_email_null"].mean() * 100,
        "invalid_postal_pct": g["_bad_postal"].mean() * 100,
        "duplicate_id_pct": g["_dup_id"].mean() * 100,
        "underage_pct": g["_underage"].mean() * 100,
    }).reindex(all_dates)
    prof["tombstone_records"] = prof["tombstone_records"].fillna(0)  # a missing day = 0 rows
    prof.index.name = "date"
    return prof


# --------------------------------------------------------------------------------------
# 2. Detectors
# --------------------------------------------------------------------------------------
def zscore_30d(s: pd.Series, k: float = Z_K) -> pd.DataFrame:
    base = s.shift(1).rolling(Z_WINDOW, min_periods=Z_WINDOW)
    mu, sd = base.mean(), base.std()
    z = (s - mu) / sd.replace(0, np.nan)
    return pd.DataFrame({"expected": mu, "lo": mu - k * sd, "hi": mu + k * sd, "score": z, "alert": z.abs() > k})


def dow_zscore(s: pd.Series) -> pd.DataFrame:
    lags = pd.concat([s.shift(7 * k) for k in range(1, DOW_WEEKS + 1)], axis=1)
    mu, sd = lags.mean(axis=1), lags.std(axis=1)
    ok = lags.notna().sum(axis=1) == DOW_WEEKS
    mu, sd = mu.where(ok), sd.where(ok)
    z = (s - mu) / sd.replace(0, np.nan)
    return pd.DataFrame({"expected": mu, "lo": mu - Z_K * sd, "hi": mu + Z_K * sd, "score": z, "alert": z.abs() > Z_K})


def _transform(metric):
    """Volumes are multiplicative (log scale); rates are additive (identity)."""
    if metric in VOLUME_METRICS:
        return (lambda x: np.log1p(np.asarray(x, float))), (lambda y: np.expm1(y))
    return (lambda x: np.asarray(x, float)), (lambda y: y)


HOLIDAY_SHRINK = 0.85    # trust last year's holiday surprise, but not 100 %
GENERIC_SHRINK = 0.5     # an unseen holiday gets half the typical holiday uplift...
UNSEEN_BAND, SEEN_BAND = 2.0, 1.3   # ...and a wider alert band while it's still uncertain


def make_model():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                         min_samples_leaf=4, l2_regularization=1.0,
                                         early_stopping=False, random_state=0)


def ml_expected(s: pd.Series, cal: pd.DataFrame, confirmed: pd.Series, metric: str,
                n_rows: pd.Series | None = None) -> pd.DataFrame:
    """Walk-forward, two-stage expected-value detector.

    Stage 1  "Normal day" model. Gradient-boosted trees learn weekday, long weekend,
             pay-day, jackpot size and season, relative to the recent level (28-day rolling
             median, lagged one day). Trained WITHOUT named holidays/events.
    Stage 2  Holiday memory. Every named holiday is one of a kind (Christmas is quiet,
             New Year's Eve is huge) and comes once a year -- too rare for trees. So we
             remember how far each holiday sat from a normal day last time and apply it
             again. A holiday never seen before gets a damped "typical holiday" uplift and
             a wider alert band. This is the "a human remembers last Christmas" step, and
             why the model is much quieter in its second year.

    `confirmed` marks days a human triaged as real incidents *after* they alerted -- those
    are dropped from training (the feedback loop). Missed incidents stay in training, as
    they would in production.
    """
    fwd, inv = _transform(metric)
    y = pd.Series(fwd(s.values), index=s.index)
    X = cal[CAL_FEATURES].copy()
    X["jackpot_musd"] = cal["jackpot_musd"]
    X["level"] = y.shift(1).rolling(28, min_periods=7).median()  # never includes today

    n = len(s)
    Xv, yv, lvl = X.values.astype(float), y.values, X["level"].values
    names = (cal["holiday_name"] + cal["event_name"]).values
    is_rate = metric in RATE_METRICS
    nv = None if n_rows is None else n_rows.values.astype(float)

    pred = np.full(n, np.nan)
    holiday_adj = np.zeros(n)
    band = np.ones(n)             # alert-band multiplier (wider on holidays)
    resid = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    center = np.zeros(n)          # recent median residual (removes slow bias, e.g. growth)
    alert = np.zeros(n, bool)
    drift = np.zeros(n, bool)
    excluded = np.zeros(n, bool)  # confirmed-incident days removed from training
    model, insample_sigma = None, None

    # Cold start: a quick backtest inside the warm-up window gives honest out-of-sample
    # residuals to size the alert band before the model has any live history.
    bt_split = int(ML_WARMUP * 0.6)
    tr0 = np.arange(bt_split)
    tr0 = tr0[~np.isnan(lvl[:bt_split]) & ~np.isnan(yv[:bt_split])]
    m0 = make_model().fit(Xv[tr0], yv[tr0] - lvl[tr0])
    te = np.arange(bt_split, ML_WARMUP)
    backtest = yv[te] - (m0.predict(Xv[te]) + lvl[te])
    backtest = backtest[~np.isnan(backtest)]

    for i in range(ML_WARMUP, n):
        if model is None or (i - ML_WARMUP) % ML_RETRAIN_EVERY == 0:
            tr = np.arange(i)
            tr = tr[~excluded[:i] & ~np.isnan(lvl[:i]) & ~np.isnan(yv[:i])]
            named = names[tr] != ""
            yt = yv[tr] - lvl[tr]  # learn the deviation from recent level
            model = make_model().fit(Xv[tr][~named], yt[~named])
            r_in = yt - model.predict(Xv[tr])
            insample_sigma = 1.4826 * np.median(np.abs(r_in[~named] - np.median(r_in[~named])))
            # Holiday memory: deviation from a normal day at each holiday's latest clean occurrence.
            last_dev = {names[j]: r for j, r in zip(tr[named], r_in[named])}
            generic = GENERIC_SHRINK * np.median(r_in[named]) if named.any() else 0.0
            # Score the coming week in one batch (all features are known in advance).
            blk = np.arange(i, min(i + ML_RETRAIN_EVERY, n))
            pred[blk] = model.predict(Xv[blk]) + lvl[blk]
            for b in blk:
                if names[b] in last_dev:
                    holiday_adj[b], band[b] = HOLIDAY_SHRINK * last_dev[names[b]], SEEN_BAND
                elif names[b]:
                    holiday_adj[b], band[b] = generic, UNSEEN_BAND
                pred[b] += holiday_adj[b]

        resid[i] = yv[i] - pred[i]

        # Error scale from the model's own recent out-of-sample residuals (last 180 days).
        win = slice(max(ML_WARMUP, i - 180), i)
        hist = resid[win][~np.isnan(resid[win]) & ~excluded[win] & (names[win] == "")]
        if len(hist) < 30:  # cold start: top up with backtest residuals
            hist = np.concatenate([backtest, hist])
        if len(hist) >= 10:
            center[i] = np.median(hist)
            sig = 1.4826 * np.median(np.abs(hist - center[i]))
        else:
            center[i], sig = 0.0, 2.0 * insample_sigma
        if is_rate and nv is not None and nv[i] > 0 and not np.isnan(pred[i]):
            # Proportions: never tighter than binomial sampling noise for today's row count.
            p_hat = min(max(pred[i] / 100, 1 / nv[i]), 0.5)
            sig = max(sig, 100 * np.sqrt(p_hat * (1 - p_hat) / nv[i]))
        sigma[i] = max(sig, 1e-6) * band[i]
        alert[i] = abs(resid[i] - center[i]) > ML_K * sigma[i]

        # Sustained-shift (drift) rule: mean residual over the last 7 days. One-off spikes
        # already alerted are left out so they don't echo into a phantom "drift".
        lo_i = i - DRIFT_WINDOW + 1
        w = resid[lo_i:i + 1] - center[i]
        keep = ~np.isnan(w) & ~alert[lo_i:i + 1]
        if keep.sum() >= DRIFT_WINDOW - 2 and not alert[i]:
            drift[i] = abs(w[keep].mean()) > DRIFT_K * sigma[i] / np.sqrt(keep.sum()) * 1.5

        if (alert[i] or drift[i]) and confirmed.iloc[i]:
            excluded[i] = True

    pc = pred + center
    lo, hi = inv(pc - ML_K * sigma), inv(pc + ML_K * sigma)
    return pd.DataFrame({"expected": inv(pc), "lo": lo, "hi": hi, "score": (resid - center) / sigma,
                         "alert": alert | drift, "spike_alert": alert, "drift_alert": drift,
                         "holiday_adj": holiday_adj}, index=s.index)


# --------------------------------------------------------------------------------------
# 3. Evaluation
# --------------------------------------------------------------------------------------
def day_type(cal_row) -> str:
    if cal_row.is_holiday:
        return "Stat holiday"
    if cal_row.is_event:
        return "Event day"
    if cal_row.is_long_weekend:
        return "Long weekend"
    if cal_row.dow in (4, 5, 6):
        return "Weekend"
    return "Weekday"


def main():
    OUT.mkdir(exist_ok=True)
    act = pd.read_csv(DATA / "activity_daily.csv", parse_dates=["date"], index_col="date")
    cal = build_calendar(act.index.min(), act.index.max())
    cal["jackpot_musd"] = act["jackpot_musd"]
    prof = profile_tombstone(DATA / "tombstone_feed.csv.gz", act.index)
    metrics = pd.concat([act[["activity_records", "daily_spend"]], prof], axis=1)
    metrics.to_csv(OUT / "daily_metrics.csv")

    inc = pd.read_csv(DATA / "incidents.csv", parse_dates=["date"])
    eval_start = metrics.index[ML_WARMUP]
    cal["day_type"] = [day_type(r) for r in cal.itertuples()]

    detectors = {"zscore_30d": "Rolling 30-day z-score (3σ)", "zscore_30d_k2": "Rolling 30-day z-score (2σ)",
                 "dow_zscore": "Same-weekday z-score (8 wk)",
                 "ml_expected": "ML expected-value model"}
    long_rows, alert_rows, inc_rows = [], [], []

    for m in METRICS:
        s = metrics[m].astype(float)
        truth = pd.Series(False, index=s.index)
        truth.loc[inc.loc[inc.metric == m, "date"]] = True
        res = {"zscore_30d": zscore_30d(s), "dow_zscore": dow_zscore(s), "zscore_30d_k2": zscore_30d(s, k=2.0),
               "ml_expected": ml_expected(s, cal, truth, m, None if m in VOLUME_METRICS else metrics["tombstone_records"])}

        for d, r in res.items():
            r = r.copy()
            r["actual"], r["metric"], r["detector"], r["is_incident"] = s, m, d, truth
            r["alert"] = r["alert"].fillna(False).astype(bool)
            long_rows.append(r.reset_index())
            ev = r.loc[eval_start:]
            fp = ev[ev.alert & ~ev.is_incident]
            for dt_, row in fp.iterrows():
                alert_rows.append({"date": dt_, "metric": m, "detector": d, "kind": "false_alert",
                                   "day_type": cal.at[dt_, "day_type"],
                                   "label": cal.at[dt_, "holiday_name"] or cal.at[dt_, "event_name"] or ""})

            for iid, grp in inc[inc.metric == m].groupby("incident_id"):
                hit_days = r.loc[grp.date, "alert"]
                first_hit = hit_days[hit_days].index.min() if hit_days.any() else None
                inc_rows.append({"incident_id": iid, "metric": m, "detector": d, "kind": grp.kind.iloc[0],
                                 "description": grp.description.iloc[0], "start": grp.date.min(),
                                 "detected": bool(hit_days.any()),
                                 "days_to_detect": None if first_hit is None else (first_hit - grp.date.min()).days})

    long = pd.concat(long_rows, ignore_index=True)
    long.to_csv(OUT / "detector_output.csv", index=False)
    fps = pd.DataFrame(alert_rows)
    incs = pd.DataFrame(inc_rows)
    fps.to_csv(OUT / "false_alerts.csv", index=False)
    incs.to_csv(OUT / "incident_detection.csv", index=False)

    # ---- Summary ----
    eval_days = len(metrics.loc[eval_start:])
    summary = {"eval_start": str(eval_start.date()), "eval_end": str(metrics.index[-1].date()),
               "eval_days": eval_days, "detectors": {}}
    print(f"\nEvaluation window {eval_start.date()} -> {metrics.index[-1].date()} ({eval_days} days)\n")
    print(f"{'detector':<30}{'incidents caught':>18}{'false alerts':>14}{'precision':>11}")
    for d, label in detectors.items():
        di = incs[incs.detector == d]
        caught = int(di.groupby(["incident_id", "metric"]).detected.any().sum())
        total = len(di)
        n_fp = int((fps.detector == d).sum()) if len(fps) else 0
        ev = long[(long.detector == d) & (long.date >= eval_start)]
        n_tp_days = int((ev.alert & ev.is_incident).sum())
        precision = n_tp_days / max(n_tp_days + n_fp, 1)
        by_type = fps[fps.detector == d].day_type.value_counts().to_dict() if len(fps) else {}
        by_metric = fps[fps.detector == d].metric.value_counts().to_dict() if len(fps) else {}
        summary["detectors"][d] = {"label": label, "incidents_caught": caught, "incidents_total": total,
                                   "false_alerts": n_fp, "alert_precision": round(precision, 3),
                                   "false_alerts_by_day_type": by_type, "false_alerts_by_metric": by_metric}
        print(f"{label:<30}{caught:>10}/{total:<7}{n_fp:>14}{precision:>11.0%}")
        print(f"{'':<4}false alerts by day type: {by_type}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\nPer-incident detection:")
    piv = incs.pivot_table(index=["incident_id", "metric", "kind"], columns="detector", values="detected", aggfunc="first")
    print(piv[list(detectors)].replace({True: "caught", False: "MISSED"}).to_string())


if __name__ == "__main__":
    main()
