"""Bundle results/ into report/report_data.json and inline it into the HTML report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES, REP = ROOT / "results", ROOT / "report"
REPO_URL = "https://github.com/searchmont-analytics/zscore-vs-ml"
PORTFOLIO_URL = "/#lab"
FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' "
           "rx='7' fill='%231a1a19'/%3E%3Crect x='7' y='17' width='4' height='8' rx='1.5' fill='%233987e5'/%3E%3Crect x='14' "
           "y='11' width='4' height='14' rx='1.5' fill='%233987e5'/%3E%3Crect x='21' y='7' width='4' height='18' rx='1.5' "
           "fill='%23199e70'/%3E%3C/svg%3E")
SITE_HEAD = f"""<!doctype html>
<html lang="en-CA" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Case study: why a rolling z-score raised false alerts on weekends and holidays, and how a seasonal ML model fixed it. Simulated Ontario gaming data.">
<meta property="og:title" content="Beyond the Z-Score | Travis McCormack">
<meta property="og:description" content="Data observability case study: z-score vs. seasonal ML on customer tombstone and daily play data.">
<meta property="og:type" content="article">
<link rel="icon" href="{FAVICON}">
<script>try {{ var t = localStorage.getItem('theme'); if (t) document.documentElement.dataset.theme = t; }} catch (e) {{}}</script>
<style>body {{ margin: 0; }} [hidden] {{ display: none !important; }} img {{ max-width: 100%; }}</style>
</head>
<body>
"""
SITE_TAIL = """
<script>
(function () {
  var btn = document.getElementById('themeToggle');
  function label() { btn.textContent = document.documentElement.dataset.theme === 'light' ? 'Dark theme' : 'Light theme'; }
  label();
  btn.addEventListener('click', function () {
    var next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('theme', next); } catch (e) {}
    label();
  });
})();
</script>
</body>
</html>
"""
SITE_TOPBAR = f"""  <nav class="topnav" aria-label="Site">
    <a href="{PORTFOLIO_URL}">&larr; Back to portfolio</a>
    <div class="right"><a href="{REPO_URL}">Code on GitHub</a><button type="button" id="themeToggle">Light theme</button></div>
  </nav>"""
SITE_REPO = f"""    <p class="prose">Source, data generator and evaluation: <a href="{REPO_URL}" style="color:var(--ml)">{REPO_URL.replace("https://", "")}</a></p>"""
SITE_FOOTLINK = f"""<span><a href="{PORTFOLIO_URL}">Travis McCormack · Data &amp; analytics portfolio</a></span>"""
DETS = ["ml_expected", "zscore_30d", "zscore_30d_k2", "dow_zscore"]


def r(x, nd):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def main():
    summary = json.loads((RES / "summary.json").read_text())
    start = pd.Timestamp(summary["eval_start"])
    d = pd.read_csv(RES / "detector_output.csv", parse_dates=["date"])
    d = d[d.date >= start]
    cal_cols = pd.read_csv(ROOT / "data" / "activity_daily.csv", parse_dates=["date"], index_col="date")

    from calendar_features import build_calendar
    cal = build_calendar(cal_cols.index.min(), cal_cols.index.max())
    label = (cal["holiday_name"] + cal["event_name"]).replace({"Labor Day": "Labour Day"})

    series = {}
    for m, g in d.groupby("metric"):
        nd = 0 if m in ("activity_records", "tombstone_records") else (0 if m == "daily_spend" else 3)
        dates = sorted(g.date.unique())
        out = {"dates": [pd.Timestamp(x).strftime("%Y-%m-%d") for x in dates]}
        base = g[g.detector == "ml_expected"].set_index("date").reindex(dates)
        out["actual"] = [r(v, nd) for v in base.actual]
        out["incident"] = [bool(v) for v in base.is_incident]
        out["label"] = [label.get(pd.Timestamp(x), "") for x in dates]
        for det in ["ml_expected", "zscore_30d", "dow_zscore"]:
            gg = g[g.detector == det].set_index("date").reindex(dates)
            out[det] = {"exp": [r(v, nd) for v in gg.expected], "lo": [r(v, nd) for v in gg.lo],
                        "hi": [r(v, nd) for v in gg.hi], "alert": [bool(v) for v in gg.alert]}
        series[m] = out

    inc = pd.read_csv(RES / "incident_detection.csv", parse_dates=["start"])
    incidents = []
    for (iid, metric), g in inc.groupby(["incident_id", "metric"]):
        row = g.iloc[0]
        incidents.append({"id": int(iid), "metric": metric, "kind": row.kind, "description": row.description,
                          "start": row.start.strftime("%Y-%m-%d"),
                          "day": row.start.strftime("%a"),
                          "detected": {r_.detector: bool(r_.detected) for r_ in g.itertuples()}})

    fp = pd.read_csv(RES / "false_alerts.csv", parse_dates=["date"])
    fp["q"] = fp.date.dt.to_period("Q").astype(str)
    quarters = sorted(fp.q.unique())
    by_q = {det: [int(((fp.detector == det) & (fp.q == q)).sum()) for q in quarters] for det in DETS}
    ml_fp = fp[fp.detector == "ml_expected"].assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d"))
    data = {"summary": summary, "series": series, "incidents": incidents,
            "quarters": quarters, "fp_by_quarter": by_q,
            "ml_false_alerts": ml_fp[["date", "metric", "day_type", "label"]].fillna("").to_dict("records")}

    REP.mkdir(exist_ok=True)
    js = json.dumps(data, separators=(",", ":"))
    (REP / "report_data.json").write_text(js)
    tpl = (REP / "template.html").read_text()
    page = tpl.replace("/*__DATA__*/null", js)
    (REP / "data_observability_poc.html").write_text(page)

    # Standalone copy for the portfolio site (GitHub Pages): full document, back link,
    # theme shared with the main site via localStorage("theme"), link to the code repo.
    site = page
    for k, v in {"<!--__TOPBAR__-->": SITE_TOPBAR, "<!--__REPO__-->": SITE_REPO,
                 "<!--__FOOTLINK__-->": SITE_FOOTLINK}.items():
        site = site.replace(k, v)
    out = ROOT / "site" / "observability-lab"
    out.mkdir(parents=True, exist_ok=True)
    # Template's own <title>/<link>/<style> go in <head>; the rest is body content.
    cut = site.index('<div class="wrap">')
    head_bits = site[:cut].replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
    doc = SITE_HEAD.replace("</head>\n<body>", head_bits + "</head>\n<body>") + site[cut:] + SITE_TAIL
    (out / "index.html").write_text(doc)
    print(f"report written ({len(js)/1024:.0f} KB data) + site/observability-lab/index.html")


if __name__ == "__main__":
    main()
