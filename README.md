# Beyond the Z-Score: data observability POC

**Case study:** https://data-analytics-travis-mccormack.com/observability-lab/

A proof of concept for data-quality observability on a customer tombstone feed and daily play activity
(record volume + spend) for an Ontario gaming operator. All data is synthetic.

It shows why a rolling 30-day z-score is not enough when the business has a strong rhythm (weekends,
long weekends, holidays, Super Bowl, lottery jackpots), and how an expected-value ML model fixes it.

## Results (Nov 30 2024 – Oct 31 2026, 7 metrics, 15 injected incidents / 21 incident signals)

| Detector | Incidents caught | False alerts | Precision |
|---|---|---|---|
| Rolling 30-day z-score (3σ) | 9 / 21 | 34 | 28% |
| Rolling 30-day z-score (2σ) | 11 / 21 | 216 | 6% |
| Same-weekday z-score (8 wk) | 18 / 21 | 185 | 11% |
| **ML expected-value model** | **18 / 21** | **13** | **66%** |

## Layout

```
src/calendar_features.py  Ontario stat holidays, events, long weekends, pay-days, jackpot simulation
src/generate_data.py      synthetic feeds + incident ground truth  -> data/
src/observability.py      profile tombstone rows, run 4 detectors, score them -> results/
src/build_report.py       inline results into report/ and site/observability-lab/index.html
```

Simulated data for illustration only. No OLG data is used or shown.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/generate_data.py
OMP_NUM_THREADS=2 .venv/bin/python src/observability.py   # ~3 min
.venv/bin/python src/build_report.py
```

## How the ML detector works

1. **Normal-day model:** gradient-boosted trees predict the log deviation from the 28-day rolling median using
   weekday, long weekend, pay-day, season and jackpot size. It is trained without named holidays and retrained weekly, walk-forward.
2. **Holiday memory:** each named holiday reuses 85% of how far it sat from a normal day last time.
   A holiday it hasn't seen yet gets half the typical holiday uplift and a 2× wider alert band.
3. **Alerting:** a day alerts when |actual − expected| is more than 3.5 × the robust out-of-sample error.
   A 7-day mean-residual rule catches slow drift. Rate metrics get a binomial noise floor.
4. **Feedback:** confirmed incidents are dropped from training. Missed ones stay in, as they would in production.

Known gap: holiday-weekend spend dips of about 30% are near the spend noise floor. A spend-per-record reconciliation
rule is the better control for those.
