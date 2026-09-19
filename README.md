# Market Risk Statistical Validation Framework

**Do the standard Value-at-Risk models actually hold up out of sample?
A classical-statistics audit on NIFTY 50 (2019–2024).**

Python · NumPy · pandas · SciPy · statsmodels · arch · Matplotlib · Streamlit

---

## Aim

Take five VaR models a risk desk actually uses — **Historical**, **Parametric
Normal**, **Parametric Student-t**, **EWMA (RiskMetrics, λ=0.94)** and
**GARCH(1,1)** — and put every one of them through the same regulatory-style
validation battery on six years of NIFTY 50 data:

| Test | Question | Distribution |
|---|---|---|
| Kupiec (1995) Proportion-of-Failures | Did the model breach the right **number** of times? | χ²(1) |
| Christoffersen (1998) independence | Did the breaches arrive at independent **times**, or in bursts? | χ²(1) |
| Christoffersen joint conditional coverage | Both together — the test a model must survive to be usable | χ²(2) |

Plus **Expected Shortfall at 97.5%**, the measure the Basel FRTB adopted in place
of 99% VaR, and **Basel traffic-light capital zones**, the multiplier a
regulator actually applies to a backtest rather than its p-value.

Every estimate is walk-forward out of sample: VaR for day *t* is fitted on days
*t−250 … t−1* and then tested against day *t*. Nothing that describes day *t* has
seen day *t*.

## Motivation — why I built this

Most VaR write-ups stop at *computing* a number. That is the easy half. The
question a risk function is actually asked is **"can we rely on this number?"** —
and that is a hypothesis-testing problem, not an estimation problem.

Three specific things shaped the design.

**A single p-value hides the failure that matters.** A model can breach exactly
5% of the time and still be dangerous if every breach lands in the same
fortnight. Kupiec is structurally blind to that — it sees only the count. Only
the independence test sees it. So the full battery is reported per model, and the
verdict column follows the joint test rather than either half.

**Look-ahead bias is silent, and it makes results look *better*.** While building
the volatility-regime section I found that labelling a day "high volatility" from
a rolling window that includes that day's own return manufactures statistical
significance. On i.i.d. noise with a true variance ratio of exactly 1.0, the
leaky labelling rejects equal variance decisively; the correct lagged labelling
finds nothing. The notebook keeps both versions side by side so the size of the
bug is measurable instead of merely described. That one `.shift(1)` is the most
important line in the repository, and two regression tests fail if it is removed.

**Basel moved on, so the project should too.** The FRTB replaced 99% VaR with
97.5% ES because VaR says nothing about how bad a breach is and is not
sub-additive. The ES section reproduces the argument on real data. Separately,
a statistical verdict (reject / fail to reject) is not what a bank acts on — it
acts on a capital multiplier, so the traffic-light section translates the same
backtest into the actual regulatory consequence.

Deliberately classical throughout — distributions, maximum likelihood,
hypothesis tests, OLS with heteroskedasticity-robust errors, likelihood-ratio
backtests. GARCH(1,1) sits on the same side of that line as everything else
here: it is a fully specified likelihood fit by MLE, with two estimated
parameters (persistence and reaction) where EWMA has one assumed one (a fixed
decay). **No machine learning anywhere.** The point is inference, not
prediction.

---

## Headline findings

Numbers below come from a full run of `main.ipynb` and the `app.py` dashboard
at their default settings (250-day rolling window, 95% VaR / 97.5% ES) and are
reproducible by re-running either — both import the same `risk_utils.py`
functions, so a figure can't differ between the notebook and the dashboard.
There is no separate results file; the notebook itself is the record.

**1 — Kupiec is necessary but not sufficient, and this is the project's central
result.** At least one model at each confidence level passes the
proportion-of-failures test and is then rejected by the independence test: the
breach *count* is right while the breach *timing* is clustered. Clustered
breaches are the failure mode that turns a limit excess into a capital event. A
validation report that stops at Kupiec can sign off a model that was wrong for
two straight weeks.

**2 — Look-ahead bias fabricates significance from pure noise.** On synthetic
i.i.d. Normal returns, where the true high-vol/low-vol variance ratio is exactly
1.0 by construction, the naive rolling-window labelling reports a ratio well
above 1 and rejects equal variance. The ex-ante labelling recovers ≈1.0 and
correctly fails to reject. The regime effect on real NIFTY data is genuine — but
the naive implementation could not have established that, because it rejects on
random numbers too.

**3 — VaR calibration does not transfer across confidence levels.** Models that
pass Kupiec at 95% fail at 99%. A validation performed at one level says nothing
about another, which is a direct argument against the common practice of
validating once and reporting everywhere.

**4 — Risk lives in the second moment.** Mean return is statistically
indistinguishable between low- and high-volatility regimes (Welch's t-test,
Mann-Whitney U); variance is decisively different (Levene / Brown-Forsythe).
"High volatility" does not mean "the market is falling" — it means outcomes are
more uncertain in both directions.

**5 — Returns are decisively non-Normal, confirmed two independent ways.**
Jarque-Bera rejects Normality at any conventional level (fat tails, negative
skew), and the measured ES/VaR ratio exceeds the Normal benchmark at every
confidence level — the same conclusion reached from the loss side rather than the
moment side.

**6 — A one-day return threshold is a weak proxy for a volatility regime.**
Tightening the stress-flag percentile trades recall away far faster than it buys
precision. A point detector cannot identify a state, which is the argument for
volatility-reactive models (EWMA, and GARCH(1,1) alongside it) over a
point-in-time rule.

**7 — Beta is not total risk.** Systematic exposure varies materially across the
four stocks, with TCS and INFY statistically below the market. HC1
heteroskedasticity-robust standard errors were used because return residuals are
conditionally heteroskedastic by construction; classical OLS errors overstate the
precision of every one of these betas.

**8 — Estimating persistence (GARCH) did not beat assuming it (EWMA), on this
data.** At the default settings, Historical, Normal Parametric, EWMA and
GARCH(1,1) all survive the full battery; only Student-t Parametric is
rejected, specifically for clustered breach timing. GARCH's maximum-likelihood
persistence estimate was not necessary to fix the clustering problem here — a
fixed conventional decay already worked, which is itself a useful (negative)
result about when the extra estimation earns its keep.

**9 — The Basel traffic-light zone and the independence test can disagree.**
Scaled to this project's out-of-sample window, every model lands in Basel's
red zone by exception count alone — including models the independence test
says are fine. The zone only counts whether there were too many breaches,
never whether their timing was random, which is exactly the blind spot the
rest of this project is built to expose.

---

## Methodology

**Data.** NIFTY 50 (`^NSEI`) plus RELIANCE, TCS, HDFCBANK, INFY, daily adjusted
closes, 2019-01-01 to 2024-12-31 (requested range — the notebook and dashboard
download live from Yahoo Finance on each run, so the actual end date reflects
whatever the most recent trading day was at run time). Log returns throughout,
for time-additivity.

**Volatility regimes.** 21-day trailing realised volatility, split at its median,
**lagged one day** so a day's label uses only information available at its open.
`classify_regime(..., ex_ante=False)` reproduces the biased version for the
comparison.

**Market model.** `R_i = α + β·R_m + ε`, OLS with **HC1** robust standard errors.
This is the classical market model, not CAPM — no risk-free rate is subtracted
and no equilibrium claim is made; β is a measured sensitivity, not a required
return.

**VaR.** Historical (empirical quantile), Parametric Normal, Parametric Student-t
(MLE location–scale fit), EWMA with λ=0.94, and GARCH(1,1) (Normal innovations,
fit by MLE via the `arch` package, one-step-ahead variance forecast). GARCH is
detected at import time (`risk_utils.GARCH_AVAILABLE`): if `arch` is not
installed, `VAR_METHODS` simply omits `"garch"` and every loop over it — in
this module, the notebook, and the dashboard — runs the other four models
without crashing. `pip install arch` (already in `requirements.txt`) adds it
back. Backtesting
is walk-forward on a 250-day rolling window; GARCH refits the likelihood on every
window and is the most expensive estimator in the module by a wide margin.

**Backtests.** Kupiec POF, Christoffersen independence via a first-order Markov
chain on the breach indicator, and joint conditional coverage using the identity
`LR_CC = LR_POF + LR_IND ~ χ²(2)`. Log-likelihoods use `scipy.special.xlogy` so
the 0·log(0) boundary cases are handled at their limits rather than producing
`nan`.

**Basel traffic-light zones.** `basel_traffic_light()` maps a 99% VaR exception
count onto Basel's official green (0–4) / yellow (5–9) / red (10+) zones and the
associated capital-multiplier add-on (base k=3.00). Officially defined for a
250-trading-day window; this project's out-of-sample window is longer, so the
function scales the boundaries proportionally and explicitly flags the result as
an approximation rather than the literal regulatory table.

**Expected Shortfall.** Historical, Normal closed form, and Student-t closed form
at 97.5%, reported with the ES/VaR ratio as a tail-heaviness diagnostic.
Student-t ES returns `nan` when the fitted degrees of freedom are ≤ 1, because
the conditional mean does not exist there.

---

## Verification

There is no automated test suite in this project (no `pytest`, no `tests/`
folder) — worth saying plainly rather than leaving it implied. Two things do
provide real confidence in the numbers, short of formal test coverage:

* **One engine, two consumers.** `main.ipynb` and `app.py` both import their
  statistics from `risk_utils.py` rather than reimplementing anything, so a
  number can't quietly drift between the notebook and the dashboard — if one
  is wrong, both are wrong the same way, which at least makes a bug visible
  rather than silently inconsistent.
* **The look-ahead demonstration is itself a verification exercise.** Section
  3 doesn't just describe the look-ahead bug — it runs the naive and ex-ante
  labelling on synthetic i.i.d. data where the true answer is known in
  advance (variance ratio = 1.0 by construction), and shows the naive version
  gets it wrong. That's the same logic a unit test would use, just run
  inline in the notebook rather than pinned as an automated regression.

**Honest next step, not yet done:** a `pytest` suite covering `risk_utils.py`
directly — closed-form functions checked against independent recomputation
(e.g. EWMA against its algebraic unrolling, Kupiec against a hand-derived
likelihood-ratio value), and the look-ahead and clustering findings above
pinned as regression tests so they can't silently break. Framed here as
planned, not implied to already exist.

---

## Reproducing the results

```bash
pip install -r requirements.txt          # runtime (dashboard + engine, incl. `arch` for GARCH)

streamlit run app.py                     # interactive dashboard
jupyter notebook main.ipynb              # full analysis narrative
```

Both pull price data live from Yahoo Finance on every run — there is no
committed data snapshot in this repo, so each run needs an internet
connection and reflects whatever Yahoo Finance returns at that moment. This
means results can drift slightly run to run as more recent trading days
become available, rather than being pinned to a fixed historical window.

`risk_utils.py` is the single source of truth for every figure: the notebook
and the dashboard both import it rather than reimplementing anything, so no
number is calculated twice. The GARCH cells (Section 8b, and the equivalent
backtest in the dashboard) refit a maximum-likelihood optimisation on every
rolling window and are noticeably slower than the rest of the notebook to
execute — expect the first run to take longer, not to fail.

---

## Project structure

```
Market Risk Statistical Validation Framework/
├── risk_utils.py           # the engine: all statistics, VaR, ES and backtests
├── main.ipynb              # analysis narrative, 13 sections with observations
├── app.py                  # Streamlit dashboard over the same engine
├── style.css               # dashboard styling
└── requirements.txt
```

Five files, one job each. `risk_utils.py` is imported by both the notebook
and the dashboard, deliberately: a number shown in the dashboard is computed
by the exact same function the notebook's narrative walks through.

---

## Dashboard

`streamlit run app.py` — an 8-chapter guided narrative (Executive Summary,
Distribution, Volatility Regimes, Systematic Risk, Real-Time Detection,
Estimating the Loss, Out-of-Sample Validation, and a closing Verdict), read
in order via Next/Previous or jumped to directly from the sidebar's Story
list. Every chapter states its question before its answer, and the
Executive Summary and Verdict chapters are computed live from whatever data
is actually loaded — not fixed text. Downloads price data live from Yahoo
Finance on each run.

---

## Scope and honest limitations

* One market, one asset class, roughly six years. A single COVID-scale shock
  dominates the tail, so tail conclusions rest on few effective observations.
  Extending the window to also include a structurally different crisis (e.g.
  the 2008 GFC or the 2013 taper tantrum) is a natural next step — the date
  range is a configurable constant at the top of the notebook and in the
  dashboard's sidebar, but has not been run with a longer window in this
  revision, so this remains open rather than done.
* At 99% the expected breach count over this sample is around 12, so the
  Christoffersen tests have low power there. **Failing to reject is not evidence
  of adequacy** — and the notebook says so rather than claiming a pass.
* GARCH(1,1) uses a Normal innovation distribution. A Student-t innovation
  (combining the fat-tail finding from Section 2 with GARCH's persistence
  estimate) is the natural next refinement and was not implemented here.
* The Basel traffic-light zones (Section 9b) use boundaries scaled from the
  official 250-day table, because this project's out-of-sample window is
  longer than 250 days — an explicitly flagged approximation, not the exact
  regulatory calibration.
* VaR is one-day and unscaled. The √h scaling rule assumes i.i.d. returns, which
  the volatility-clustering result shows is false, so multi-day figures were
  deliberately not reported.
* Regime classification is a median split on trailing volatility — transparent
  and testable, but cruder than a Markov-switching model.
* Equity only. No fixed income, FX, options, or portfolio-level VaR aggregation
  (which is where VaR's failure of sub-additivity would bite hardest).

---

## Key design principle

> The interesting question about a risk model is never "what number does it
> produce" but **"under what conditions does it fail, and would my validation
> have caught it?"**

Two failures in this project were invisible in the model output and visible only
under a properly specified test: a look-ahead leak that *improved* the reported
p-value, and a VaR model with a correct breach count and dangerously clustered
breach timing. Both were found by the test, not by the eye — which is the entire
case for building the test.
