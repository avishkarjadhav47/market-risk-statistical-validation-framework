"""
risk_utils.py
=============
Core statistical engine for the Market Risk Statistical Validation project.

Everything here is CLASSICAL statistics: distributions, hypothesis testing,
OLS regression, confidence intervals, Value-at-Risk, Expected Shortfall and
regulatory-style backtesting. No machine learning is used anywhere in this
module. It is imported by both main.ipynb (the analysis notebook) and app.py
(the Streamlit dashboard), so every number is computed by one implementation
and displayed twice.

Design rule followed throughout: any quantity used to make a statement about
day t must be computable from information available strictly before day t.
Where that rule is deliberately relaxed (for a retrospective comparison), the
function says so explicitly.
"""

import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.special import xlogy

# GARCH(1,1) is an optional feature: the rest of this module, the notebook,
# and the dashboard must all still work if the `arch` package is not
# installed. Detected once, here, rather than re-attempted on every call.
try:
    import arch as _arch_check  # noqa: F401
    GARCH_AVAILABLE = True
except ImportError:
    GARCH_AVAILABLE = False

TRADING_DAYS = 252

# RiskMetrics (1996) decay factor for daily data. Kept as a module constant
# because it is a convention, not a fitted parameter.
EWMA_LAMBDA = 0.94

DEFAULT_SNAPSHOT = Path(__file__).resolve().parent / "data" / "prices.csv"


def _safe_log_likelihood(counts, probs):
    """
    Sum of counts * log(probs), returning 0 for any term where count == 0.

    Binomial and Markov-chain likelihood ratios routinely produce 0 * log(0)
    terms (for example a backtest with no consecutive violations). scipy's
    xlogy defines that term as 0, which is the correct limit, and avoids the
    special-case branching that these tests otherwise need.
    """
    return float(np.sum([xlogy(c, p) for c, p in zip(counts, probs)]))


# ---------------------------------------------------------------------------
# 1. DATA & RETURNS
# ---------------------------------------------------------------------------

def download_prices(tickers, start, end, max_retries: int = 3, timeout: int = 30, pause: float = 1.5):
    """
    Download adjusted close prices for a list of tickers from Yahoo Finance,
    one ticker at a time.

    `tickers` should include the market index (e.g. '^NSEI') plus the
    individual stocks (e.g. 'RELIANCE.NS'). Returns a wide DataFrame indexed
    by date with one column per ticker that succeeded -- a ticker that never
    returns usable data after retrying is simply absent as a column, rather
    than the whole call failing, so the caller can see exactly which ticker
    is the problem.

    auto_adjust=True adjusts historical prices for splits and dividends, so
    the resulting return series reflects total return rather than a series
    with artificial jumps on corporate-action dates.

    Deliberately sequential, not batched: yfinance's batched multi-ticker
    download fires requests for every ticker at once, and Yahoo's rate
    limiter is far more likely to trip on that burst than on requests spaced
    `pause` seconds apart -- in practice this is the difference between
    'some tickers silently come back empty' and 'all of them work'. Each
    ticker gets its own retry-with-backoff, independent of the others.
    """
    import time

    import yfinance as yf

    columns = {}
    failures = {}

    for ticker in tickers:
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                series = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    timeout=timeout,
                )["Close"]

                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]

                if series.dropna().shape[0] > 1:
                    columns[ticker] = series
                    last_error = None
                    break

                last_error = RuntimeError("empty/all-NaN response")
            except Exception as e:
                last_error = e

            if attempt < max_retries:
                time.sleep(pause * attempt)

        if last_error is not None:
            failures[ticker] = last_error

        time.sleep(pause)  # space out requests regardless of outcome

    if not columns:
        raise ConnectionError(
            "Every ticker failed after retrying -- this is a network-level "
            f"failure, not a per-ticker one. Last errors: {failures}. Wait "
            "a few minutes (Yahoo Finance rate-limiting) or check your "
            "internet connection, then try again."
        )

    if failures:
        print(f"Warning: no usable data for {list(failures.keys())} after retrying.")

    data = pd.DataFrame(columns)

    return data.dropna(how="all")


def load_prices(tickers, start, end, snapshot_path=DEFAULT_SNAPSHOT, allow_download=True):
    """
    Snapshot-first price loader.

    Reads a CSV snapshot at data/prices.csv when one exists there and covers
    the requested tickers, and only calls Yahoo Finance otherwise. No such
    snapshot is committed in this repository, so by default every call
    downloads live -- this check exists so that manually placing a
    data/prices.csv file (matching the format download_prices() returns)
    would work without any code changes, not because one ships here.

    Downloading live means reproducibility depends on Yahoo Finance
    returning the same history each time, which it does for historical
    dates but not for the most recent trading days as they get revised.
    """
    snapshot_path = Path(snapshot_path)

    if snapshot_path.exists():
        snap = pd.read_csv(snapshot_path, index_col=0, parse_dates=True)
        wanted = [t for t in tickers if t in snap.columns]

        if len(wanted) == len(tickers):
            sliced = snap.loc[str(start):str(end), wanted]
            if not sliced.empty:
                return sliced.dropna(how="all")

    if not allow_download:
        raise FileNotFoundError(
            f"No usable snapshot at {snapshot_path} and downloading is disabled."
        )

    return download_prices(tickers, start, end)


def log_returns(prices):
    """
    Log returns: r_t = ln(P_t / P_{t-1}).

    Log returns are used rather than simple returns because they are additive
    over time, which is what makes multi-day aggregation and the volatility
    scaling used later internally consistent.
    """
    return np.log(prices / prices.shift(1)).dropna()


def descriptive_stats(returns: pd.Series) -> pd.Series:
    """Mean, standard deviation (daily and annualised), skewness, excess kurtosis."""
    return pd.Series({
        "mean_daily": returns.mean(),
        "std_daily": returns.std(),
        "annualised_vol": returns.std() * np.sqrt(TRADING_DAYS),
        "skewness": stats.skew(returns),
        "excess_kurtosis": stats.kurtosis(returns),  # Fisher: 0 == Normal
    })


# ---------------------------------------------------------------------------
# 2. DISTRIBUTION ANALYSIS — Normal vs Student-t
# ---------------------------------------------------------------------------

def fit_normal(returns):
    """Maximum-likelihood Normal fit. Returns loc and scale."""
    mu, sigma = stats.norm.fit(returns)
    return {"loc": mu, "scale": sigma}


def fit_student_t(returns):
    """
    Maximum-likelihood Student-t fit. Returns df, loc and scale.

    Low degrees of freedom means heavy tails. Note that the fitted `scale` is
    NOT the sample standard deviation: the MLE trades a tighter scale for
    fatter tails in order to fit the peak of the distribution as well as its
    extremes. That trade-off is what drives the Student-t VaR result in
    Section 7 and is discussed there.
    """
    df, loc, scale = stats.t.fit(returns)
    return {"df": df, "loc": loc, "scale": scale}


def goodness_of_fit_normal(returns):
    """
    Jarque-Bera test for normality.

    H0: returns are Normally distributed (skewness 0, excess kurtosis 0).
    p < 0.05 rejects normality.
    """
    jb_stat, jb_p = stats.jarque_bera(returns)
    return {
        "jarque_bera_stat": jb_stat,
        "p_value": jb_p,
        "reject_normality_at_5pct": jb_p < 0.05,
    }


# ---------------------------------------------------------------------------
# 3. VOLATILITY REGIME CLASSIFICATION + HYPOTHESIS TESTING
# ---------------------------------------------------------------------------

def classify_regime(
    returns: pd.Series,
    window: int = 21,
    method: str = "median",
    ex_ante: bool = True,
):
    """
    Label each day 'low_vol' or 'high_vol' from trailing realised volatility.

    ex_ante=True (default, and the only version reported)
        The rolling standard deviation is lagged by one day, so day t's label
        depends only on returns up to and including day t-1.

    ex_ante=False
        The naive version: the rolling window ending at day t INCLUDES day t's
        own return.

    Why this distinction matters, and why it is exposed as a parameter rather
    than silently fixed: with ex_ante=False a day is labelled 'high_vol'
    partly BECAUSE its own return was large in absolute terms. Any subsequent
    test of whether return dispersion differs between the two regimes is then
    partly testing an identity rather than a property of the market — the
    labels were constructed from the quantity being tested. Levene's test in
    particular is inflated by that construction.

    Lagging by one day makes the classification a genuine information set: it
    is what an observer could have known at the open of day t. It also makes
    the Section 5 comparison a real ex-ante detection question instead of a
    retrospective one.

    The notebook reports both so the size of the effect is visible.

    method='median'
        Above the sample median of trailing volatility -> 'high_vol'.
        A RELATIVE regime definition; it does not claim half of all trading
        days were crises.

    method='tercile'
        Bottom third -> 'low_vol', top third -> 'high_vol', middle dropped.
        Gives cleaner separation at the cost of discarding a third of the data.

    Returns
    -------
    (regime, roll_vol) : (pd.Series of labels, pd.Series of trailing volatility)
        Both are indexed identically so they can be aligned downstream.
    """
    roll_vol = returns.rolling(window).std()

    if ex_ante:
        # Day t is described by volatility computed through day t-1 only.
        roll_vol = roll_vol.shift(1)

    roll_vol = roll_vol.dropna()

    if method == "median":
        cutoff = roll_vol.median()
        regime = pd.Series(
            np.where(roll_vol > cutoff, "high_vol", "low_vol"),
            index=roll_vol.index,
        )

    elif method == "tercile":
        q1, q2 = roll_vol.quantile([1 / 3, 2 / 3])
        regime = pd.Series(index=roll_vol.index, dtype=object)
        regime[roll_vol <= q1] = "low_vol"
        regime[roll_vol >= q2] = "high_vol"
        regime = regime.dropna()

    else:
        raise ValueError("method must be 'median' or 'tercile'")

    return regime, roll_vol


def compare_regimes(returns: pd.Series, regime: pd.Series):
    """
    Compare returns across the low- and high-volatility regimes.

    Tests reported, and what each one is for:

    Mann-Whitney U — the PRIMARY test. Non-parametric, so it does not rely on
        the normality assumption that Section 2 rejects outright.

    Welch's t-test — complementary test of the difference in mean return,
        allowing unequal variances (which Levene confirms are unequal).

    Levene — tests whether the two regimes have equal variance. This is the
        dispersion question as opposed to the direction question.

    95% CI on the mean difference (low_vol minus high_vol), computed with a
        Welch-Satterthwaite degrees-of-freedom correction. Reported because a
        confidence interval communicates the precision of the estimate, which
        a p-value alone does not.
    """
    idx = returns.index.intersection(regime.index)
    aligned = returns.reindex(idx)
    regime_aligned = regime.reindex(idx)

    low_vol = aligned[regime_aligned == "low_vol"].dropna()
    high_vol = aligned[regime_aligned == "high_vol"].dropna()

    t_stat, t_p = stats.ttest_ind(low_vol, high_vol, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(low_vol, high_vol, alternative="two-sided")
    lev_stat, lev_p = stats.levene(low_vol, high_vol)

    diff = low_vol.mean() - high_vol.mean()

    var_low, var_high = low_vol.var(ddof=1), high_vol.var(ddof=1)
    n_low, n_high = len(low_vol), len(high_vol)

    se = np.sqrt(var_low / n_low + var_high / n_high)

    df_welch = (
        (var_low / n_low + var_high / n_high) ** 2
        / (
            (var_low / n_low) ** 2 / (n_low - 1)
            + (var_high / n_high) ** 2 / (n_high - 1)
        )
    )

    t_critical = stats.t.ppf(0.975, df=df_welch)

    return {
        "n_low_vol": n_low,
        "n_high_vol": n_high,
        "mean_low_vol": low_vol.mean(),
        "mean_high_vol": high_vol.mean(),
        "std_low_vol": low_vol.std(),
        "std_high_vol": high_vol.std(),
        "t_test_stat": t_stat,
        "t_test_p": t_p,
        "mannwhitney_stat": u_stat,
        "mannwhitney_p": u_p,
        "levene_stat": lev_stat,
        "levene_p": lev_p,
        "mean_diff": diff,
        "mean_diff_95ci": (diff - t_critical * se, diff + t_critical * se),
    }


# ---------------------------------------------------------------------------
# 4. MARKET MODEL REGRESSION (classical OLS, not machine learning)
# ---------------------------------------------------------------------------

def market_model_regression(stock_returns: pd.Series, market_returns: pd.Series):
    """
    Single-factor market model: R_i = alpha + beta * R_m + epsilon.

    Estimated by OLS in statsmodels, so this is closed-form classical
    inference rather than a fitted ML model. Returns the full inferential
    output — standard errors, p-values and confidence intervals — not just
    point estimates, because in a risk context the precision of beta matters
    as much as its value.

    Standard errors are HC1 (White / heteroskedasticity-robust) rather than
    classical. Daily equity returns exhibit volatility clustering, so the
    homoskedasticity assumption behind classical OLS standard errors fails,
    and classical SEs are then biased downward — making beta look more
    precisely estimated than it is. HC1 corrects the uncertainty without
    touching the alpha and beta point estimates.

    This is a MARKET MODEL, not CAPM: no risk-free rate or excess-return
    transformation is applied. It answers "how sensitive is this stock to the
    market", which is what systematic risk means operationally, rather than
    testing an asset-pricing equilibrium condition.
    """
    df = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    df.columns = ["stock", "market"]

    X = sm.add_constant(df["market"])
    model = sm.OLS(df["stock"], X).fit(cov_type="HC1")

    return {
        "alpha": model.params["const"],
        "beta": model.params["market"],
        "r_squared": model.rsquared,
        "alpha_se": model.bse["const"],
        "beta_se": model.bse["market"],
        "alpha_p": model.pvalues["const"],
        "beta_p": model.pvalues["market"],
        "alpha_ci95": tuple(model.conf_int().loc["const"]),
        "beta_ci95": tuple(model.conf_int().loc["market"]),
        "n_obs": int(model.nobs),
        "se_type": "HC1 (heteroskedasticity-robust)",
        "model": model,
    }


# ---------------------------------------------------------------------------
# 5. STRESS-DAY FLAGGING + TYPE I / TYPE II ERROR ANALYSIS
# ---------------------------------------------------------------------------

def flag_stress_days(returns: pd.Series, percentile: float = 5.0, window: int = None):
    """
    Fast, reactive rule: flag a day when its return falls below a lower
    percentile of the return distribution.

    window=None (default)
        Threshold computed from the full sample. This is retrospective: the
        threshold "knows" the whole history including the future.

    window=int
        Threshold computed from a trailing rolling window, so the flag for day
        t uses only data before day t. This is the version to report if the
        rule is being presented as a detection rule rather than a description.
    """
    if window is None:
        threshold = np.percentile(returns, percentile)
        flagged = returns < threshold
        return flagged, threshold

    rolling_threshold = (
        returns.rolling(window)
        .quantile(percentile / 100.0)
        .shift(1)
    )

    flagged = (returns < rolling_threshold).where(rolling_threshold.notna())
    flagged = flagged.dropna().astype(bool)

    return flagged, rolling_threshold.dropna()


def confusion_matrix_metrics(actual_stress: pd.Series, predicted_stress: pd.Series):
    """
    Classification metrics for a detection rule evaluated against a reference
    risk state.

    actual_stress    : boolean series, e.g. regime == 'high_vol'
    predicted_stress : boolean series, e.g. output of flag_stress_days

    Used here as a statistical evaluation of a decision rule and its Type I /
    Type II error trade-off, not as an ML model score.
    """
    idx = actual_stress.index.intersection(predicted_stress.index)
    a = actual_stress.reindex(idx).astype(bool)
    p = predicted_stress.reindex(idx).astype(bool)

    tp = int((a & p).sum())
    fp = int((~a & p).sum())
    fn = int((a & ~p).sum())
    tn = int((~a & ~p).sum())

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "recall_sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "type_I_error_rate_FPR": fp / (fp + tn) if (fp + tn) else np.nan,
        "type_II_error_rate_FNR": fn / (fn + tp) if (fn + tp) else np.nan,
    }


# ---------------------------------------------------------------------------
# 6. VALUE AT RISK
#
# All four estimators return VaR as a POSITIVE number representing a loss.
# ---------------------------------------------------------------------------

def historical_var(returns: pd.Series, alpha: float = 0.05):
    """
    Historical (empirical) VaR: the negative of the empirical alpha-quantile.

    Imposes no distributional assumption, which is its main strength. Its main
    weakness is that a 250-day window contains only about 2.5 observations
    below the 1% quantile, so the deep tail is estimated from almost no data.
    """
    return -np.percentile(returns, alpha * 100)


def parametric_var(returns: pd.Series, alpha: float = 0.05):
    """Parametric Normal VaR: -(mu + z_alpha * sigma)."""
    mu, sigma = returns.mean(), returns.std()
    return -(mu + stats.norm.ppf(alpha) * sigma)


def student_t_var(returns: pd.Series, alpha: float = 0.05):
    """
    Parametric Student-t VaR from a maximum-likelihood location-scale t fit.

    Because the MLE buys heavy tails by shrinking the scale parameter, this
    estimator can be LESS conservative than the Normal at moderate
    confidence levels and only more conservative deep in the tail. Section 7
    of the notebook quantifies that crossover.
    """
    df, loc, scale = stats.t.fit(returns)
    return -(loc + scale * stats.t.ppf(alpha, df))


def ewma_volatility(returns, lam: float = EWMA_LAMBDA):
    """
    One-step-ahead RiskMetrics EWMA volatility forecast.

        sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}

    The recursion is seeded with the sample variance of the supplied window
    and iterated over every observation, so the value returned is the forecast
    for the period immediately AFTER the last observation given. That makes it
    directly usable in a walk-forward backtest with no look-ahead.

    lam = 0.94 is the RiskMetrics daily convention. It is a convention rather
    than a fitted parameter, which is deliberate: it keeps this estimator
    inside the classical scope of the project instead of requiring the
    likelihood machinery of a GARCH model, while still capturing the
    volatility clustering that a flat 250-day window cannot.
    """
    r = np.asarray(returns, dtype=float)

    if r.size < 2:
        raise ValueError("EWMA volatility needs at least two observations.")

    variance = float(np.var(r, ddof=1))

    for observation in r:
        variance = lam * variance + (1.0 - lam) * observation ** 2

    return float(np.sqrt(variance))


def ewma_var(returns, alpha: float = 0.05, lam: float = EWMA_LAMBDA):
    """
    EWMA VaR: -(z_alpha * sigma_ewma), following the RiskMetrics zero-mean
    convention.

    The drift term is dropped on purpose. Over a one-day horizon the expected
    return is tiny relative to volatility, and estimating it from 250 noisy
    observations adds more variance to the VaR estimate than the drift removes
    from its bias.
    """
    sigma = ewma_volatility(returns, lam=lam)
    return -(stats.norm.ppf(alpha) * sigma)


# ---------------------------------------------------------------------------
# 6b. GARCH(1,1) VOLATILITY — the natural comparison to EWMA
#
# EWMA reacts to volatility with a decay lambda fixed by convention (0.94).
# GARCH(1,1) keeps the same idea -- today's variance depends on yesterday's
# variance and yesterday's squared return -- but ESTIMATES how much by
# maximum likelihood instead of assuming it:
#
#     sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
#
# This is still classical inference (a fully specified likelihood, fit by
# MLE), not a machine-learning model -- it simply has two estimated
# parameters (alpha, beta) where EWMA has one assumed one (lambda). It is
# included specifically to test whether estimating persistence from the data
# reduces breach clustering further than EWMA's fixed decay does.
# ---------------------------------------------------------------------------

def garch_volatility(returns, p: int = 1, q: int = 1):
    """
    One-step-ahead GARCH(1,1) volatility forecast, on the ORIGINAL return
    scale, fit by maximum likelihood via the `arch` package.

    Returns are internally rescaled by x100 before fitting and the forecast
    is rescaled back afterwards. This is standard practice for the `arch`
    package on daily equity return series (returns of order 1e-2 sit close
    to the optimiser's numerical floor) and does not change the
    interpretation of the result -- it is purely a conditioning step.

    Note on cost: unlike EWMA (a closed-form recursion), this refits a
    likelihood by numerical optimisation every time it is called. In a
    walk-forward backtest with ~1,000+ rolling windows this is the slowest
    estimator in the module by a wide margin -- expect the Section 8b
    backtest cells to take noticeably longer to run than any other cell in
    the notebook.
    """
    if not GARCH_AVAILABLE:
        raise ImportError(
            "GARCH(1,1) requires the `arch` package, which is not installed "
            "in this environment. Install it with `pip install arch` (or "
            "`pip install -r requirements.txt`) and restart the app. "
            "Every other VaR model in this project works without it."
        )

    from arch import arch_model

    r = np.asarray(returns, dtype=float) * 100.0

    if r.size < 30:
        raise ValueError("GARCH(1,1) needs a reasonably long window (>= 30 obs).")

    model = arch_model(r, mean="Zero", vol="GARCH", p=p, q=q, dist="normal", rescale=False)

    # The underlying scipy optimiser occasionally reports a non-fatal
    # convergence code as a raw print rather than a catchable warnings.warn,
    # so both channels are silenced -- the fitted result is still used
    # normally either way; this only suppresses console noise.
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        result = model.fit(disp="off")

    forecast = result.forecast(horizon=1, reindex=False)
    sigma_next_pct = float(np.sqrt(forecast.variance.values[-1, 0]))

    return sigma_next_pct / 100.0


def garch_var(returns, alpha: float = 0.05, p: int = 1, q: int = 1):
    """
    GARCH(1,1) VaR, following the same zero-mean RiskMetrics-style convention
    as `ewma_var`: -(z_alpha * sigma_garch). The zero-drift assumption is
    dropped for the same reason given there -- over a one-day horizon the
    estimated mean adds more variance to the VaR estimate than its removal
    corrects for bias.

    This is the direct comparison to `ewma_var`: same convention, same
    walk-forward usage, different estimate of tomorrow's volatility.
    """
    sigma = garch_volatility(returns, p=p, q=q)
    return -(stats.norm.ppf(alpha) * sigma)


# ---------------------------------------------------------------------------
# 6c. EXPECTED SHORTFALL
#
# Basel's Fundamental Review of the Trading Book replaced 99% VaR with 97.5%
# Expected Shortfall as the regulatory market-risk measure. Two reasons, both
# of which this project can demonstrate rather than assert:
#
#   1. VaR says nothing about how bad a breach is. Two portfolios with
#      identical VaR can have very different losses beyond it.
#   2. VaR is not sub-additive, so it can report that splitting a portfolio
#      reduces total risk. ES is coherent and cannot.
#
# ES at level alpha is the expected loss CONDITIONAL on the loss exceeding
# VaR at that same level. It is returned as a positive number.
# ---------------------------------------------------------------------------

def historical_es(returns: pd.Series, alpha: float = 0.025):
    """Empirical ES: the mean of all returns at or below the alpha-quantile."""
    returns = pd.Series(returns).dropna()
    cutoff = np.percentile(returns, alpha * 100)
    tail = returns[returns <= cutoff]

    if tail.empty:
        return np.nan

    return float(-tail.mean())


def parametric_es(returns: pd.Series, alpha: float = 0.025):
    """
    Normal ES in closed form:

        ES = -mu + sigma * phi(z_alpha) / alpha

    where phi is the standard Normal density. Unlike the historical estimator
    this uses every observation to estimate two parameters rather than
    averaging the handful of points in the tail, so it is far more stable —
    at the cost of assuming a distribution Section 2 already rejected.
    """
    mu, sigma = returns.mean(), returns.std()
    z = stats.norm.ppf(alpha)
    return float(-mu + sigma * stats.norm.pdf(z) / alpha)


def student_t_es(returns: pd.Series, alpha: float = 0.025):
    """
    Student-t ES in closed form. For a location-scale t with df v,

        E[T | T <= t_alpha] = -(1/alpha) * f_v(t_alpha) * (v + t_alpha^2)/(v - 1)

    and ES = -(loc + scale * that quantity).

    The mean of a Student-t only exists for v > 1, so ES is undefined below
    that. Returns NaN in that case rather than a misleading number.
    """
    df, loc, scale = stats.t.fit(returns)

    if df <= 1:
        return np.nan

    t_alpha = stats.t.ppf(alpha, df)
    conditional_mean = -(1.0 / alpha) * stats.t.pdf(t_alpha, df) * (df + t_alpha ** 2) / (df - 1.0)

    return float(-(loc + scale * conditional_mean))


def es_summary(returns: pd.Series, alpha: float = 0.025):
    """VaR and ES side by side for all three distributional assumptions."""
    return pd.DataFrame(
        {
            "VaR": [
                historical_var(returns, alpha),
                parametric_var(returns, alpha),
                student_t_var(returns, alpha),
            ],
            "ES": [
                historical_es(returns, alpha),
                parametric_es(returns, alpha),
                student_t_es(returns, alpha),
            ],
        },
        index=["Historical", "Normal", "Student-t"],
    ).assign(ES_to_VaR=lambda d: d["ES"] / d["VaR"])


# ---------------------------------------------------------------------------
# 7. WALK-FORWARD BACKTESTING
# ---------------------------------------------------------------------------

VAR_METHODS = ("historical", "parametric", "student_t", "ewma") + (
    ("garch",) if GARCH_AVAILABLE else ()
)


def rolling_out_of_sample_var(
    returns: pd.Series,
    window: int = 250,
    alpha: float = 0.05,
    method: str = "historical",
    lam: float = EWMA_LAMBDA,
):
    """
    Walk-forward out-of-sample VaR backtest with no look-ahead bias.

    For each day t, VaR is estimated from returns[t-window : t] — strictly
    before t — and then tested against the realised return on day t. The
    window then advances by one day. Nothing in the estimate for day t has
    seen day t or anything after it.

    method: 'historical' | 'parametric' | 'student_t' | 'ewma' | 'garch'

    Returns a DataFrame indexed by date with columns
    ['var_estimate', 'actual_return', 'violation'].
    """
    if method not in VAR_METHODS:
        raise ValueError(f"method must be one of {VAR_METHODS}")

    values = returns.values
    dates = returns.index
    records = []

    for i in range(window, len(values)):
        trailing = pd.Series(values[i - window:i])

        if method == "historical":
            var_est = historical_var(trailing, alpha)
        elif method == "parametric":
            var_est = parametric_var(trailing, alpha)
        elif method == "student_t":
            var_est = student_t_var(trailing, alpha)
        elif method == "ewma":
            var_est = ewma_var(trailing, alpha, lam=lam)
        elif method == "garch":
            var_est = garch_var(trailing, alpha)
        else:
            # Unreachable given the VAR_METHODS check above -- kept explicit
            # rather than folded into a catch-all `else`, which previously
            # made it possible for a new method name to silently reuse
            # another estimator's branch.
            raise ValueError(f"No estimator wired up for method='{method}'")

        actual = values[i]
        records.append((dates[i], var_est, actual, bool(actual < -var_est)))

    return pd.DataFrame(
        records,
        columns=["date", "var_estimate", "actual_return", "violation"],
    ).set_index("date")


def kupiec_pof_test(n_violations: int, n_obs: int, alpha: float = 0.05):
    """
    Kupiec (1995) Proportion-of-Failures test of UNCONDITIONAL coverage.

    H0: the true violation probability equals the model's stated alpha.

        LR_POF = -2 * [ log L(alpha) - log L(pi_hat) ]   ~   chi2(1)

    where log L(p) = (T - x) log(1 - p) + x log(p).

    p < 0.05 rejects the model: it produced significantly more or fewer
    exceptions than it promised.

    What this test CANNOT see: whether the violations were spread evenly or
    all arrived in one week. A model that breaches ten times in March and
    never again passes Kupiec while being useless for risk management. That
    is what the Christoffersen tests below are for.
    """
    x, T = int(n_violations), int(n_obs)
    pi_hat = x / T if T else np.nan

    log_l0 = _safe_log_likelihood([T - x, x], [1 - alpha, alpha])
    log_l1 = _safe_log_likelihood([T - x, x], [1 - pi_hat, pi_hat])

    lr_stat = -2.0 * (log_l0 - log_l1)
    p_value = stats.chi2.sf(lr_stat, df=1)

    return {
        "test": "Kupiec POF (unconditional coverage)",
        "observed_violations": x,
        "expected_violations": round(alpha * T, 2),
        "observed_rate": pi_hat,
        "expected_rate": alpha,
        "LR_stat": lr_stat,
        "p_value": p_value,
        "reject_model_at_5pct": bool(p_value < 0.05),
    }


def violation_transitions(violations):
    """
    Count the four first-order Markov transitions in a violation sequence.

    n_ij = number of days where the indicator moved from state i to state j,
    with 1 meaning 'a violation occurred'. n11 is therefore the number of
    violations immediately following another violation — the quantity that
    detects clustering.
    """
    v = np.asarray(pd.Series(violations).dropna()).astype(bool)

    previous, current = v[:-1], v[1:]

    return {
        "n00": int(np.sum(~previous & ~current)),
        "n01": int(np.sum(~previous & current)),
        "n10": int(np.sum(previous & ~current)),
        "n11": int(np.sum(previous & current)),
    }


def christoffersen_independence_test(violations):
    """
    Christoffersen (1998) test of INDEPENDENCE of VaR violations.

    H0: a violation today is independent of whether there was a violation
    yesterday, i.e. pi_01 == pi_11.

        pi_01 = n01 / (n00 + n01)   P(violation | no violation yesterday)
        pi_11 = n11 / (n10 + n11)   P(violation | violation yesterday)
        pi    = (n01 + n11) / T     pooled violation rate

        LR_IND = -2 * [ log L_pooled - log L_Markov ]   ~   chi2(1)

    p < 0.05 means violations CLUSTER. Economically that is the more serious
    failure mode: it says the model does not react to changing volatility, so
    breaches arrive in bursts exactly when a risk manager most needs the
    number to be right.
    """
    counts = violation_transitions(violations)
    n00, n01, n10, n11 = counts["n00"], counts["n01"], counts["n10"], counts["n11"]

    total = n00 + n01 + n10 + n11
    pi_01 = n01 / (n00 + n01) if (n00 + n01) else np.nan
    pi_11 = n11 / (n10 + n11) if (n10 + n11) else np.nan
    pi = (n01 + n11) / total if total else np.nan

    log_l_pooled = _safe_log_likelihood([n00 + n10, n01 + n11], [1 - pi, pi])
    log_l_markov = _safe_log_likelihood(
        [n00, n01, n10, n11],
        [1 - pi_01, pi_01, 1 - pi_11, pi_11],
    )

    lr_stat = -2.0 * (log_l_pooled - log_l_markov)
    p_value = stats.chi2.sf(lr_stat, df=1)

    return {
        "test": "Christoffersen independence",
        **counts,
        "p_violation_given_calm": pi_01,
        "p_violation_given_violation": pi_11,
        "pooled_violation_rate": pi,
        "LR_stat": lr_stat,
        "p_value": p_value,
        "reject_independence_at_5pct": bool(p_value < 0.05),
    }


def christoffersen_conditional_coverage_test(violations, alpha: float = 0.05):
    """
    Christoffersen joint test of CONDITIONAL coverage.

        LR_CC = LR_POF + LR_IND   ~   chi2(2)

    The two component statistics are asymptotically independent, so they add.
    This is the test a model has to pass to be usable: the right NUMBER of
    violations (Kupiec) arriving at the right TIMES (independence).

    A model can fail this in three distinguishable ways, and reporting the
    components separately says which:
      - correct count, clustered timing  -> passes Kupiec, fails independence
      - wrong count, random timing       -> fails Kupiec, passes independence
      - both                             -> fails both
    """
    v = pd.Series(violations).dropna().astype(bool)

    pof = kupiec_pof_test(int(v.sum()), len(v), alpha=alpha)
    ind = christoffersen_independence_test(v)

    lr_cc = pof["LR_stat"] + ind["LR_stat"]
    p_value = stats.chi2.sf(lr_cc, df=2)

    return {
        "test": "Christoffersen conditional coverage",
        "LR_POF": pof["LR_stat"],
        "LR_IND": ind["LR_stat"],
        "LR_CC": lr_cc,
        "p_value": p_value,
        "reject_model_at_5pct": bool(p_value < 0.05),
        "kupiec": pof,
        "independence": ind,
    }


def full_backtest_report(backtest: pd.DataFrame, alpha: float = 0.05, label: str = ""):
    """
    Run the complete validation battery on the output of
    `rolling_out_of_sample_var` and return one flat row.

    Reporting all three tests together is the point of the exercise: a single
    p-value invites the reader to treat validation as pass/fail, whereas the
    three together say what specifically is wrong with the model.
    """
    violations = backtest["violation"].astype(bool)

    pof = kupiec_pof_test(int(violations.sum()), len(violations), alpha=alpha)
    ind = christoffersen_independence_test(violations)
    ccov = christoffersen_conditional_coverage_test(violations, alpha=alpha)

    return {
        "method": label,
        "n_obs": len(violations),
        "violations": pof["observed_violations"],
        "expected": pof["expected_violations"],
        "observed_rate": pof["observed_rate"],
        "kupiec_LR": pof["LR_stat"],
        "kupiec_p": pof["p_value"],
        "kupiec_reject": pof["reject_model_at_5pct"],
        "independence_LR": ind["LR_stat"],
        "independence_p": ind["p_value"],
        "clustering_detected": ind["reject_independence_at_5pct"],
        "cc_LR": ccov["LR_CC"],
        "cc_p": ccov["p_value"],
        "cc_reject": ccov["reject_model_at_5pct"],
        "p_violation_given_calm": ind["p_violation_given_calm"],
        "p_violation_given_violation": ind["p_violation_given_violation"],
    }


# ---------------------------------------------------------------------------
# 7b. BASEL TRAFFIC-LIGHT BACKTESTING ZONES
#
# A statistical verdict (reject / fail to reject at 5%) is not what a bank
# actually acts on. Basel's Internal Models Approach (IMA) translates a VaR
# backtest's exception count directly into a capital multiplier via a
# green/yellow/red "traffic light" classification. This is the regulatory
# consequence of the same violation counts already produced above.
# ---------------------------------------------------------------------------

_BASEL_ZONE_BOUNDARIES = [
    # (upper bound on exceptions inclusive, zone, multiplier add-on)
    (4, "green", 0.00),
    (5, "yellow", 0.40),
    (6, "yellow", 0.50),
    (7, "yellow", 0.65),
    (8, "yellow", 0.75),
    (9, "yellow", 0.85),
]
_BASEL_BASE_MULTIPLIER = 3.00
_BASEL_OFFICIAL_WINDOW = 250


def basel_traffic_light(n_violations: int, n_obs: int = 250):
    """
    Basel Internal Models Approach 'traffic-light' backtesting classification.

    Officially defined for 99% VaR backtested over a 250-trading-day window.
    The exception count maps to a zone and an add-on to the capital
    multiplier k (base k = 3.00):

        Exceptions (of 250)   Zone     Add-on   Multiplier
        0 - 4                 Green    0.00     3.00
        5                     Yellow   0.40     3.40
        6                     Yellow   0.50     3.50
        7                     Yellow   0.65     3.65
        8                     Yellow   0.75     3.75
        9                     Yellow   0.85     3.85
        10+                   Red      1.00     4.00

    When n_obs != 250 (as in this project's longer out-of-sample windows),
    the boundaries are scaled proportionally (boundary * n_obs / 250,
    rounded to the nearest integer) so the framework can still be applied
    for illustration. That scaling is NOT part of the official Basel text --
    it is an approximation used here, and `is_official_250_day_window` in
    the returned dict says so explicitly rather than silently applying an
    out-of-scope table as if it were the real one.
    """
    scale = n_obs / _BASEL_OFFICIAL_WINDOW

    zone, addon = "red", 1.00
    for boundary, zone_name, add_on in _BASEL_ZONE_BOUNDARIES:
        scaled_boundary = round(boundary * scale)
        if n_violations <= scaled_boundary:
            zone, addon = zone_name, add_on
            break

    is_official = n_obs == _BASEL_OFFICIAL_WINDOW

    return {
        "n_violations": int(n_violations),
        "n_obs": int(n_obs),
        "zone": zone,
        "multiplier_addon": addon,
        "capital_multiplier": _BASEL_BASE_MULTIPLIER + addon,
        "is_official_250_day_window": is_official,
        "note": (
            "Exact Basel table (n_obs = 250)."
            if is_official
            else f"n_obs={n_obs} != 250; boundaries scaled proportionally as "
                 "an illustration, not the official Basel calibration."
        ),
    }


# ---------------------------------------------------------------------------
# 8. TRANSLATE A RISK NUMBER INTO A CURRENCY LOSS
# ---------------------------------------------------------------------------

def var_to_exposure_loss(var_estimate: float, exposure_amount: float):
    """
    Convert a VaR or ES percentage into a currency loss for a given exposure.

    Pure arithmetic, included because a risk number is only actionable once it
    is expressed in the units a decision-maker allocates capital in.
    """
    return var_estimate * exposure_amount
