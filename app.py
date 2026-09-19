"""
app.py
======
Streamlit dashboard for the Market Risk Statistical Validation project.

STRUCTURE
    A guided narrative, not a set of independent tabs: an Executive Summary
    opens with the live headline result, each following chapter answers the
    question the previous chapter's finding raised, and a Verdict chapter
    closes the loop. The sidebar's Story list lets you jump directly to any
    chapter; the button at the foot of each chapter moves through them in
    order, which is the intended first read.

AIM
    Take five standard one-day Value-at-Risk models, run them out of sample on
    Indian equity data, and decide with formal statistical tests which ones are
    actually calibrated — rather than reporting a single VaR number and stopping.

MOTIVATION
    A VaR figure on its own is unfalsifiable. Basel's traffic-light framework
    exists precisely because a bank's own model will always look reasonable
    until someone counts its exceptions and asks whether the count, and the
    TIMING of that count, is consistent with the confidence level claimed. This
    dashboard is that counting exercise made interactive.

This file handles layout, visualisation and display only. Every statistical
calculation lives in risk_utils.py, so the notebook and the dashboard cannot
disagree with each other.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import streamlit as st

import risk_utils as ru


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Market Risk Statistical Validation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# LOAD CUSTOM CSS
# ============================================================

try:
    with open("style.css", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True,
        )
except FileNotFoundError:
    st.warning(
        "style.css was not found. The dashboard will use the default "
        "Streamlit styling."
    )


# ============================================================
# VISUAL DESIGN REGISTER
# ============================================================

CHART_BG = "#fffdf9"
TEXT = "#2b211b"
MUTED = "#786b61"
GRID = "#e7dfd7"

ACCENT = "#6f4e37"
ACCENT_DARK = "#513724"
SECONDARY = "#8b7564"
RISK = "#9a4a3f"

# One colour per VaR method, used consistently in every chart and table so a
# reader can follow a single model across the whole dashboard.
METHOD_COLOURS = {
    "historical": ACCENT,
    "parametric": SECONDARY,
    "student_t": "#5f7f6f",
    "ewma": "#8a6d9e",
    "garch": "#b5723f",
}

METHOD_LABELS = {
    "historical": "Historical",
    "parametric": "Normal Parametric",
    "student_t": "Student-t Parametric",
    "ewma": "EWMA (RiskMetrics)",
    "garch": "GARCH(1,1)",
}

# Basel traffic-light zone -> the verdict-pill CSS class it should render
# with. Reuses the existing .verdict.pass / .warning / .fail classes from
# style.css rather than inventing new ones, since green/yellow/red map onto
# exactly that pass/caution/fail semantic already used everywhere else.
ZONE_COLOURS = {"green": "pass", "yellow": "warning", "red": "fail"}

def style_chart(ax):
    """Apply the common visual language to all Matplotlib charts."""

    ax.set_facecolor(CHART_BG)

    # Remove unnecessary chart borders.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("#cfc4b9")
    ax.spines["bottom"].set_color("#cfc4b9")

    # Minimal ticks.
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

    # Light horizontal grid for quantitative reading.
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)

    ax.set_axisbelow(True)

    # Title styling.
    ax.title.set_color(TEXT)
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("600")

    # Legend styling.
    legend = ax.get_legend()

    if legend is not None:
        legend.get_frame().set_facecolor(CHART_BG)
        legend.get_frame().set_edgecolor("#ddd3ca")
        legend.get_frame().set_alpha(0.95)

        for text in legend.get_texts():
            text.set_color(TEXT)
            text.set_fontsize(8.5)


def format_pvalue(p):
    """Display very small p-values without misleading rounding."""

    if pd.isna(p):
        return "N/A"

    if p == 0:
        return "<1e-300"

    if p < 0.0001:
        return f"{p:.2e}"

    return f"{p:.4f}"

def verdict_pill(passed: bool, pass_text: str, fail_text: str) -> str:
    """Small HTML pill for plain-English statistical verdicts."""

    cls, label = ("pass", pass_text) if passed else ("fail", fail_text)

    return f'<span class="verdict {cls}">{label}</span>'


def section_note(text: str):
    """Display plain-English framing at the top of each tab."""

    st.markdown(
        f'<div class="section-note">{text}</div>',
        unsafe_allow_html=True,
    )


def takeaway(text: str):
    """
    Close every chapter with the one sentence a reader should leave with.

    Included deliberately: the failure mode of a statistics dashboard is a wall
    of correct numbers that never states a conclusion.
    """
    st.markdown(
        f'<div class="section-note"><strong>Takeaway — </strong>{text}</div>',
        unsafe_allow_html=True,
    )


CHAPTERS = [
    {"key": "hero", "nav": "Executive Summary", "title": "The Question",
     "hook": "Can we actually trust a VaR model, or only its headline number?"},
    {"key": "distribution", "nav": "1 · Distribution", "title": "Do Returns Behave Like the Textbook Assumes?",
     "hook": "Every parametric VaR model starts by assuming a distribution. Is that assumption even true?"},
    {"key": "regimes", "nav": "2 · Volatility Regimes", "title": "Does Risk Cluster Over Time?",
     "hook": "If volatility comes in regimes, a single 'average' risk number is already misleading."},
    {"key": "beta", "nav": "3 · Systematic Risk", "title": "How Much Risk Is Market-Wide?",
     "hook": "Some of a stock's risk is the market's risk. How much, and how confidently can we say so?"},
    {"key": "errors", "nav": "4 · Real-Time Detection", "title": "Can a Simple Rule Catch It in Real Time?",
     "hook": "Regimes are only knowable in hindsight. Could a cheap, live rule substitute for one?"},
    {"key": "estimate", "nav": "5 · Estimating the Loss", "title": "Five Ways to Price Tomorrow's Risk",
     "hook": "Different assumptions about distribution and volatility produce several different VaR numbers."},
    {"key": "validate", "nav": "6 · Out-of-Sample Validation", "title": "Did the Models Actually Hold Up?",
     "hook": "A VaR estimate is a claim. Walking it forward, day by day, is how that claim gets tested."},
    {"key": "verdict", "nav": "7 · The Verdict", "title": "What Survived, and What It Means",
     "hook": "Pulling every finding above into the one conclusion that actually matters."},
]
CHAPTER_KEYS = [c["key"] for c in CHAPTERS]
CHAPTER_BY_KEY = {c["key"]: c for c in CHAPTERS}


def chapter_header(key: str):
    """Render the numbered kicker, serif title, hook line and progress rail
    that open every chapter — the visual device that makes this read as one
    continuous argument rather than a set of unrelated tabs."""

    idx = CHAPTER_KEYS.index(key)
    chapter = CHAPTER_BY_KEY[key]

    st.markdown(
        f'<p class="chapter-kicker">Chapter {idx} of {len(CHAPTERS) - 1}'
        f' &nbsp;·&nbsp; {chapter["nav"]}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<h2 class="chapter-title">{chapter["title"]}</h2>', unsafe_allow_html=True)
    st.markdown(f'<p class="chapter-hook">{chapter["hook"]}</p>', unsafe_allow_html=True)

    st.progress(idx / (len(CHAPTERS) - 1))
    st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)


def _goto_chapter(target_key: str):
    """Navigate to a chapter from anywhere in the app (Next/Prev buttons, the
    Verdict chapter's back-link, etc). Only ever touches the logical
    'chapter' state, never the sidebar radio's own widget key directly --
    the radio is synced from 'chapter' at the top of the script, before it
    renders, on the next run that st.rerun() triggers. Writing to the
    widget's key from here (after it has already rendered this run) is
    exactly what Streamlit blocks."""
    st.session_state["chapter"] = target_key
    st.rerun()


def chapter_nav(key: str):
    """Previous / Next buttons at the foot of every chapter, so the default
    way through the app is to read it in order — the sidebar's Story list is
    there for jumping around, not for the first pass."""

    idx = CHAPTER_KEYS.index(key)
    st.markdown("---")
    prev_col, spacer, next_col = st.columns([1, 2, 1])

    with prev_col:
        if idx > 0:
            prev_chapter = CHAPTERS[idx - 1]
            if st.button(f"← {prev_chapter['nav']}", use_container_width=True):
                _goto_chapter(prev_chapter["key"])

    with next_col:
        if idx < len(CHAPTERS) - 1:
            next_chapter = CHAPTERS[idx + 1]
            if st.button(f"{next_chapter['nav']} →", type="primary", use_container_width=True):
                _goto_chapter(next_chapter["key"])
# ============================================================
# HEADER — small and persistent; the argument itself opens in
# the Executive Summary chapter, not in a wall of text here.
# ============================================================

top1, top2 = st.columns([3, 1])

with top1:
    st.markdown(
        '<p class="app-kicker">Testing whether standard VaR models survive out-of-sample validation</p>'
        '<h1 style="margin-top:0;">Market Risk Statistical Validation</h1>',
        unsafe_allow_html=True,
    )

with top2:
    st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)
# SIDEBAR — CONFIGURATION
# ============================================================

st.sidebar.markdown(
    '<p class="app-kicker" style="margin-bottom:0.1rem;">Market Risk Validation</p>',
    unsafe_allow_html=True,
)
st.sidebar.header("Get started")

with st.sidebar.expander("Data & parameters", expanded=False):

    market_ticker = st.text_input("Market index ticker", "^NSEI")

    stock_input = st.text_input(
        "Stock tickers (comma-separated)",
        "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS",
    )

    stock_tickers = [
        ticker.strip()
        for ticker in stock_input.split(",")
        if ticker.strip()
    ]

    start_date = st.date_input("Start date", pd.to_datetime("2019-01-01"))
    end_date = st.date_input("End date", pd.to_datetime("2024-12-31"))

    st.markdown("---")

    regime_window = st.slider(
        "Rolling volatility window (days)",
        10,
        60,
        21,
    )

    var_alpha_pct = st.select_slider(
        "VaR confidence level",
        options=[90, 95, 99],
        value=95,
    )

    es_alpha_pct = st.select_slider(
        "Expected Shortfall level",
        options=[95.0, 97.5, 99.0],
        value=97.5,
        help="Basel FRTB uses 97.5% ES as the regulatory capital measure.",
    )

    flag_pct = st.select_slider(
        "Stress-flag percentile",
        options=[1, 5, 10],
        value=5,
    )

    use_snapshot = st.checkbox(
        "Prefer committed data snapshot",
        value=True,
        help=(
            "Reads data/prices.csv if it exists in this project folder, "
            "which would make results reproducible without a live Yahoo "
            "Finance call. No such file is committed in this repo, so this "
            "currently has no effect and every run downloads live."
        ),
    )

st.sidebar.caption(
    "Defaults match the settings this project's documented findings were "
    "run at. Data downloads live from Yahoo Finance each run, so exact "
    "figures can drift slightly as new trading days become available. Open "
    "**Data & parameters** to change tickers, dates, or confidence levels."
)

run = st.sidebar.button(
    "▶  Run analysis",
    type="primary",
    use_container_width=True,
)

st.sidebar.caption(
    "Backtests are cached, so moving between chapters does not recompute them."
)


# ============================================================# SESSION STATE
# ============================================================

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

if run:
    st.session_state.analysis_started = True


# ============================================================
# CACHED COMPUTATION
# ============================================================
# The walk-forward backtest re-fits a distribution on every one of ~1,200
# rolling windows for each of five methods (GARCH refits an MLE optimisation
# on every window and is the slowest of the five). Without caching, Streamlit reruns
# the whole thing on every widget interaction, which made the original version
# unusable once the Student-t MLE was added.

@st.cache_data(show_spinner=False)
def load_data(tickers, start, end, prefer_snapshot):
    """
    Load prices, preferring the committed snapshot over a live download.

    Returns the frame and a note describing where it came from, so the UI can
    be honest about which data the numbers on screen were computed from.
    """
    if prefer_snapshot:
        frame = ru.load_prices(tickers, start, end)
        source = (
            "committed snapshot (data/prices.csv)"
            if ru.DEFAULT_SNAPSHOT.exists()
            else "live download from Yahoo Finance"
        )
    else:
        frame = ru.download_prices(tickers, start, end)
        source = "live download from Yahoo Finance"

    return frame, source


@st.cache_data(show_spinner=False)
def run_backtests(returns, window, alpha, lam):
    """Walk-forward backtest for all five VaR methods, computed once."""

    return {
        method: ru.rolling_out_of_sample_var(
            returns,
            window=window,
            alpha=alpha,
            method=method,
            lam=lam,
        )
        for method in ru.VAR_METHODS
    }


@st.cache_data(show_spinner=False)
def validation_table(returns, window, alpha, lam):
    """Kupiec, Christoffersen independence and joint coverage for every method."""

    backtests = run_backtests(returns, window, alpha, lam)

    return pd.DataFrame(
        [
            ru.full_backtest_report(bt, alpha=alpha, label=METHOD_LABELS[method])
            for method, bt in backtests.items()
        ]
    )

# ============================================================
# DATA LOADING
# ============================================================

if not st.session_state.analysis_started:
    st.info(
        "Set your configuration in the sidebar and click "
        "**Run analysis** to begin."
    )
    st.stop()


if start_date >= end_date:
    st.error("Start date must be earlier than end date.")
    st.stop()


with st.spinner("Loading price data..."):

    all_tickers = [market_ticker] + stock_tickers

    try:
        prices, data_source = load_data(
            all_tickers,
            str(start_date),
            str(end_date),
            use_snapshot,
        )

        prices = prices.rename(columns={market_ticker: "MARKET"})

    except Exception as e:
        st.error(
            "Could not load price data. This is almost always Yahoo "
            "Finance rate-limiting or a network issue, not a bug here — "
            "wait a few minutes and click **Run analysis** again, or check "
            f"you have an internet connection. Details: {e}"
        )
        st.stop()


if "MARKET" not in prices.columns:
    st.error(
        "The selected market ticker could not be loaded. "
        "Check the market index ticker."
    )
    st.stop()

returns = ru.log_returns(prices)
market_returns = returns["MARKET"]

if market_returns.empty:
    st.error(
        f"Loaded a price table ({prices.shape[0]} rows, columns: "
        f"{', '.join(prices.columns)}) but it produced zero market return "
        "observations after computing log returns — almost always because "
        "the live Yahoo Finance download came back empty or near-empty "
        "(common: rate-limiting). Wait a few minutes and click **Run "
        "analysis** again; if it persists, try a different network."
    )
    st.stop()

# Display label for the index, so the copy never hard-codes NIFTY 50 while the
# user is looking at a different ticker.
MARKET_LABEL = market_ticker

alpha = 1 - var_alpha_pct / 100
es_alpha = round(1 - es_alpha_pct / 100, 6)

st.caption(
    f"{len(market_returns):,} daily observations for {MARKET_LABEL} "
    f"({market_returns.index.min():%d %b %Y} to "
    f"{market_returns.index.max():%d %b %Y}) — source: {data_source}."
)


# ============================================================
# EXECUTIVE-SUMMARY DATA — computed once at fixed, documented
# defaults (250-day window) so the Executive Summary and Verdict
# chapters can state a real, live result no matter which chapter
# the sidebar's in-chapter sliders are currently set to.
# ============================================================

DEFAULT_WINDOW = 250

with st.spinner(f"Preparing executive summary ({len(ru.VAR_METHODS)}-model backtest at default settings)..."):
    hero_reports = validation_table(market_returns, DEFAULT_WINDOW, alpha, ru.EWMA_LAMBDA)

hero_es = ru.es_summary(market_returns, es_alpha)

hero_survivors = [row["method"] for _, row in hero_reports.iterrows() if not row["cc_reject"]]
hero_clustered = [row["method"] for _, row in hero_reports.iterrows() if row["clustering_detected"]]
hero_count_only_failures = [
    row["method"] for _, row in hero_reports.iterrows()
    if (not row["kupiec_reject"]) and row["clustering_detected"]
]

# ============================================================
# SIDEBAR — STORY NAVIGATION
# ============================================================

if "chapter" not in st.session_state:
    st.session_state["chapter"] = "hero"

# Sync the radio widget's own backing key from our logical 'chapter' state
# BEFORE the widget renders below -- this is the one place in the script
# where writing to '_chapter_radio' is legal, since the widget hasn't been
# instantiated yet this run. Next/Prev buttons (chapter_nav, further down
# the page) only ever touch 'chapter', never '_chapter_radio' directly --
# writing to a widget's own key AFTER it has already rendered in the same
# run is what Streamlit blocks, which is exactly what broke on Next/Prev
# clicks before this fix.
st.session_state["_chapter_radio"] = st.session_state["chapter"]

st.sidebar.markdown("---")
st.sidebar.header("Story")


def _sync_chapter_from_radio():
    st.session_state["chapter"] = st.session_state["_chapter_radio"]


st.sidebar.radio(
    "Chapters",
    options=CHAPTER_KEYS,
    format_func=lambda k: CHAPTER_BY_KEY[k]["nav"],
    key="_chapter_radio",
    on_change=_sync_chapter_from_radio,
    label_visibility="collapsed",
)

_idx = CHAPTER_KEYS.index(st.session_state["chapter"])
st.sidebar.caption(f"Chapter {_idx} of {len(CHAPTERS) - 1} — read in order, or jump ahead.")

chapter = st.session_state["chapter"]

if chapter == "hero":

    chapter_header("hero")

    st.markdown(
        "A VaR number on its own is unfalsifiable — nothing about a single "
        f"figure can be checked. This project takes {len(ru.VAR_METHODS)} VaR models, walks "
        "each one forward one unseen day at a time, and asks the two "
        "questions a regulator actually asks: did it breach the right "
        "**number** of times, and did those breaches arrive at "
        "**independent times** rather than in bursts. Here is where that "
        f"lands, at the standard {DEFAULT_WINDOW}-day window and "
        f"{var_alpha_pct}% confidence level:"
    )

    h1, h2, h3 = st.columns(3)

    with h1:
        st.metric(
            "Models tested",
            f"{len(hero_reports)}",
            help="Historical, Normal Parametric, Student-t Parametric, EWMA, GARCH(1,1).",
        )

    with h2:
        st.metric(
            "Survive the joint test",
            f"{len(hero_survivors)} / {len(hero_reports)}",
            help="Pass BOTH Kupiec (right count) and Christoffersen independence (right timing) at 5%.",
        )

    with h3:
        st.metric(
            "Right count, wrong timing",
            f"{len(hero_count_only_failures)}",
            help="Models that pass Kupiec but are still rejected for clustered breach timing — the central failure mode this project is built to catch.",
        )

    if hero_count_only_failures:
        st.markdown(
            "<div class='section-note'><strong>The headline finding — </strong>"
            f"{', '.join(hero_count_only_failures)} "
            f"{'each has' if len(hero_count_only_failures) == 1 else 'each have'} "
            "the statistically correct number of VaR breaches, and would pass "
            "a validation report that stops at Kupiec. Christoffersen's "
            "independence test rejects "
            f"{'it' if len(hero_count_only_failures) == 1 else 'them'} anyway, "
            "because the breaches cluster in time rather than arriving "
            "independently — exactly the pattern that turns a manageable "
            "limit excess into two straight weeks of losses. That gap "
            "between the two tests is what the rest of this project is "
            "built to explain.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='section-note'><strong>At this window and "
            "confidence level — </strong>every model that gets the breach "
            "count right also gets the timing right. That is not the "
            "typical case in this project; move the confidence level or "
            "window in Chapter 6 to see where it breaks down, since a "
            "validation is only informative at the specific settings it "
            "was run at.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("##### Where this argument goes")

    route = pd.DataFrame(
        [
            {"Chapter": c["nav"], "Question": c["hook"]}
            for c in CHAPTERS
            if c["key"] != "hero"
        ]
    ).set_index("Chapter")

    st.dataframe(route, use_container_width=True)

    st.caption(
        "Each chapter's finding is the reason the next chapter exists — the "
        "full reasoning chain is in the README. Start with Chapter 1, or "
        "jump straight to Chapter 6 for the validation battery this "
        "summary is drawn from."
    )

    chapter_nav("hero")

if chapter == "distribution":

    chapter_header("distribution")

    section_note(
        "Do returns follow a Normal distribution, or are extreme moves more "
        "common than a Normal model would predict? This is the first question "
        "because the Normal Parametric VaR tested later assumes normality, and "
        "if that assumption fails here we should expect it to fail there too."
    )

    normal_params = ru.fit_normal(market_returns)
    t_params = ru.fit_student_t(market_returns)
    jb = ru.goodness_of_fit_normal(market_returns)

    c1, c2 = st.columns([1.3, 1])

    with c1:

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), facecolor=CHART_BG)

        for ax in axes:
            style_chart(ax)

        x = np.linspace(market_returns.min(), market_returns.max(), 500)

        # ----------------------------------------------------
        # Empirical distribution vs fitted distributions
        # ----------------------------------------------------

        axes[0].hist(
            market_returns,
            bins=60,
            density=True,
            alpha=0.42,
            color="#c9d8e6",
            edgecolor="none",
            label="Empirical",
        )

        axes[0].plot(
            x,
            stats.norm.pdf(x, **normal_params),
            color=ACCENT,
            linewidth=2.0,
            label="Normal fit",
        )

        axes[0].plot(
            x,
            stats.t.pdf(x, **t_params),
            color=SECONDARY,
            linewidth=2.0,
            label="Student-t fit",
        )

        axes[0].set_title("Empirical vs fitted distributions", loc="left")
        axes[0].set_xlabel("Daily log return")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        # ----------------------------------------------------
        # Q-Q plot vs Normal
        # ----------------------------------------------------

        stats.probplot(market_returns, dist="norm", plot=axes[1])

        axes[1].set_title("Normal Q-Q diagnostic", loc="left")
        axes[1].set_xlabel("Theoretical quantiles")
        axes[1].set_ylabel("Ordered values")

        # Style scipy-generated Q-Q elements.
        if len(axes[1].lines) >= 1:
            axes[1].lines[0].set_color(ACCENT)
            axes[1].lines[0].set_linewidth(1.8)

        if len(axes[1].lines) >= 2:
            axes[1].lines[1].set_color(SECONDARY)
            axes[1].lines[1].set_markersize(3.5)
            axes[1].lines[1].set_markeredgewidth(0)

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)

    with c2:

        st.metric(
            "Jarque-Bera p-value (market)",
            format_pvalue(jb["p_value"]),
        )

        st.markdown(
            "Reject normality at 5%: "
            + verdict_pill(
                jb["reject_normality_at_5pct"],
                "Yes — fat-tailed",
                "No — looks Normal",
            ),
            unsafe_allow_html=True,
        )

        st.metric(
            "Student-t fit — degrees of freedom",
            f"{t_params['df']:.2f}",
        )

        st.metric(
            "Excess kurtosis (Normal = 0)",
            f"{stats.kurtosis(market_returns):.2f}",
        )

        st.caption(
            "Lower degrees of freedom mean heavier tails. Note that the fitted "
            "Student-t *scale* is smaller than the sample standard deviation: "
            "the MLE buys fatter tails by tightening the body of the "
            "distribution. That trade-off is what drives the Student-t VaR "
            "result in Chapter 5, and it is not obvious until you see it."
        )

    st.markdown("##### Descriptive statistics — all series")

    descriptive_display = returns.apply(ru.descriptive_stats).T

    st.dataframe(descriptive_display, use_container_width=True)

    takeaway(
        f"returns on {MARKET_LABEL} are decisively non-Normal "
        f"(Jarque-Bera p = {format_pvalue(jb['p_value'])}, excess kurtosis "
        f"{stats.kurtosis(market_returns):.2f}), and a Student-t with "
        f"{t_params['df']:.2f} degrees of freedom fits the tails far better. "
        "Any risk model that assumes normality is therefore working against "
        "the data — Chapter 6 tests whether that assumption actually costs anything."
    )

# ============================================================
# CHAPTER 2 — VOLATILITY REGIMES
# ============================================================

    chapter_nav("distribution")

if chapter == "regimes":

    chapter_header("regimes")

    section_note(
        "Days are split into 'low-volatility' and 'high-volatility' regimes "
        "using trailing realised volatility. The median split is a RELATIVE "
        "definition — it does not claim half of all trading days were crises. "
        "The question is whether return behaviour differs across the two "
        "environments."
    )

    ex_ante = st.radio(
        "Regime labelling",
        options=[True, False],
        index=0,
        horizontal=True,
        format_func=lambda flag: (
            "Ex-ante (volatility lagged one day)"
            if flag
            else "Naive (window includes the day being labelled)"
        ),
        help=(
            "The naive version labels a day high-volatility using a window that "
            "contains that day's own return. Switch between them and watch the "
            "Levene p-value."
        ),
    )

    regime, rolling_vol = ru.classify_regime(
        market_returns,
        window=regime_window,
        method="median",
        ex_ante=ex_ante,
    )

    result = ru.compare_regimes(market_returns, regime)

    fig, ax = plt.subplots(figsize=(10, 3.4), facecolor=CHART_BG)

    style_chart(ax)

    rolling_vol.plot(
        ax=ax,
        color=ACCENT,
        linewidth=1.25,
        label=f"{regime_window}-day rolling volatility",
    )

    ax.axhline(
        rolling_vol.median(),
        color=RISK,
        linestyle="--",
        linewidth=1.25,
        label="Median low/high-vol cutoff",
    )

    ax.set_title("Trailing realised volatility", loc="left")
    ax.set_ylabel("Rolling volatility")
    ax.set_xlabel("Date")
    ax.legend()

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Welch t-test p-value (mean)",
        format_pvalue(result["t_test_p"]),
    )

    c2.metric(
        "Mann-Whitney p-value (distribution)",
        format_pvalue(result["mannwhitney_p"]),
    )

    c3.metric(
        "Levene p-value (variance)",
        format_pvalue(result["levene_p"]),
    )

    regime_counts = regime.value_counts()

    c1, c2 = st.columns(2)
    c1.metric("Low-vol observations", int(regime_counts.get("low_vol", 0)))
    c2.metric("High-vol observations", int(regime_counts.get("high_vol", 0)))

    st.markdown(
        "**Reading this:** returns are non-Normal (Chapter 1), so Mann-Whitney U is "
        "the primary non-parametric comparison of the two return samples. The "
        "Welch t-test complements it by testing mean equality while allowing "
        "unequal variances. 95% CI for the mean difference "
        f"(low-vol − high-vol): ({result['mean_diff_95ci'][0]:.5f}, "
        f"{result['mean_diff_95ci'][1]:.5f})."
    )

    st.markdown(
        "Variance differs sharply between regimes: "
        + verdict_pill(
            result["levene_p"] < 0.05,
            "Confirmed",
            "Not confirmed",
        ),
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Look-ahead bias: the same test computed both ways
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("##### What the one-day lag is actually worth")

    st.caption(
        "The same Levene test, run on both labellings. This comparison is the "
        "reason the lag is a parameter rather than a silent implementation "
        "detail — the difference is the size of a methodological error that is "
        "easy to make and invisible in the output."
    )

    leak_rows = []

    for flag in (False, True):
        regime_variant, _ = ru.classify_regime(
            market_returns,
            window=regime_window,
            method="median",
            ex_ante=flag,
        )
        variant_result = ru.compare_regimes(market_returns, regime_variant)

        leak_rows.append(
            {
                "Labelling": "Ex-ante (lagged)" if flag else "Naive (leaky)",
                "Levene p": format_pvalue(variant_result["levene_p"]),
                "Mann-Whitney p": format_pvalue(variant_result["mannwhitney_p"]),
                "Welch t p": format_pvalue(variant_result["t_test_p"]),
            }
        )

    st.dataframe(
        pd.DataFrame(leak_rows).set_index("Labelling"),
        use_container_width=True,
    )

    st.markdown(
        "Under the naive labelling a day is classified high-volatility partly "
        "*because* its own return was large in absolute value. A test of whether "
        "return dispersion differs between the regimes is then partly testing "
        "the construction of the labels rather than a property of the market. "
        "Lagging the volatility by one day makes the label a genuine information "
        "set — what an observer could have known at the open. Every regime "
        "result reported elsewhere in this project uses the lagged version."
    )

    takeaway(
        "volatility regimes differ overwhelmingly in dispersion and barely at "
        "all in mean return, which is the empirical case for modelling risk "
        "rather than trying to forecast direction. The lag comparison above is "
        "the more useful point for a reviewer: the leaky version inflates the "
        "same test statistic, and nothing in the output would have revealed it."
    )

# ============================================================
# CHAPTER 3 — BETA
# ============================================================

    chapter_nav("regimes")

if chapter == "beta":

    chapter_header("beta")

    section_note(
        "Classical market-model OLS regression: "
        "R_stock = alpha + beta · R_market + error. "
        f"Beta measures systematic sensitivity to {MARKET_LABEL} movements, "
        "while R² measures how much of the stock's return variation the market "
        "factor explains. This is the market model, not a test of CAPM — no "
        "risk-free rate is subtracted and no cross-sectional pricing claim is made."
    )

    rows = []

    for ticker in stock_tickers:

        if ticker not in returns.columns:
            continue

        res = ru.market_model_regression(returns[ticker], market_returns)

        rows.append(
            {
                "stock": ticker,
                "alpha": res["alpha"],
                "alpha_p": res["alpha_p"],
                "beta": res["beta"],
                "beta_p": res["beta_p"],
                "beta_ci_low": res["beta_ci95"][0],
                "beta_ci_high": res["beta_ci95"][1],
                "r_squared": res["r_squared"],
            }
        )

    if not rows:
        st.warning("No stock series were available for market-model regression.")
    else:

        reg_table = pd.DataFrame(rows).set_index("stock")

        c1, c2 = st.columns([1, 1.2])

        with c1:

            fig, ax = plt.subplots(figsize=(6.2, 3.8), facecolor=CHART_BG)

            style_chart(ax)

            beta = reg_table["beta"]

            beta_err = np.vstack(
                [
                    beta - reg_table["beta_ci_low"],
                    reg_table["beta_ci_high"] - beta,
                ]
            )

            ax.bar(
                reg_table.index,
                beta,
                yerr=beta_err,
                capsize=4,
                width=0.58,
                color=ACCENT,
                edgecolor="none",
                error_kw={
                    "elinewidth": 1.2,
                    "ecolor": TEXT,
                    "capthick": 1.2,
                },
            )

            ax.axhline(
                1.0,
                color=RISK,
                linestyle="--",
                linewidth=1.3,
                label="Market beta = 1",
            )

            ax.set_title("Systematic market exposure", loc="left")
            ax.set_ylabel("Beta")
            ax.legend()

            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()

            st.pyplot(fig, use_container_width=True)

        with c2:

            reg_display = reg_table.copy()

            for col in ["alpha", "beta", "beta_ci_low", "beta_ci_high", "r_squared"]:
                if col in reg_display.columns:
                    reg_display[col] = reg_display[col].map(lambda v: f"{v:.4f}")

            for col in ["alpha_p", "beta_p"]:
                if col in reg_display.columns:
                    reg_display[col] = reg_display[col].map(format_pvalue)

            st.dataframe(reg_display, use_container_width=True)

        st.caption(
            "Inference uses HC1 heteroskedasticity-robust standard errors "
            "because daily equity returns exhibit volatility clustering, which "
            "violates the constant-variance assumption behind classical OLS "
            "standard errors. The alpha and beta point estimates are unchanged; "
            "only their estimated uncertainty is corrected. A 95% beta interval "
            "lying entirely below or above 1 is evidence that the stock's "
            "systematic sensitivity genuinely differs from the index."
        )

        distinct = reg_table[
            (reg_table["beta_ci_high"] < 1) | (reg_table["beta_ci_low"] > 1)
        ]

        takeaway(
            f"{len(distinct)} of {len(reg_table)} stocks have a 95% beta "
            "interval that excludes 1, so their systematic exposure is "
            "statistically distinguishable from the index rather than merely "
            "different in the point estimate. R² also separates the names "
            f"({reg_table['r_squared'].astype(float).min():.2f} to "
            f"{reg_table['r_squared'].astype(float).max():.2f}), which is the "
            "share of each stock's variance that hedging the index would remove."
        )

# ============================================================
# CHAPTER 4 — ERROR TRADE-OFF
# ============================================================

    chapter_nav("beta")

if chapter == "errors":

    chapter_header("errors")

    section_note(
        "A fast, reactive rule flags a day as stressed if its return falls among "
        "the worst X% of returns seen up to that point. This chapter scores that "
        "one-day rule against the slower high-volatility regime and shows the "
        "trade-off between false alarms and missed stress. Both sides are "
        "ex-ante: the flag threshold uses only the trailing window, and the "
        "regime label is lagged, so this is a detection question rather than a "
        "restatement of the labels."
    )

    # Always the ex-ante regime here, independent of the Chapter 2 toggle.
    regime_ex_ante, _ = ru.classify_regime(
        market_returns,
        window=regime_window,
        method="median",
        ex_ante=True,
    )

    flagged, threshold = ru.flag_stress_days(
        market_returns,
        percentile=flag_pct,
        window=250,
    )

    actual = (regime_ex_ante.reindex(flagged.index) == "high_vol").fillna(False)

    metrics = ru.confusion_matrix_metrics(actual, flagged)

    cm_df = pd.DataFrame(
        [
            [metrics["TP"], metrics["FP"]],
            [metrics["FN"], metrics["TN"]],
        ],
        index=["Flagged", "Not flagged"],
        columns=["Reference: High-vol", "Reference: Low-vol"],
    )

    c1, c2 = st.columns([1, 1.3])

    with c1:

        st.markdown("##### Fast stress flag vs high-volatility regime")

        st.dataframe(cm_df, use_container_width=True)

        st.caption(
            f"Fast rule: flagging returns below the trailing {flag_pct}th "
            f"percentile of the previous 250 days (average threshold: "
            f"{float(np.mean(threshold)):.4%})."
        )

    with c2:

        m1, m2 = st.columns(2)

        m1.metric(
            "Recall (high-vol captured)",
            f"{metrics['recall_sensitivity']:.1%}",
        )

        m2.metric(
            "Specificity (low-vol not flagged)",
            f"{metrics['specificity']:.1%}",
        )

        m3, m4 = st.columns(2)

        m3.metric(
            "Type I error (false alarms)",
            f"{metrics['type_I_error_rate_FPR']:.1%}",
        )

        m4.metric(
            "Type II error (missed high-vol)",
            f"{metrics['type_II_error_rate_FNR']:.1%}",
        )

        st.metric("Precision (flags that were high-vol)", f"{metrics['precision']:.1%}")

    # --------------------------------------------------------
    # Threshold sweep: the trade-off, not a single point on it
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("##### The whole trade-off curve")

    sweep_rows = []

    for pct in (1, 2, 5, 10, 20):
        f_pct, _ = ru.flag_stress_days(market_returns, percentile=pct, window=250)
        a_pct = (regime_ex_ante.reindex(f_pct.index) == "high_vol").fillna(False)
        m_pct = ru.confusion_matrix_metrics(a_pct, f_pct)

        sweep_rows.append(
            {
                "Threshold percentile": f"worst {pct}%",
                "Recall": f"{m_pct['recall_sensitivity']:.1%}",
                "Precision": f"{m_pct['precision']:.1%}",
                "False-alarm rate": f"{m_pct['type_I_error_rate_FPR']:.1%}",
                "Missed high-vol": f"{m_pct['type_II_error_rate_FNR']:.1%}",
            }
        )

    st.dataframe(
        pd.DataFrame(sweep_rows).set_index("Threshold percentile"),
        use_container_width=True,
    )

    st.caption(
        "A single confusion matrix invites the reader to treat one threshold as "
        "the answer. Sweeping it shows what is actually being chosen: recall and "
        "the false-alarm rate move together, so the threshold encodes a "
        "preference about which error is more expensive, and no percentile "
        "dominates the others."
    )

    takeaway(
        "a single-day return shock is a weak proxy for a sustained volatility "
        "regime — tightening the threshold cuts false alarms but sacrifices "
        "recall roughly proportionally, and precision stays modest at every "
        "setting. The practical reading is that one-day triggers should escalate "
        "for review, not drive automated de-risking on their own."
    )

# ============================================================
# CHAPTER 5 — VAR & EXPECTED SHORTFALL
# ============================================================

    chapter_nav("errors")

if chapter == "estimate":

    chapter_header("estimate")

    section_note(
        "Value-at-Risk answers 'how bad is a bad day?' at a chosen confidence "
        "level. Expected Shortfall answers the question VaR cannot: 'given that "
        "we are past that threshold, how bad is it on average?' Basel FRTB "
        "replaced the 99% VaR capital charge with 97.5% ES for exactly that "
        "reason, and because ES is sub-additive while VaR is not."
    )

    var_estimates = {
        "historical": ru.historical_var(market_returns, alpha),
        "parametric": ru.parametric_var(market_returns, alpha),
        "student_t": ru.student_t_var(market_returns, alpha),
        "ewma": ru.ewma_var(market_returns, alpha),
    }

    if ru.GARCH_AVAILABLE:
        var_estimates["garch"] = ru.garch_var(market_returns, alpha)

    st.markdown(f"##### {var_alpha_pct}% one-day VaR — full sample")

    cols = st.columns(len(var_estimates))

    for col, (method, value) in zip(cols, var_estimates.items()):
        col.metric(METHOD_LABELS[method], f"{value:.4%}")

    if ru.GARCH_AVAILABLE:
        st.caption(
            "EWMA (RiskMetrics, lambda = "
            f"{ru.EWMA_LAMBDA}) and GARCH(1,1) both weight recent squared returns "
            "more heavily than distant ones, so unlike the first three they reflect "
            "today's volatility rather than the average of the whole window. The "
            "difference between them: EWMA's decay is a fixed convention, GARCH's "
            "persistence and reaction parameters are estimated from the data by "
            "maximum likelihood — Chapter 6 tests whether that estimation earns its keep."
        )
    else:
        st.caption(
            "EWMA (RiskMetrics, lambda = "
            f"{ru.EWMA_LAMBDA}) weights recent squared returns more heavily than "
            "distant ones, so unlike the first three it reflects today's "
            "volatility rather than the average of the whole window."
        )
        st.info(
            "GARCH(1,1) is not shown because the `arch` package is not "
            "installed in this environment — run `pip install arch` (or "
            "`pip install -r requirements.txt`) to add it. Everything else "
            "on this page is unaffected."
        )

    max_method = max(var_estimates, key=var_estimates.get)

    st.markdown(
        "At this confidence level, "
        + verdict_pill(
            True,
            f"{METHOD_LABELS[max_method]} is the most conservative "
            f"({var_estimates[max_method]:.4%})",
            "Compare VaR estimates",
        ),
        unsafe_allow_html=True,
    )

    if var_alpha_pct == 95 and var_estimates["student_t"] < var_estimates["parametric"]:
        st.info(
            "Worth pausing on: at 95% the Student-t VaR is **smaller** than the "
            "Normal one, even though the Student-t has visibly fatter tails. "
            "The MLE fits heavy tails by shrinking the scale parameter, and at "
            "1.645 standard deviations the tighter scale still dominates the "
            "fatter tail. The ordering flips further out — heavy tails only pay "
            "off deep in the tail, which is precisely where they matter."
        )

    # --------------------------------------------------------
    # Expected Shortfall
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown(f"##### {es_alpha_pct}% Expected Shortfall — full sample")

    es_table = ru.es_summary(market_returns, alpha=es_alpha)

    es_display = es_table.copy()
    es_display["VaR"] = es_display["VaR"].map(lambda v: f"{v:.4%}")
    es_display["ES"] = es_display["ES"].map(
        lambda v: "N/A" if pd.isna(v) else f"{v:.4%}"
    )
    es_display["ES_to_VaR"] = es_display["ES_to_VaR"].map(
        lambda v: "N/A" if pd.isna(v) else f"{v:.3f}x"
    )
    es_display.columns = ["VaR", "Expected Shortfall", "ES / VaR"]

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.dataframe(es_display, use_container_width=True)

    with c2:
        st.markdown(
            "**Why the ratio matters.** ES / VaR is a direct read on tail "
            "thickness. Under a Normal distribution at 97.5% it is about 1.18 "
            "and cannot be anything else. When the empirical and Student-t "
            "ratios come out materially above that, the extra is the part of "
            "the loss distribution a Normal VaR model structurally cannot see, "
            "no matter how the confidence level is tuned."
        )

    # --------------------------------------------------------
    # Translate to monetary loss
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("##### Translate into an estimated loss")

    exposure = st.number_input(
        "Illustrative total exposure (Rs)",
        min_value=0.0,
        value=4_00_00_000.0,
        step=10_00_000.0,
        format="%.0f",
    )

    money_rows = []

    for label in es_table.index:
        var_value = es_table.loc[label, "VaR"]
        es_value = es_table.loc[label, "ES"]

        money_rows.append(
            {
                "Model": label,
                f"{es_alpha_pct}% VaR loss": (
                    f"Rs {ru.var_to_exposure_loss(var_value, exposure):,.0f}"
                ),
                f"{es_alpha_pct}% ES loss": (
                    "N/A"
                    if pd.isna(es_value)
                    else f"Rs {ru.var_to_exposure_loss(es_value, exposure):,.0f}"
                ),
                "Difference": (
                    "N/A"
                    if pd.isna(es_value)
                    else f"Rs {ru.var_to_exposure_loss(es_value - var_value, exposure):,.0f}"
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(money_rows).set_index("Model"),
        use_container_width=True,
    )

    st.caption(
        "The Difference column is the capital gap between provisioning for the "
        "threshold and provisioning for the average outcome beyond it. On a "
        "single desk it looks like a rounding error; it is the reason FRTB "
        "changed measure."
    )

    hist_ratio = es_table.loc["Historical", "ES_to_VaR"]

    takeaway(
        f"at the {es_alpha_pct}% level the historical ES sits {hist_ratio:.2f}x "
        "its own VaR, against the ~1.18x a Normal distribution permits. Two "
        "practical consequences: the loss beyond the VaR threshold is worse than "
        "a Normal model can express, and choosing between VaR and ES changes "
        "the capital number materially even with the data and confidence level "
        "held fixed."
    )

# ============================================================
# CHAPTER 6 — OUT-OF-SAMPLE VALIDATION
# ============================================================

    chapter_nav("estimate")

if chapter == "validate":

    chapter_header("validate")

    section_note(
        "Each day's VaR is estimated from the preceding rolling window only, "
        "then tested against the next unseen day, and the window advances. "
        "Nothing in the estimate for a given day has seen that day. Three tests "
        "are then applied to the resulting exception sequence: Kupiec asks "
        "whether the NUMBER of breaches is right, Christoffersen's independence "
        "test asks whether their TIMING is random, and the joint test asks for "
        "both at once."
    )

    window = st.slider(
        "Rolling estimation window (trading days)",
        100,
        500,
        250,
        step=10,
    )

    with st.spinner(
        f"Running walk-forward backtest for {len(ru.VAR_METHODS)} models"
        + (
            " (GARCH refits an MLE on every window, so the first run is the "
            "slowest — cached after that)..."
            if ru.GARCH_AVAILABLE
            else "..."
        )
    ):
        backtests = run_backtests(market_returns, window, alpha, ru.EWMA_LAMBDA)
        reports = validation_table(market_returns, window, alpha, ru.EWMA_LAMBDA)

    # --------------------------------------------------------
    # Rolling VaR chart
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=CHART_BG)

    style_chart(ax)

    backtests["historical"]["actual_return"].plot(
        ax=ax,
        color="#9fb8c9",
        alpha=0.60,
        linewidth=0.7,
        label="Realised return",
    )

    for method, bt in backtests.items():
        (-bt["var_estimate"]).plot(
            ax=ax,
            color=METHOD_COLOURS[method],
            linewidth=1.6,
            label=f"Rolling {METHOD_LABELS[method]} VaR",
        )

    ax.axhline(0, color="#cfc4b9", linewidth=0.8)

    ax.set_title(
        f"Walk-forward {var_alpha_pct}% VaR vs realised returns",
        loc="left",
    )

    ax.set_ylabel("Return / VaR")
    ax.set_xlabel("Date")
    ax.legend(ncol=2)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    st.caption(
        "The EWMA line is the visibly reactive one — it widens within days of a "
        "volatility shock and narrows again afterwards, while the three "
        "window-based estimators step slowly and stay wide long after the shock "
        "has passed. That difference in responsiveness is what the independence "
        "test below is designed to price."
    )

    # --------------------------------------------------------
    # The validation battery, all methods and all three tests
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("##### Validation battery")

    st.caption(
        f"One table, three tests, {len(ru.VAR_METHODS)} models. A p-value below 0.05 rejects the "
        "null hypothesis stated in the column header."
    )

    battery = pd.DataFrame(
        {
            "Model": reports["method"],
            "Days": reports["n_obs"],
            "Breaches": reports["violations"],
            "Expected": reports["expected"],
            "Rate": reports["observed_rate"].map(lambda v: f"{v:.2%}"),
            "Kupiec LR": reports["kupiec_LR"].map(lambda v: f"{v:.3f}"),
            "Kupiec p": reports["kupiec_p"].map(format_pvalue),
            "Indep. LR": reports["independence_LR"].map(lambda v: f"{v:.3f}"),
            "Indep. p": reports["independence_p"].map(format_pvalue),
            "Joint LR": reports["cc_LR"].map(lambda v: f"{v:.3f}"),
            "Joint p": reports["cc_p"].map(format_pvalue),
        }
    ).set_index("Model")

    st.dataframe(battery, use_container_width=True)

    st.markdown(
        "**Null hypotheses.** Kupiec: the true breach probability equals "
        f"{alpha:.1%} (chi-square, 1 df). Independence: a breach today is "
        "unrelated to a breach yesterday (chi-square, 1 df). Joint conditional "
        "coverage: both hold simultaneously — the statistic is the sum of the "
        "other two, since they are asymptotically independent (chi-square, 2 df)."
    )

    # --------------------------------------------------------
    # Per-model verdicts, stated in words
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("##### Verdicts")

    def diagnose(row):
        """Name the specific failure mode rather than reporting pass/fail."""

        count_ok = not row["kupiec_reject"]
        timing_ok = not row["clustering_detected"]

        if count_ok and timing_ok:
            return (
                True,
                "Right number of breaches, arriving at random times. Survives "
                "the full battery.",
            )

        if count_ok and not timing_ok:
            return (
                False,
                "Correct breach count but the breaches CLUSTER. This is the "
                "dangerous failure: the model passes a naive coverage check "
                "while concentrating its errors in the periods that matter.",
            )

        if not count_ok and timing_ok:
            direction = (
                "too many" if row["observed_rate"] > alpha else "too few"
            )
            return (
                False,
                f"Breach timing looks random, but there are {direction} of them "
                "— the model is simply miscalibrated at this confidence level.",
            )

        return (
            False,
            "Fails on both count and timing: wrong number of breaches, and they "
            "cluster.",
        )

    for _, row in reports.iterrows():

        passed, explanation = diagnose(row)

        c1, c2 = st.columns([1, 3])

        with c1:
            st.markdown(f"**{row['method']}**")
            st.caption(f"{row['n_obs']} out-of-sample days")
            st.markdown(
                verdict_pill(
                    passed,
                    "Survives all three tests",
                    "Rejected",
                ),
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(explanation)

            st.caption(
                f"P(breach | calm yesterday) = "
                f"{row['p_violation_given_calm']:.2%}  ·  "
                f"P(breach | breach yesterday) = "
                f"{row['p_violation_given_violation']:.2%}  ·  "
                f"expected {alpha:.2%} under independence"
            )

    st.caption(
        "Those two conditional probabilities are the whole intuition behind the "
        "independence test. Under a correctly specified model they should both "
        "sit near the nominal rate; when the second is several times the first, "
        "a breach is a warning that another is coming."
    )

    # --------------------------------------------------------
    # Basel traffic-light zones — the capital consequence
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("##### Basel traffic-light zones — what a regulator does with this")

    st.caption(
        "A p-value is not what a bank acts on — it acts on a capital multiplier. "
        "Basel's Internal Models Approach maps the SAME violation counts above "
        "onto a green / yellow / red zone and a multiplier add-on (base k = 3.00), "
        "officially defined for 99% VaR over a 250-trading-day window."
    )

    zone_rows = []
    for _, row in reports.iterrows():
        zone = ru.basel_traffic_light(int(row["violations"]), int(row["n_obs"]))
        zone_rows.append({
            "Model": row["method"],
            "Violations": zone["n_violations"],
            "N (days)": zone["n_obs"],
            "Zone": zone["zone"],
            "Multiplier (k)": f"{zone['capital_multiplier']:.2f}",
            "Official 250-day table?": "Yes" if zone["is_official_250_day_window"] else "No — scaled",
        })

    zone_df = pd.DataFrame(zone_rows).set_index("Model")

    zc1, zc2 = st.columns([1.4, 1])

    with zc1:
        st.dataframe(zone_df, use_container_width=True)

    with zc2:
        for method_label, row in zone_df.iterrows():
            cls = ZONE_COLOURS[row["Zone"]]
            st.markdown(
                f'<span class="verdict {cls}">{row["Zone"].upper()}</span> '
                f'&nbsp; **{method_label}** — {row["Violations"]} exceptions, '
                f'k = {row["Multiplier (k)"]}',
                unsafe_allow_html=True,
            )

    if not zone_df["Official 250-day table?"].eq("Yes").any():
        st.caption(
            f"This backtest runs over {int(reports['n_obs'].iloc[0])} days, not "
            "the official 250 — zone boundaries above are scaled proportionally "
            "as a documented approximation (`ru.basel_traffic_light`), not the "
            "exact regulatory calibration. Set the window slider near 250 for "
            "the literal Basel table."
        )

    st.caption(
        "Notice this can disagree with the Verdicts above: a model can sit in "
        "the green zone by exception count alone while still being the model "
        "the joint coverage test rejects for clustered timing — the zone counts "
        "only WHEN it isn't right, not both failure modes at once."
    )

    # --------------------------------------------------------
    # Where the breaches actually landed
    # --------------------------------------------------------

    st.markdown("---")
    st.markdown("##### When the breaches happened")

    fig, ax = plt.subplots(figsize=(10, 2.6), facecolor=CHART_BG)

    style_chart(ax)

    for position, (method, bt) in enumerate(backtests.items()):
        breach_dates = bt.index[bt["violation"].astype(bool)]

        ax.scatter(
            breach_dates,
            np.full(len(breach_dates), position),
            s=14,
            color=METHOD_COLOURS[method],
            marker="|",
            linewidths=1.6,
        )

    ax.set_yticks(range(len(backtests)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in backtests], fontsize=8.5)
    ax.set_ylim(-0.6, len(backtests) - 0.4)
    ax.set_title("Out-of-sample VaR breaches over time", loc="left")
    ax.set_xlabel("Date")
    ax.grid(axis="y", visible=False)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    st.caption(
        "Vertical bands of breaches in the same weeks across several models are "
        "clustering made visible. This chart and the independence p-values above "
        "are two views of the same statistic — the test exists because the eye "
        "cannot judge whether a pattern like this is significant."
    )

    with st.expander("Transition counts behind the independence test"):

        st.caption(
            "n_ij counts days where the breach indicator moved from state i to "
            "state j. n11 — a breach immediately following a breach — is the "
            "quantity that detects clustering."
        )

        transition_rows = []

        for method, bt in backtests.items():
            counts = ru.violation_transitions(bt["violation"])
            transition_rows.append({"Model": METHOD_LABELS[method], **counts})

        st.dataframe(
            pd.DataFrame(transition_rows).set_index("Model"),
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Overall conclusion
    # --------------------------------------------------------

    survivors = [
        row["method"]
        for _, row in reports.iterrows()
        if not row["cc_reject"]
    ]

    clustered = [
        row["method"]
        for _, row in reports.iterrows()
        if row["clustering_detected"]
    ]

    if survivors:
        verdict_text = (
            f"of {len(reports)} models, {len(survivors)} survive the joint conditional-"
            f"coverage test at 5%: {', '.join(survivors)}. "
        )
    else:
        verdict_text = (
            "no model survives the joint conditional-coverage test at 5% on this "
            "sample and window. "
        )

    if clustered:
        verdict_text += (
            f"Breach clustering is detected for {', '.join(clustered)}, which "
            "matters more than the headline count: a model can report the right "
            "number of exceptions and still fail here, and the independence "
            "test is the only thing on this page that catches it."
        )
    else:
        verdict_text += (
            "No model shows statistically significant breach clustering on this "
            "sample, so the count tests carry the verdict."
        )

    takeaway(verdict_text)

    st.markdown(
        "**What changes the answer.** Move the window slider and the verdicts "
        "move with it: a short window makes every model reactive but noisy, a "
        "long one makes them stable but slow to widen in a crisis. The window "
        "length is a modelling choice with a testable consequence, which is the "
        "practical point of building the battery rather than quoting one VaR."
    )

    chapter_nav("validate")

if chapter == "verdict":

    chapter_header("verdict")

    widest_tail_model = hero_es["ES_to_VaR"].idxmax()
    widest_tail_ratio = hero_es.loc[widest_tail_model, "ES_to_VaR"]

    st.markdown(
        f"""
Every chapter above removed one assumption a naive risk model makes
silently, and showed what breaks when that assumption is false. Chapter 1
found returns are not Normal; Chapter 2 showed the deeper issue is not a
fixed non-Normal shape but *time-varying* risk, measured correctly only by
guarding against look-ahead bias. Chapter 3 showed part of that risk is
shared, market-wide exposure. Chapter 4 showed a cheap, real-time rule
cannot substitute for a properly estimated, continuously updated risk
measure — which Chapters 5 and 6 then built and tested.

**The central result, stated in this run's actual numbers:** of
{len(hero_reports)} models backtested at a {DEFAULT_WINDOW}-day window and
{var_alpha_pct}% confidence, **{len(hero_survivors)} survive** the joint
conditional-coverage test — {', '.join(hero_survivors) if hero_survivors else 'none of them'}.
{
    f"{', '.join(hero_count_only_failures)} in particular "
    + ("gets" if len(hero_count_only_failures) == 1 else "get")
    + " the violation **count** right and still fail on **timing** — "
    "count-correctness and timing-correctness are different properties, "
    "and a validation report that checks only the first would have signed "
    "off on a model that was wrong for weeks at a stretch."
    if hero_count_only_failures else
    "No model in this run exhibits the 'right count, wrong timing' failure "
    "specifically — revisit Chapter 6 at a longer window or different "
    "confidence level, since that gap is usually where it shows up."
}

**On Expected Shortfall.** VaR is silent on severity once breached; ES is
not. At the {es_alpha_pct}% level tested in Chapter 5, **{widest_tail_model}**
shows the widest gap between the two (ES/VaR ≈ {widest_tail_ratio:.2f}×) —
the concrete, on-this-data version of the argument Basel's FRTB made when
it replaced 99% VaR with 97.5% ES as the regulatory capital measure.

**On the Basel traffic-light translation (Chapter 6).** A statistical
verdict is not what a bank acts on — it acts on a capital multiplier. The
zone table there applies the same violation counts to Basel's official
framework and shows the mechanism explicitly: a model can sit in the green
zone on exception count alone while still being the model rejected above
for clustered timing.

**The one conclusion.** Statistical adequacy under a single test is not
model adequacy. Every failure mode surfaced in this project was found by
asking what the previous chapter's finding implied, not by running a
checklist — which is also the argument for reading a validation battery in
full, rather than stopping at the first p-value that looks acceptable.
        """
    )

    st.markdown("---")
    st.markdown("##### Read the full case")
    st.caption(
        "This page summarises live, session-specific numbers. The full "
        "written argument — including the look-ahead bias demonstration and "
        "every regression test that pins these results — is in `README.md` "
        "and `main.ipynb`."
    )

    if st.button("← Back to the Executive Summary", use_container_width=True):
        _goto_chapter("hero")
# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "All statistics computed in risk_utils.py, the single engine shared with "
    "main.ipynb. Regime labels are lagged one day and every VaR figure in "
    "Chapter 6 is estimated out of sample."
)

