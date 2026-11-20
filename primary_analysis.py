"""
Pre-registered primary analysis for the edge study.

Commit this file (with its hash) BEFORE forward data collection starts.
Every constant in PREREG below is part of the pre-registration: changing any of
them after data exists starts a new arm.

Input: one row per candidate in the daily eligible pool (NOT just the picks).
Required columns
    decision_date : date the system ran (data as-of the prior close)
    ticker        : symbol
    archetype     : code-defined archetype label (never an LLM label)
    sector        : point-in-time sector as of decision_date (e.g. GICS sector), from the snapshot
    llm_score     : the LLM's score for this candidate (scored independently, shuffled order)
    picked        : bool, True if in the top-k the system would have shown
    r             : realized R-multiple from the hardened plan-aware engine
                    (unfilled limit entry = 0.0, never dropped)
    fwd_ret       : UNCENSORED forward return, next open -> close `fwd_horizon_days` later
                    (no stops, no targets). Separates selection skill from the lossy plan.
    resolved      : bool, True once the trade plan has fully closed out AND fwd_ret is known.
                    Every plan must resolve: time stop, limit expiry, and a delisting rule
                    (last trade price) guarantee it.
    arm           : model/engine arm identifier (model version + engine hash)
Optional column
    listwise_rank : EXPLORATORY comparative arm: rank from ONE call that sees the whole
                    (shuffled) pool, i.e. the product's comparative task. Higher = better.

Partial resolution: a decision date enters the analysis only when ALL its candidates have
resolved. The analysis set is the first N decision dates in calendar order; the readout
refuses to run until every one of them is complete. Never "skip" a slow day: the slow
days are exactly the ones with long-running trades, so skipping them is selection bias.

Only numpy / pandas / scipy, so it runs anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────── PRE-REGISTERED CONSTANTS ───────────────────────────
@dataclass(frozen=True)
class Prereg:
    look1_days: int = 260             # decision dates in the Look 1 analysis set
    look2_days: int = 520             # Look 2, run only if Look 1 is INCONCLUSIVE
    alpha_look1: float = 0.015        # one-sided; H1: IC > 0.  Bonferroni total 0.05
    alpha_look2: float = 0.035
    fwd_horizon_days: int = 20        # FILL before commit: horizon for the uncensored fwd_ret
    min_pool: int = 8                 # days with fewer scored candidates are skipped from IC (counted)
    block_days: int = 20              # primary inference: t-test on means of non-overlapping 20-day blocks.
                                      # Simulated false-positive rate 5.0% vs 7-9% for Newey-West(20).
    hac_lags: int = 20                # Newey-West reported as a secondary diagnostic only
    sesoi_ic: float = 0.04            # smallest effect size of interest, mean daily rank IC.
                                      # In simulation, IC 0.04 at pool 30 ~= top-3 beating the rest by ~0.10R.
    n_permutations: int = 2000        # within-day score shuffles for the placebo test
    seed: int = 20261023


PREREG = Prereg()


# ─────────────────────────────── core statistics ────────────────────────────────
def newey_west_se(x: np.ndarray, lags: int) -> float:
    """HAC standard error of the mean of x (Bartlett kernel)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    u = x - x.mean()
    gamma0 = u @ u / n
    s = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1 - lag / (lags + 1)
        s += 2 * w * (u[lag:] @ u[:-lag]) / n
    return float(np.sqrt(max(s, 0) / n))


def hac_mean_test(x: np.ndarray, lags: int, alpha: float) -> dict:
    x = np.asarray(x, dtype=float)
    mean = float(x.mean())
    se = newey_west_se(x, lags)
    t = mean / se if se > 0 else np.nan
    df = len(x) - 1
    p_one = float(stats.t.sf(t, df))
    crit = stats.t.ppf(1 - alpha, df)  # one-sided bound at alpha == two-sided CI at 1 - 2*alpha
    return {
        "n_days": len(x),
        "mean": mean,
        "hac_se": se,
        "t": float(t),
        "p_one_sided": p_one,
        "lower_bound_one_sided": mean - crit * se,
        "ci90_two_sided": (mean - crit * se, mean + crit * se),
    }


def block_mean_test(x: np.ndarray, block: int, alpha: float) -> dict:
    """Primary inference. Average consecutive non-overlapping blocks, one-sided t-test on block means.
    Trailing days that don't fill a block are dropped (reported)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n_blocks = len(x) // block
    m = x[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    res = stats.ttest_1samp(m, 0.0, alternative="greater")
    se = m.std(ddof=1) / np.sqrt(n_blocks)
    crit = stats.t.ppf(1 - alpha, n_blocks - 1)
    return {"n_blocks": n_blocks, "days_dropped_tail": len(x) - n_blocks * block,
            "mean": float(m.mean()), "se": float(se), "t": float(res.statistic),
            "p_one_sided": float(res.pvalue), "lower_bound_one_sided": float(m.mean() - crit * se),
            "block_means": m}


def tost_equivalence(block_means: np.ndarray, bound: float, alpha: float) -> dict:
    """Two one-sided tests on block means: can we rule out effects larger than +/- bound?"""
    x = np.asarray(block_means, dtype=float)
    mean, se, df = float(x.mean()), x.std(ddof=1) / np.sqrt(len(x)), len(x) - 1
    p_lower = stats.t.sf((mean + bound) / se, df)   # H0: mean <= -bound
    p_upper = stats.t.cdf((mean - bound) / se, df)  # H0: mean >= +bound
    p = float(max(p_lower, p_upper))
    return {"bound": bound, "p_tost": p, "equivalent_to_zero": p < alpha}


# ─────────────────────────────── daily aggregation ──────────────────────────────
def _spearman_per_day(scores: np.ndarray, r: np.ndarray) -> float:
    if np.ptp(scores) == 0 or np.ptp(r) == 0:
        return np.nan
    return float(stats.spearmanr(scores, r).statistic)


def _sector_demeaned_ranks(x: np.ndarray, sector: np.ndarray) -> np.ndarray:
    a = stats.rankdata(x)
    for k in np.unique(sector):
        m = sector == k
        a[m] -= a[m].mean()
    return a


def _sector_neutral_ic(scores: np.ndarray, r: np.ndarray, sector: np.ndarray) -> float:
    """PRIMARY. Rank within day, remove sector-day mean ranks, correlate residuals.
    Only within-sector ordering counts, so persistent LLM sector tastes riding sector
    trends cannot masquerade as selection skill (simulated: raw IC false-positive rate
    rose ~3x in that scenario; sector-neutral IC stayed near alpha)."""
    a, b = _sector_demeaned_ranks(scores, sector), _sector_demeaned_ranks(r, sector)
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan


class ReadoutNotReady(RuntimeError):
    pass


def analysis_set(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """First n_days decision dates in calendar order, all fully resolved, or raise."""
    dates = np.sort(df["decision_date"].unique())
    if len(dates) < n_days:
        raise ReadoutNotReady(f"only {len(dates)} decision dates logged; need {n_days}")
    chosen = dates[:n_days]
    sub = df[df["decision_date"].isin(chosen)]
    open_days = sub.groupby("decision_date")["resolved"].all()
    if not open_days.all():
        pending = open_days[~open_days].index
        raise ReadoutNotReady(f"{len(pending)} of the first {n_days} decision dates still have open "
                              f"candidates (latest {pd.Timestamp(pending.max()).date()}). Wait.")
    return sub


def daily_table(df: pd.DataFrame, cfg: Prereg = PREREG) -> pd.DataFrame:
    """Expects a fully-resolved analysis set (see analysis_set)."""
    assert df["resolved"].all(), "daily_table needs a fully resolved analysis set"
    rows = []
    for date, g in df.groupby("decision_date", sort=True):
        g = g[g["llm_score"].notna()]          # parse failures: excluded from IC, counted below
        row = {"decision_date": date, "n_scored": len(g), "n_picked": int(g["picked"].sum())}
        if len(g) >= cfg.min_pool:
            sc, sec = g["llm_score"].to_numpy(), g["sector"].to_numpy()
            row["ic"] = _sector_neutral_ic(sc, g["r"].to_numpy(), sec)          # PRIMARY
            row["ic_raw"] = _spearman_per_day(sc, g["r"].to_numpy())             # secondary
            row["ic_fwd_ret"] = _spearman_per_day(sc, g["fwd_ret"].to_numpy())
            if "listwise_rank" in g and g["listwise_rank"].notna().sum() >= cfg.min_pool:
                h = g[g["listwise_rank"].notna()]
                row["ic_listwise"] = _spearman_per_day(h["listwise_rank"].to_numpy(), h["r"].to_numpy())
            picked, rest = g.loc[g["picked"], "r"], g.loc[~g["picked"], "r"]
            row["pick_minus_rest"] = (picked.mean() - rest.mean()) if len(picked) and len(rest) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("decision_date")


# ─────────────────────────────── robustness checks ──────────────────────────────
def placebo_permutation(df: pd.DataFrame, observed_mean_ic: float, cfg: Prereg = PREREG) -> dict:
    """Shuffle LLM scores within each sector-day; how often does random within-sector
    ordering match the observed sector-neutral IC?"""
    rng = np.random.default_rng(cfg.seed)
    d = df[df["llm_score"].notna()]
    days = []
    for _, g in d.groupby("decision_date"):
        if len(g) < cfg.min_pool:
            continue
        sec = g["sector"].to_numpy()
        a = _sector_demeaned_ranks(g["llm_score"].to_numpy(), sec)
        b = _sector_demeaned_ranks(g["r"].to_numpy(), sec)
        norm = np.sqrt((a @ a) * (b @ b))
        if norm > 0:
            # permuting raw scores within a sector == permuting their sector-demeaned ranks
            days.append((a, b, norm, sec, np.argsort(sec, kind="stable")))
    null_means = np.empty(cfg.n_permutations)
    for i in range(cfg.n_permutations):
        ics = np.empty(len(days))
        for j, (a, b, norm, sec, base) in enumerate(days):
            order = np.lexsort((rng.random(len(a)), sec))  # grouped by sector, shuffled within
            ap = np.empty_like(a); ap[base] = a[order]
            ics[j] = ap @ b / norm
        null_means[i] = ics.mean()
    p = float((1 + np.sum(null_means >= observed_mean_ic)) / (1 + cfg.n_permutations))
    return {"p_permutation": p, "null_p95": float(np.quantile(null_means, 0.95))}


def leave_one_month_out(daily: pd.DataFrame, col: str = "ic") -> dict:
    s = daily[col].dropna()
    months = pd.to_datetime(s.index).to_period("M")
    means = {str(m): float(s[months != m].mean()) for m in months.unique()}
    return {"min_mean_dropping_one_month": min(means.values()),
            "worst_month_dropped": min(means, key=means.get),
            "all_positive": all(v > 0 for v in means.values())}


# ─────────────────────────────────── report ─────────────────────────────────────
def primary_report(df: pd.DataFrame, look: int = 1, cfg: Prereg = PREREG) -> dict:
    n_days, alpha = {1: (cfg.look1_days, cfg.alpha_look1), 2: (cfg.look2_days, cfg.alpha_look2)}[look]
    data = analysis_set(df, n_days)
    daily = daily_table(data, cfg)

    def col(name):
        return daily[name].dropna().to_numpy() if name in daily else np.array([])

    ic = col("ic")
    primary = block_mean_test(ic, cfg.block_days, alpha)
    secondary_raw = block_mean_test(col("ic_raw"), cfg.block_days, alpha)
    secondary = block_mean_test(col("pick_minus_rest"), cfg.block_days, alpha)
    secondary_fwd = block_mean_test(col("ic_fwd_ret"), cfg.block_days, alpha)
    exploratory_listwise = (block_mean_test(col("ic_listwise"), cfg.block_days, alpha)
                            if len(col("ic_listwise")) >= 2 * cfg.block_days else None)
    equivalence = tost_equivalence(primary["block_means"], cfg.sesoi_ic, alpha)
    for d in (primary, secondary_raw, secondary, secondary_fwd, exploratory_listwise):
        if d:
            d["block_means"] = [round(v, 4) for v in d["block_means"]]

    verdict = (
        "EDGE (primary significant; confirm robustness checks)" if primary["p_one_sided"] < alpha
        else "NULL: edges >= SESOI ruled out" if equivalence["equivalent_to_zero"]
        else "INCONCLUSIVE: not significant, but effects >= SESOI not ruled out"
    )
    return {
        "prereg": asdict(cfg),
        "look": look, "alpha": alpha,
        "decision_dates": len(daily),
        "days_skipped_small_pool": int(daily["ic"].isna().sum()),
        "score_parse_failures": int(data["llm_score"].isna().sum()),
        "primary_sector_neutral_IC": primary,
        "secondary_raw_IC": secondary_raw,
        "secondary_pick_minus_rest_R": secondary,
        "secondary_IC_uncensored_fwd_ret": secondary_fwd,
        "exploratory_listwise_IC": exploratory_listwise,
        "equivalence_ic": equivalence,
        "diagnostic_newey_west_ic": hac_mean_test(ic, cfg.hac_lags, alpha),
        "placebo": placebo_permutation(data, primary["mean"], cfg),
        "leave_one_month_out_ic": leave_one_month_out(daily, "ic"),
        "verdict": verdict,
    }


if __name__ == "__main__":
    import json, sys
    frame = pd.read_parquet(sys.argv[1]) if sys.argv[1].endswith(".parquet") else pd.read_csv(sys.argv[1], parse_dates=["decision_date"])
    look = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(json.dumps(primary_report(frame, look=look), indent=2, default=str))
