"""Readout for the edge study under addendum 2 (2026-11-13). Changes no frozen file.

    python readout_checks.py arms    [--chain PATH]              # arm continuity only; prints no outcome
    python readout_checks.py readout [--chain PATH] [--look 1|2] [--arm ARM]
    python readout_checks.py selftest

Addendum 1 (2026-11-12): arms are never pooled. The readout runs on ONE arm, by default the arm of
the chain's first decision date; decision dates logged under any other arm are excluded and listed.

Addendum 2 (2026-11-13): the PRIMARY test is the archetype-and-sector-neutral IC. The plan's
original sector-neutral IC is demoted to a reported secondary (primary_analysis.primary_report runs
unchanged and its full output is included). Reason, from decision-time fields only: in the first six
decision dates the model scored one archetype about 20 points above the other and drew 17 of 18
picks from it, and the two archetypes carry structurally different plans, so a pooled IC would
mostly measure which archetype did better in the period.

THE PRIMARY STATISTIC, exactly (per decision date in the analysis set):
  1. Keep candidates with a parsed score. A date with fewer than `min_pool` (8) of them is excluded
     and counted.
  2. Rank the scores, and separately the realized R, across all kept candidates of the day.
     Ties get the AVERAGE of the ranks they span (scipy.stats.rankdata, method="average").
  3. Regress each rank vector by least squares on a dummy for every sector present that day plus a
     dummy for every archetype present except the alphabetically first (no separate intercept), and
     keep the residuals.
  4. Daily IC = sum(a*b) / sqrt(sum(a*a) * sum(b*b)) for the two residual vectors. If either has
     zero variation (below 1e-12) the date is excluded and counted.
  On a date with a single archetype this equals the original sector-neutral IC.
INFERENCE, unchanged from the plan: daily ICs in date order, means of consecutive non-overlapping
20-day blocks, one-sided t-test of mean > 0 at alpha 0.015 (Look 1) / 0.035 (Look 2); NULL when two
one-sided tests rule out |mean IC| >= 0.04 at the same alpha; otherwise INCONCLUSIVE.
Heavy ties in the scores (the model repeats values) lower the power of any rank statistic; nothing
here or in the scoring is changed because of it.

This file is deliberately NOT named study_*.py: study_log.git_commit() marks the commit "+dirty"
for any uncommitted file matching that pattern, and engine_hash() does not read this file.
The chain reader mirrors study_log.to_records_frame; `readout` cross-checks the two when study_log
is importable.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import primary_analysis as PA

ADDENDUM = "addendum 2, 2026-11-13"
DEFAULT_CHAIN = os.path.join("study_log", "chain.jsonl")
FWD_TERMINAL = ("RESOLVED", "DELISTED")
ZERO = 1e-12
COLS = ["decision_date", "candidate_id", "ticker", "archetype", "sector", "llm_score", "picked", "listwise_rank",
        "arm", "resolved", "r", "fwd_resolved", "fwd_ret"]


class MixedArms(RuntimeError):
    pass


# ───────────────────────────────── chain -> frame ─────────────────────────────────
def load_frame(chain_path):
    """One row per scored candidate; resolution fields from its resolution record (none yet =
    unresolved). Same columns and rules as study_log.to_records_frame."""
    cands, res = {}, {}
    with open(chain_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            t = rec.get("type")
            if t == "candidate" and rec.get("scored"):
                cands[rec["candidate_id"]] = rec
            elif t == "resolution":
                res[rec["candidate_id"]] = rec
    rows = []
    for cid, c in cands.items():
        r = res.get(cid)
        rows.append({"decision_date": pd.Timestamp(c["decision_date"]), "candidate_id": cid, "ticker": c["ticker"],
                     "archetype": c["archetype"], "sector": c["sector"],
                     "llm_score": float(c["llm_score"]) if c["llm_score"] is not None else float("nan"),
                     "picked": bool(c["picked"]),
                     "listwise_rank": float(c["listwise_rank"]) if c["listwise_rank"] is not None else float("nan"),
                     "arm": c["arm"], "resolved": r is not None, "r": r["realized_r"] if r else float("nan"),
                     "fwd_resolved": bool(r) and r["fwd_status"] in FWD_TERMINAL,
                     "fwd_ret": r["fwd_ret"] if r and r["fwd_ret"] is not None else float("nan")})
    df = pd.DataFrame(rows, columns=COLS)
    return df.sort_values(["decision_date", "candidate_id"]).reset_index(drop=True)


# ─────────────────────────────── arms (addendum 1) ───────────────────────────────
def arm_table(df):
    """One row per arm in order of first appearance: first/last decision date, days, candidates."""
    if df.empty:
        return pd.DataFrame(columns=["arm", "first_date", "last_date", "decision_dates", "candidates"])
    g = df.groupby("arm", sort=False)
    out = pd.DataFrame({"first_date": g["decision_date"].min(), "last_date": g["decision_date"].max(),
                        "decision_dates": g["decision_date"].nunique(), "candidates": g.size()})
    return out.sort_values("first_date").reset_index()


def select_arm(df, arm=None):
    """(rows of the one arm to analyse, arm id, arm table). Default: the arm of the first decision
    date. Raises MixedArms if any decision date carries two arms, or if `arm` is unknown."""
    tab = arm_table(df)
    per_day = df.groupby("decision_date")["arm"].nunique()
    if (per_day > 1).any():
        raise MixedArms(f"{int((per_day > 1).sum())} decision date(s) carry more than one arm; the chain needs review")
    if tab.empty:
        return df, None, tab
    chosen = arm if arm is not None else tab["arm"].iloc[0]
    if chosen not in set(tab["arm"]):
        raise MixedArms(f"arm {chosen!r} is not in the chain; arms: {list(tab['arm'])}")
    return df[df["arm"] == chosen].reset_index(drop=True), chosen, tab


# ─────────────────────── the primary statistic (addendum 2) ───────────────────────
def _design(sector, archetype):
    cols = [(sector == k).astype(float) for k in np.unique(sector)]
    cols += [(archetype == a).astype(float) for a in np.unique(archetype)[1:]]   # alphabetically first = baseline
    return np.column_stack(cols)


def _residual_ranks(x, X):
    r = stats.rankdata(x, method="average")
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    return r - X @ beta


def archetype_neutral_ic(scores, r, sector, archetype):
    X = _design(sector, archetype)
    a, b = _residual_ranks(scores, X), _residual_ranks(r, X)
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > ZERO else np.nan


def primary_daily(data, cfg=PA.PREREG):
    """Per decision date: the primary IC, and the sector-neutral IC within each archetype."""
    rows = []
    for date, g in data.groupby("decision_date", sort=True):
        g = g[g["llm_score"].notna()]
        row = {"decision_date": date, "n_scored": len(g)}
        if len(g) >= cfg.min_pool:
            sc, rr = g["llm_score"].to_numpy(), g["r"].to_numpy()
            sec, arch = g["sector"].to_numpy(), g["archetype"].to_numpy()
            row["ic"] = archetype_neutral_ic(sc, rr, sec, arch)
            for a in np.unique(arch):
                m = arch == a
                if m.sum() >= cfg.min_pool:
                    row[f"ic_within_{a}"] = PA._sector_neutral_ic(sc[m], rr[m], sec[m])
        rows.append(row)
    return pd.DataFrame(rows).set_index("decision_date")


def permutation_check(data, observed_mean, cfg=PA.PREREG):
    """Shuffle scores within each sector-and-archetype cell of each day; how often does a random
    within-cell ordering reach the observed mean daily IC? A robustness figure, not the verdict."""
    rng = np.random.default_rng(cfg.seed)
    days = []
    for _, g in data.groupby("decision_date", sort=True):
        g = g[g["llm_score"].notna()]
        if len(g) < cfg.min_pool:
            continue
        sec, arch = g["sector"].to_numpy().astype(str), g["archetype"].to_numpy().astype(str)
        X = _design(sec, arch)
        M = np.eye(len(g)) - X @ np.linalg.pinv(X)
        rs = stats.rankdata(g["llm_score"].to_numpy(), method="average")
        b = M @ stats.rankdata(g["r"].to_numpy(), method="average")
        a0 = M @ rs
        if (a0 @ a0) * (b @ b) <= ZERO ** 2:
            continue
        cell = np.unique(np.char.add(np.char.add(sec, "|"), arch), return_inverse=True)[1]
        days.append((M, rs, b, b @ b, cell, np.argsort(cell, kind="stable")))
    null = np.empty(cfg.n_permutations)
    for i in range(cfg.n_permutations):
        ics = np.full(len(days), np.nan)
        for j, (M, rs, b, nb, cell, base) in enumerate(days):
            order = np.lexsort((rng.random(len(rs)), cell))          # grouped by cell, shuffled within
            p = np.empty_like(rs)
            p[base] = rs[order]
            a = M @ p
            d = np.sqrt((a @ a) * nb)
            if d > ZERO:
                ics[j] = a @ b / d
        null[i] = np.nanmean(ics) if len(days) else np.nan
    return {"p_permutation": float((1 + np.sum(null >= observed_mean)) / (1 + cfg.n_permutations)),
            "null_p95": float(np.nanquantile(null, 0.95)) if len(days) else None}


def _descriptive_block(x, cfg, alpha):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 * cfg.block_days:
        return {"n_days": int(len(x)), "mean": float(x.mean()) if len(x) else None,
                "note": f"fewer than {2 * cfg.block_days} usable days; no test run"}
    out = PA.block_mean_test(x, cfg.block_days, alpha)
    out["block_means"] = [round(float(v), 4) for v in out["block_means"]]
    out["n_days"] = int(len(x))
    return out


def primary_report(data, alpha, cfg=PA.PREREG):
    daily = primary_daily(data, cfg)
    ic_col = daily["ic"] if "ic" in daily else pd.Series(dtype=float)
    ic = ic_col.dropna().to_numpy()
    if len(ic) < 2 * cfg.block_days:
        raise PA.ReadoutNotReady(f"only {len(ic)} decision dates carry a primary IC; need at least {2 * cfg.block_days}")
    test = PA.block_mean_test(ic, cfg.block_days, alpha)
    equivalence = PA.tost_equivalence(test["block_means"], cfg.sesoi_ic, alpha)
    verdict = ("EDGE (primary significant; read the robustness figures)" if test["p_one_sided"] < alpha
               else "NULL: edges >= SESOI ruled out" if equivalence["equivalent_to_zero"]
               else "INCONCLUSIVE: not significant, but effects >= SESOI not ruled out")
    scored = data[data["llm_score"].notna()]
    tilt = {a: {"candidates": int(len(g)), "mean_score": round(float(g["llm_score"].mean()), 2),
                "mean_r": round(float(g["r"].mean()), 4), "picked_share": round(float(g["picked"].mean()), 4)}
            for a, g in scored.groupby("archetype")}
    test["block_means"] = [round(float(v), 4) for v in test["block_means"]]
    return {"statistic": "archetype-and-sector-neutral rank IC (average ranks for ties)", "alpha": alpha,
            "decision_dates": int(len(daily)), "days_below_min_pool": int((daily["n_scored"] < cfg.min_pool).sum()),
            "days_without_variation": int(((daily["n_scored"] >= cfg.min_pool) & ic_col.reindex(daily.index).isna()).sum()),
            "test": test, "equivalence": equivalence, "verdict": verdict,
            "robustness": {"permutation": permutation_check(data, test["mean"], cfg),
                           "leave_one_month_out": PA.leave_one_month_out(daily, "ic")},
            "descriptive": {"within_archetype_IC": {c.replace("ic_within_", ""): _descriptive_block(daily[c].to_numpy(), cfg, alpha)
                                                    for c in daily.columns if c.startswith("ic_within_")},
                            "tilt_by_archetype": tilt}}


def run_readout(df, look=1, arm=None, cfg=PA.PREREG):
    one, chosen, tab = select_arm(df, arm)
    n_days, alpha = {1: (cfg.look1_days, cfg.alpha_look1), 2: (cfg.look2_days, cfg.alpha_look2)}[look]
    original = PA.primary_report(one, look=look, cfg=cfg)          # raises ReadoutNotReady until complete
    primary = primary_report(PA.analysis_set(one, n_days), alpha, cfg)
    excluded = tab[tab["arm"] != chosen]
    same = primary["verdict"].split(":")[0].split(" ")[0] == original["verdict"].split(":")[0].split(" ")[0]
    return {"rules": ADDENDUM, "look": look, "arm_analysed": chosen,
            "arm_is_registered_first_arm": bool(len(tab) and chosen == tab["arm"].iloc[0]),
            "arms_excluded": [{"arm": r.arm, "decision_dates": int(r.decision_dates), "first_date": str(r.first_date.date()),
                               "last_date": str(r.last_date.date())} for r in excluded.itertuples()],
            "verdict": primary["verdict"], "primary": primary,
            "secondary_original_analysis": original,
            "original_primary_agrees": bool(same),
            "note": None if same else "The plan's original sector-neutral IC gives a different verdict "
                    f"({original['verdict']}). Under addendum 2 the verdict above stands; report both."}


# ──────────────────────────────────── self-test ────────────────────────────────────
def _synthetic(days=280, pool=30, within_skill=0.0, score_tilt=0.0, arch_r=0.0, seed=7):
    """Scored, resolved candidates. score_tilt: how strongly the model prefers MOMENTUM. arch_r: how
    much better (worse if negative) momentum outcomes are. within_skill: real within-group skill."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-11-05", periods=days)
    sectors = np.array([f"S{i}" for i in range(6)])
    rows = []
    for d in dates:
        arch = rng.choice(["MOMENTUM_STRONG_BUY", "CONTRARIAN_SUPPORT"], pool)
        sec = rng.choice(sectors, pool)
        mom = (arch == "MOMENTUM_STRONG_BUY").astype(float)
        skill = rng.normal(size=pool)
        score = np.clip(np.round((50 + 12 * skill + 20 * score_tilt * mom + rng.normal(0, 6, pool)) / 3) * 3, 0, 100)  # tied scores
        r = within_skill * skill + arch_r * mom + rng.normal(0, 1, pool)
        top = set(np.argsort(-score)[:3])
        for i in range(pool):
            rows.append({"decision_date": d, "candidate_id": f"{d.date()}|T{i}|{arch[i]}", "ticker": f"T{i}",
                         "archetype": arch[i], "sector": sec[i], "llm_score": float(score[i]), "picked": i in top,
                         "listwise_rank": float("nan"), "arm": "m@abc", "resolved": True, "r": float(r[i]),
                         "fwd_resolved": True, "fwd_ret": float(r[i] / 20)})
    return pd.DataFrame(rows, columns=COLS)


def selftest():
    n = 0

    def check(cond, what):
        nonlocal n
        assert cond, "FAILED: " + what
        n += 1

    rng = np.random.default_rng(1)
    sc, r = np.round(rng.normal(size=20), 1), rng.normal(size=20)       # rounded: ties present
    sec, one = rng.choice(["a", "b", "c"], 20), np.array(["M"] * 20)
    check(abs(archetype_neutral_ic(sc, r, sec, one) - PA._sector_neutral_ic(sc, r, sec)) < 1e-9,
          "with a single archetype the primary equals the original sector-neutral IC, ties included")
    check(np.isnan(archetype_neutral_ic(np.ones(12), r[:12], sec[:12], one[:12])), "all scores tied: the day is excluded")
    # 1. the false positive: model prefers momentum, momentum did better, no selection skill
    out = run_readout(_synthetic(score_tilt=1.0, arch_r=0.6))
    check(out["secondary_original_analysis"]["verdict"].startswith("EDGE") and not out["verdict"].startswith("EDGE"),
          f"tilt with the wind: original says EDGE, primary says {out['verdict'].split(':')[0]}")
    check(out["original_primary_agrees"] is False and out["note"], "the disagreement is reported")
    # 2. the false negative: model prefers momentum, momentum did WORSE, real skill inside each archetype
    out = run_readout(_synthetic(score_tilt=1.0, arch_r=-0.6, within_skill=0.25))
    check(not out["secondary_original_analysis"]["verdict"].startswith("EDGE") and out["verdict"].startswith("EDGE"),
          "tilt against the wind hides real skill from the original test; the primary finds it")
    check(out["primary"]["robustness"]["permutation"]["p_permutation"] < 0.01, "the permutation figure agrees")
    # 3. plain cases
    check(run_readout(_synthetic(within_skill=0.25))["verdict"].startswith("EDGE"), "real skill, no tilt: EDGE")
    out = run_readout(_synthetic(seed=11))
    check(not out["verdict"].startswith("EDGE"), f"no skill, no tilt: {out['verdict'].split(':')[0]}")
    check(0.02 < out["primary"]["robustness"]["permutation"]["p_permutation"] <= 1.0, "permutation figure is unremarkable with no skill")
    # 4. arms are never pooled
    df = _synthetic(days=300)
    df.loc[df["decision_date"] >= df["decision_date"].unique()[150], "arm"] = "m2@def"
    try:
        run_readout(df)
        check(False, "a 150-day first arm must not be ready")
    except PA.ReadoutNotReady as e:
        check("150" in str(e), "first arm has only 150 days: the readout waits instead of pooling the second arm")
    check(list(arm_table(df)["decision_dates"]) == [150, 150], "arm table lists both arms")
    df.loc[df["decision_date"] == df["decision_date"].unique()[10], "arm"] = ["m@abc", "m2@def"] * 15
    try:
        select_arm(df)
        check(False, "a date with two arms must be refused")
    except MixedArms:
        check(True, "a decision date carrying two arms is refused")
    print(f"OK: {n} checks passed")


# ────────────────────────────────────── CLI ──────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="Edge study readout under addenda 1 and 2.")
    ap.add_argument("cmd", choices=("arms", "readout", "selftest"))
    ap.add_argument("--chain", default=DEFAULT_CHAIN)
    ap.add_argument("--look", type=int, default=1, choices=(1, 2))
    ap.add_argument("--arm")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    df = load_frame(a.chain)
    if a.cmd == "arms":            # outcome-blind: dates and counts only
        tab = arm_table(df)
        print(tab.to_string(index=False) if len(tab) else "no scored candidates logged yet")
        print("OK: one arm" if len(tab) <= 1 else
              f"WARNING: {len(tab)} arms. The readout uses the first; later arms are excluded, never pooled.")
        return
    try:
        import study_log
        ref = study_log.to_records_frame(a.chain)
        if not ref.equals(df):
            sys.exit("this file's chain reader disagrees with study_log.to_records_frame; stop and reconcile")
    except ImportError:
        print("note: study_log not importable here; using this file's own chain reader", file=sys.stderr)
    try:
        print(json.dumps(run_readout(df, look=a.look, arm=a.arm), indent=2, default=str))
    except PA.ReadoutNotReady as e:
        sys.exit(f"not ready: {e}")
    except MixedArms as e:
        sys.exit(f"arms: {e}")


if __name__ == "__main__":
    main()
