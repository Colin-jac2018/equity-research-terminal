"""Pre-registered historical test of the two code setups (SETUP_BACKTEST_PREREG.md). No model calls.

    python setup_backtest.py run        # the one registered run (about an hour; resumes if interrupted)
    python setup_backtest.py status     # what has been run so far; prints no result
    python setup_backtest.py selftest   # synthetic prices through the frozen code; no network

Question: does entering on the first day a setup turns true, with the study's frozen levels and
grader, beat entering the same stocks on random days of the same year with the same levels and
grader? Everything that defines a trade is imported from the frozen study files and is not
re-implemented here: archetypes.ARCHETYPES (the two setups), study_pool.generate_pools (events and
cooldown), study_data.compute_levels (entry, stop, target), study_pool.resolve_plan (the hardened
plan-aware grader with the 20-session time stop), and study_config (universe screen).

The run refuses to start unless study_log.engine_hash() matches the forward arm's, so the code
under test is byte-for-byte the code the forward study runs. It reads prices only; it never opens
the study chain and writes only under setup_backtest_results/.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "setup_backtest_results")

# ───────────────────── pre-registered constants (SETUP_BACKTEST_PREREG.md §2-§5) ─────────────────────
PREREG = {
    "first_decision": "2011-01-03",      # first decision date counted
    "last_decision": "2025-11-21",       # last one: its plan and time stop finish inside 2025
    "bars_start": "2009-06-01",          # history for the 252-session indicators and the warm-up
    "bars_end": "2026-01-01",
    "warmup_sessions": 60,               # cooldown history walked before the first counted date (study: DRYRUN_WARMUP_DAYS)
    "grade_window_sessions": 35,         # bars after the decision date handed to the grader (1 entry + 20 time stop + 5 delist gap + margin)
    "placebo_per_event": 3,              # random same-ticker, same-year entry dates per event
    "placebo_salt": "setup-backtest-2026-11-13",
    "cost_bps": 20.0,                    # round trip, on entry notional; the pass line uses this
    "cost_bps_sensitivity": [10.0, 40.0],
    "alpha": 0.025,                      # one-sided, per archetype (0.05 split over two archetypes)
    "n_years": 15,
    "min_positive_years": 10,
    "min_events_per_year": 30,           # an archetype-year with fewer events is not counted as positive
    "forward_arm_engine_hash16": "50ecb1b183736c75",
}


def frozen():
    """The frozen study modules (imported, never edited)."""
    sys.path.insert(0, HERE)
    import archetypes as A
    import study_config as C
    import study_data as D
    import study_pool as P
    return A, C, D, P


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ───────────────────────────────────────── prices ─────────────────────────────────────────
def download_chunk(D, tickers, start, end):
    """study_data.load_bars' processing, with a start/end instead of a period."""
    import yfinance as yf
    raw = yf.download([D.yf_symbol(t) for t in tickers], start=start, end=end, auto_adjust=False, actions=True,
                      group_by="ticker", threads=True, progress=False)
    out = {}
    for t in tickers:
        try:
            sub = raw[D.yf_symbol(t)]
            df = sub[["Open", "High", "Low", "Close", "Volume"]].dropna()
            for col in ("Dividends", "Stock Splits"):
                df[col] = (sub[col].reindex(df.index).fillna(0.0) if col in sub.columns else 0.0)
        except KeyError:
            continue
        if len(df):
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            out[t] = df
    return out


def split_factor_series(bars):
    """study_data.split_factor for every date at once: the product of split ratios with an ex-date
    strictly after each date."""
    ratio = bars["Stock Splits"].where(bars["Stock Splits"] > 0, 1.0)
    on_or_after = ratio[::-1].cumprod()[::-1]
    return on_or_after.shift(-1).fillna(1.0)


# ─────────────────────────────────── one ticker, start to finish ───────────────────────────────────
def process_ticker(t, bars, spy, sessions, cfg, mods):
    """Events (first day a setup turns true, with the study's cooldown), each graded by the frozen
    grader, plus `placebo_per_event` random same-year entry dates on the same ticker, graded the same
    way. Returns (event rows, placebo rows, counters)."""
    A, C, D, P = mods
    first, last = pd.Timestamp(cfg["first_decision"]), pd.Timestamp(cfg["last_decision"])
    f = D.indicator_frame(bars, spy)
    i0 = max(0, sessions.searchsorted(first) - cfg["warmup_sessions"])
    walk = list(sessions[i0:sessions.searchsorted(last, side="right")])
    pools = P.generate_pools({t: bars}, {t: f}, [t], walk)
    pos = {d: i for i, d in enumerate(sessions)}
    counters = {"unresolved": 0, "no_grade_window": 0}
    cache = {}

    def grade(d, lv):
        key = d
        if key not in cache:
            j = pos.get(d)
            if j is None or j + 1 >= len(sessions):
                cache[key] = None
            else:
                asof = sessions[min(j + cfg["grade_window_sessions"], len(sessions) - 1)]
                out = P.resolve_plan(t, d, lv, bars, asof)
                cache[key] = (out["status"], out.get("realized_r")) if out["status"] in P.TERMINAL else ("OPEN", None)
        return cache[key]

    # dates on which this ticker passes the study's universe screen, as printed that day
    printed_close = f["close"] * split_factor_series(bars).reindex(f.index)
    ok = ((printed_close > C.UNIVERSE_MIN_PRICE) & (f["adv20_usd_m"] * 1e6 > C.UNIVERSE_MIN_ADV_USD) & f["atr14"].notna())
    ok.iloc[0] = False
    ok &= (f.index >= first) & (f.index <= last)
    eligible_by_year = {y: g.index for y, g in ok[ok].groupby(ok[ok].index.year)}

    events, placebo = [], []
    for d in walk:
        if d < first:
            continue
        for (tt, a) in pools[d]["pool"]:
            lv = pools[d]["levels"][(tt, a)]
            g = grade(d, lv)
            if g is None:
                counters["no_grade_window"] += 1
                continue
            if g[1] is None:
                counters["unresolved"] += 1
                continue
            events.append((t, a, d, g[0], float(g[1]), lv["stop_pct"], lv["target_anchor"]))
            pool_dates = eligible_by_year.get(d.year, pd.DatetimeIndex([]))
            pool_dates = pool_dates[pool_dates != d]
            if not len(pool_dates):
                continue
            seed = int(hashlib.sha256(f"{t}|{a}|{d.date()}|{cfg['placebo_salt']}".encode()).hexdigest()[:16], 16)
            k = min(cfg["placebo_per_event"], len(pool_dates))
            for p in pool_dates[np.sort(np.random.default_rng(seed).choice(len(pool_dates), k, replace=False))]:
                plv = D.compute_levels(D.asof_row(f.loc[p], D.split_factor(bars, p)))
                pg = grade(p, plv)
                if pg is None or pg[1] is None:
                    continue
                placebo.append((t, a, d.year, p, pg[0], float(pg[1]), plv["stop_pct"]))
    return events, placebo, counters


# ────────────────────────────────────── aggregation and the pass line ──────────────────────────────────────
def net_r(r, stop_pct, cost_bps):
    """Realized R after a round-trip cost of cost_bps on entry notional: the cost in R is the cost as
    a percent of price divided by the stop distance as a percent of price."""
    return np.asarray(r, dtype=float) - (cost_bps / 100.0) / np.abs(np.asarray(stop_pct, dtype=float))


def yearly_table(ev, pl, archetype, cost_bps):
    e, p = ev[ev["archetype"] == archetype], pl[pl["archetype"] == archetype]
    e = e.assign(net=net_r(e["r"], e["stop_pct"], cost_bps))
    p = p.assign(net=net_r(p["r"], p["stop_pct"], cost_bps))
    tab = pd.DataFrame({"events": e.groupby("year").size(), "setup_net_r": e.groupby("year")["net"].mean(),
                        "placebo_trades": p.groupby("year").size(), "placebo_net_r": p.groupby("year")["net"].mean()})
    tab["difference"] = tab["setup_net_r"] - tab["placebo_net_r"]
    return tab


def verdict(tab, cfg):
    """The registered pass line for one archetype (prereg §5). All three parts must hold."""
    usable = tab[(tab["events"] >= cfg["min_events_per_year"]) & tab["difference"].notna()]
    d = usable["difference"].to_numpy()
    if len(d) < 3:
        return {"pass": False, "reason": "fewer than 3 usable years", "usable_years": int(len(d))}
    t = stats.ttest_1samp(d, 0.0, alternative="greater")
    positive = int((d > 0).sum())
    mean_setup = float(usable["setup_net_r"].mean())
    parts = {"difference_significant": bool(t.pvalue < cfg["alpha"]),
             "positive_years_enough": positive >= cfg["min_positive_years"],
             "setup_net_r_above_zero": mean_setup > 0}
    return {"pass": all(parts.values()), "parts": parts, "usable_years": int(len(d)), "years_registered": cfg["n_years"],
            "mean_difference": float(d.mean()), "t": float(t.statistic), "p_one_sided": float(t.pvalue),
            "positive_years": positive, "mean_setup_net_r": mean_setup, "mean_placebo_net_r": float(usable["placebo_net_r"].mean())}


def report(ev, pl, cfg, archetypes):
    out = {"prereg": cfg, "events": int(len(ev)), "placebo_trades": int(len(pl)), "archetypes": {}}
    for a in archetypes:
        tab = yearly_table(ev, pl, a, cfg["cost_bps"])
        e = ev[ev["archetype"] == a]
        out["archetypes"][a] = {
            "verdict": verdict(tab, cfg),
            "by_year": {int(y): {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in row.items()} for y, row in tab.iterrows()},
            "sensitivity": {f"{c:g}bps": verdict(yearly_table(ev, pl, a, c), cfg) for c in cfg["cost_bps_sensitivity"]},
            "descriptive": {"status_share": {k: round(float(v), 4) for k, v in e["status"].value_counts(normalize=True).items()},
                            "share_positive_r": round(float((e["r"] > 0).mean()), 4) if len(e) else None,
                            "median_r": round(float(e["r"].median()), 4) if len(e) else None,
                            "median_stop_pct": round(float(e["stop_pct"].median()), 2) if len(e) else None,
                            "target_anchor_share": {k: round(float(v), 4) for k, v in e["target_anchor"].value_counts(normalize=True).items()}}}
    return out


def print_report(rep):
    cfg = rep["prereg"]
    print(f"\n{rep['events']:,} setup events and {rep['placebo_trades']:,} placebo trades, decision dates "
          f"{cfg['first_decision']} to {cfg['last_decision']}, net of {cfg['cost_bps']:g} bps round trip")
    for a, r in rep["archetypes"].items():
        print(f"\n{a}")
        print("  year  events  setup R  placebo R  difference")
        for y, row in r["by_year"].items():
            f = lambda v: "   n/a" if v is None else f"{v:+.3f}"
            print(f"  {y}  {int(row['events'] or 0):6d}  {f(row['setup_net_r'])}    {f(row['placebo_net_r'])}     {f(row['difference'])}")
        v = r["verdict"]
        if "parts" not in v:
            print(f"  FAIL: {v['reason']}")
            continue
        print(f"  mean difference {v['mean_difference']:+.3f}R over {v['usable_years']} years, t = {v['t']:.2f}, one-sided p = {v['p_one_sided']:.4f} "
              f"(needs < {cfg['alpha']})")
        print(f"  positive in {v['positive_years']} of {cfg['n_years']} years (needs {cfg['min_positive_years']}); "
              f"setup net R {v['mean_setup_net_r']:+.3f} (needs > 0); placebo net R {v['mean_placebo_net_r']:+.3f}")
        print(f"  {'PASS' if v['pass'] else 'FAIL'} at {cfg['cost_bps']:g} bps   |   " +
              "   ".join(f"{k}: {'pass' if s.get('pass') else 'fail'}" for k, s in r["sensitivity"].items()))
        d = r["descriptive"]
        print(f"  outcomes: {d['status_share']}   positive R {d['share_positive_r']:.0%}   median R {d['median_r']:+.2f}   "
              f"median stop {d['median_stop_pct']}%")
    print("\nReading it (prereg §6): a FAIL closes the question for these parameters. A PASS earns a forward test "
          "and nothing more; the universe is today's index members, so absolute returns are flattered.")


# ─────────────────────────────────────────── the run ───────────────────────────────────────────
def log_line(**kw):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "runs.log"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **kw}) + "\n")


def run(rerun_reason=None):
    mods = A, C, D, P = frozen()
    import study_log as L
    cfg = dict(PREREG)
    eh = L.engine_hash()
    if eh[:16] != cfg["forward_arm_engine_hash16"]:
        sys.exit(f"engine hash {eh[:16]} is not the forward arm's {cfg['forward_arm_engine_hash16']}: the frozen code "
                 "has changed, so this would not be the registered test. Nothing run.")
    assert (C.TIME_STOP_DAYS, C.COOLDOWN_DAYS, C.UNIVERSE_MIN_PRICE, C.UNIVERSE_MIN_ADV_USD) == (20, 10, 5.0, 10_000_000)
    final = os.path.join(RESULTS, "result.json")
    if os.path.exists(final) and not rerun_reason:
        sys.exit("the registered run already has a result (setup_backtest_results/result.json). A second run needs "
                 "--rerun-reason \"...\", which is logged; a rerun with changed code or constants is a new test.")
    me = sha256_file(os.path.abspath(__file__))
    log_line(event="start", script_sha256=me, engine_hash=eh, rerun_reason=rerun_reason)
    print(f"script sha256 {me}\nengine hash   {eh}")
    work = os.path.join(RESULTS, "work")
    os.makedirs(work, exist_ok=True)
    tickers = sorted(D.universe_list())
    spy_path = os.path.join(work, "spy.pkl")
    if not os.path.exists(spy_path):
        pickle.dump(download_chunk(D, ["SPY"], cfg["bars_start"], cfg["bars_end"])["SPY"], open(spy_path, "wb"))
    spy_bars = pickle.load(open(spy_path, "rb"))
    spy, sessions = spy_bars["Close"], spy_bars.index
    chunks = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
    t0 = time.time()
    for n, chunk in enumerate(chunks):
        part = os.path.join(work, f"part_{n:03d}.pkl")
        if os.path.exists(part):
            continue
        bars = download_chunk(D, chunk, cfg["bars_start"], cfg["bars_end"])
        ev, pl, cnt = [], [], {"unresolved": 0, "no_grade_window": 0, "no_bars": len(chunk) - len(bars)}
        for t in chunk:
            if t not in bars or len(bars[t]) < 260:
                continue
            e, p, c = process_ticker(t, bars[t], spy, sessions, cfg, mods)
            ev += e
            pl += p
            for k in c:
                cnt[k] += c[k]
        pickle.dump({"events": ev, "placebo": pl, "counters": cnt}, open(part + ".tmp", "wb"))
        os.replace(part + ".tmp", part)
        print(f"chunk {n + 1}/{len(chunks)}: {len(ev):,} events, {len(pl):,} placebo trades, {time.time() - t0:.0f}s elapsed")
        time.sleep(2)
    ev, pl, cnt = [], [], {}
    for n in range(len(chunks)):
        d = pickle.load(open(os.path.join(work, f"part_{n:03d}.pkl"), "rb"))
        ev += d["events"]
        pl += d["placebo"]
        for k, v in d["counters"].items():
            cnt[k] = cnt.get(k, 0) + v
    ev = pd.DataFrame(ev, columns=["ticker", "archetype", "date", "status", "r", "stop_pct", "target_anchor"])
    ev["year"] = pd.to_datetime(ev["date"]).dt.year
    pl = pd.DataFrame(pl, columns=["ticker", "archetype", "year", "date", "status", "r", "stop_pct"])
    rep = report(ev, pl, cfg, sorted(A.ARCHETYPES))
    rep.update(script_sha256=me, engine_hash=eh, universe_n=len(tickers), counters=cnt, rerun_reason=rerun_reason,
               finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    n_prev = len([x for x in os.listdir(RESULTS) if x.startswith("result")])
    out_path = final if not os.path.exists(final) else os.path.join(RESULTS, f"result_rerun_{n_prev}.json")
    json.dump(rep, open(out_path, "w"), indent=2, default=str)
    ev.to_csv(os.path.join(RESULTS, "events.csv"), index=False)
    log_line(event="end", result_file=os.path.basename(out_path), result_sha256=sha256_file(out_path))
    print_report(rep)
    print(f"\nwrote {out_path}")


def status():
    path = os.path.join(RESULTS, "runs.log")
    if not os.path.exists(path):
        print("no run has been started")
        return
    lines = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    print(f"{sum(1 for x in lines if x['event'] == 'start')} start(s), {sum(1 for x in lines if x['event'] == 'end')} completed")
    for x in lines:
        print(" ", x["at_utc"], x["event"], x.get("rerun_reason") or "", x.get("result_file") or "")


# ───────────────────────────────────────── self-test ─────────────────────────────────────────
def _synthetic_bars(n, start="2009-06-01", end="2014-01-01", seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    out = {}
    for i in range(n + 1):
        close = (30 + 10 * i) * np.exp(np.cumsum(rng.normal(0.0002, 0.02, len(idx))))
        op = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, len(idx)))
        hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.007, len(idx))))
        lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.007, len(idx))))
        df = pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": 2e6,
                           "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)
        out["SPY" if i == n else f"S{i:02d}"] = df
    out["S01"].loc[idx[700], "Stock Splits"] = 2.0          # a 2-for-1 split inside the sample
    out["S01"].loc[idx[900], "Stock Splits"] = 3.0
    return out


def selftest():
    mods = A, C, D, P = frozen()
    n = 0

    def check(cond, what):
        nonlocal n
        assert cond, "FAILED: " + what
        n += 1

    cfg = dict(PREREG, first_decision="2011-01-03", last_decision="2013-11-22", n_years=3, min_positive_years=2,
               min_events_per_year=5)
    bars = _synthetic_bars(8)
    spy, sessions = bars["SPY"]["Close"], bars["SPY"].index
    s = split_factor_series(bars["S01"])
    for d in (sessions[100], sessions[699], sessions[700], sessions[899], sessions[900], sessions[-1]):
        check(abs(s.loc[d] - D.split_factor(bars["S01"], d)) < 1e-12, f"split factor on {d.date()} matches study_data.split_factor")
    ev, pl, cnt = [], [], {}
    for t in [k for k in bars if k != "SPY"]:
        e, p, c = process_ticker(t, bars[t], spy, sessions, cfg, mods)
        ev += e
        pl += p
    ev = pd.DataFrame(ev, columns=["ticker", "archetype", "date", "status", "r", "stop_pct", "target_anchor"])
    ev["year"] = pd.to_datetime(ev["date"]).dt.year
    pl = pd.DataFrame(pl, columns=["ticker", "archetype", "year", "date", "status", "r", "stop_pct"])
    check(set(ev["archetype"]) == set(A.ARCHETYPES) and len(ev) > 100, f"both setups fire ({len(ev)} events)")
    check(set(ev["status"]) <= set(P.TERMINAL) and ev["r"].notna().all(), f"every event ends in a terminal state: {sorted(set(ev['status']))}")
    check(ev["date"].min() >= pd.Timestamp(cfg["first_decision"]) and ev["date"].max() <= pd.Timestamp(cfg["last_decision"]),
          "events stay inside the registered dates")
    check(0 < len(pl) <= cfg["placebo_per_event"] * len(ev) and (pl["year"] == pd.to_datetime(pl["date"]).dt.year).all(),
          "placebo trades are same-year entries, at most 3 per event")
    # the same event regraded directly by the frozen grader gives the same R
    r0 = ev.iloc[len(ev) // 2]
    f = D.indicator_frame(bars[r0.ticker], spy)
    lv = D.compute_levels(D.asof_row(f.loc[r0.date], D.split_factor(bars[r0.ticker], r0.date)))
    direct = P.resolve_plan(r0.ticker, r0.date, lv, bars[r0.ticker], sessions[-1])
    check(direct["status"] == r0.status and abs(direct["realized_r"] - r0.r) < 1e-9,
          "an event's R equals the frozen grader run on the full history (the 35-session window loses nothing)")
    e2, p2, _ = process_ticker("S03", bars["S03"], spy, sessions, cfg, mods)
    e3, p3, _ = process_ticker("S03", bars["S03"], spy, sessions, cfg, mods)
    check(e2 == e3 and p2 == p3, "a rerun reproduces the same events and the same placebo draws")
    check(abs(net_r([1.0], [-8.0], 20.0)[0] - 0.975) < 1e-12, "20 bps on an 8% stop costs 0.025R")
    rep = report(ev, pl, cfg, sorted(A.ARCHETYPES))
    check(all(not r["verdict"]["pass"] for r in rep["archetypes"].values()), "random-walk prices: neither setup passes")
    # the pass line, on constructed tables
    full = dict(PREREG)
    mk = lambda diffs, setup: pd.DataFrame({"events": 500, "setup_net_r": setup, "placebo_trades": 1500,
                                            "placebo_net_r": np.asarray(setup) - np.asarray(diffs), "difference": diffs},
                                           index=range(2011, 2011 + len(diffs)))
    good = [0.06, 0.04, 0.08, 0.05, -0.01, 0.07, 0.03, 0.06, 0.05, -0.02, 0.04, 0.06, 0.05, 0.03, 0.07]
    check(verdict(mk(good, 0.05), full)["pass"], "clear, consistent, profitable: PASS")
    check(not verdict(mk(good, -0.01), full)["pass"], "beats the placebo but loses money after costs: FAIL")
    lumpy = [0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20] + [-0.01] * 6
    v = verdict(mk(lumpy, 0.05), full)
    check(v["parts"]["difference_significant"] and not v["pass"], "significant but positive in only 9 of 15 years: FAIL")
    check(not verdict(mk([0.01, -0.01] * 7 + [0.01], 0.05), full)["pass"], "difference indistinguishable from zero: FAIL")
    thin = mk(good, 0.05)
    thin.loc[thin.index[:6], "events"] = 10
    check(not verdict(thin, full)["pass"], "years with too few events do not count toward the 10")
    print(f"OK: {n} checks passed ({len(ev)} synthetic events, {len(pl)} placebo trades through the frozen code)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pre-registered backtest of the two code setups.")
    ap.add_argument("cmd", choices=("run", "status", "selftest"))
    ap.add_argument("--rerun-reason")
    a = ap.parse_args(argv)
    {"run": lambda: run(a.rerun_reason), "status": status, "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    main()
