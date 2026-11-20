"""Re-grade the growth research log with the terminal's own engine, apply the live trading rules,
and test the picks against random names given the same plan. Read-only: it never writes to the
research log and never opens the study chain.

    python regrade_placebo.py                      # research_log.json + dashboard.py in this folder
    python regrade_placebo.py --through 2026-10-23 --draws 200 --hold-days 90
    python regrade_placebo.py selftest             # no network: synthetic prices through the real engine

What it answers
  1. HEADLINE. Every logged pick graded by dashboard.grade_logged_pick (the Track Record's own path),
     so the first table should match the Track Record tab. Adds what the tab leaves out: the alpha's
     standard error with picks clustered by report, results net of a trading cost, and R.
  2. YOUR RULES. The same picks under the rules you trade by: skip a pick whose earnings print falls
     inside the hold unless its ticket carries a hedge overlay; take at most 3 picks per report in
     rank order; size each at $100 risk, capped at $2,000. Picks the earnings rule would have skipped
     are reported as their own book, so you can see what the rule cost or saved.
     NOT modelled: the 5-position and $500 open-risk limits, the sector caps, the drawdown step-down.
  3. PLACEBO. For each pick, random names from the same report's candidate list get the SAME plan,
     rescaled to their price (same entry type and distance, same stop and target distances, same
     ladder and trail), and are graded by the same engine over the same dates. If random names with
     your plan do as well as your picks, the result comes from the plan and the period, not from
     selection. Reported as pick minus placebo, with a report-clustered standard error.

Engine functions are pulled out of dashboard.py by name; the Streamlit app itself never runs.
"""
import argparse
import ast
import copy
import json
import math
import os
import pickle
import re
import sys
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RISK_DOLLARS, POSITION_CAP, MAX_PER_REPORT = 100.0, 2000.0, 3     # the journal's Rules sheet
UNDECIDED = ("PENDING", "EXPIRED_NO_FILL", "NO_PLAN", "EXCLUDED", None)


# ─────────────────────────── the terminal's engine, without the app ───────────────────────────
class _St:
    """Stands in for streamlit: cache_data is a pass-through decorator, anything else a no-op."""
    @staticmethod
    def cache_data(fn=None, **kw):
        return fn if callable(fn) else (lambda f: f)

    def __getattr__(self, name):
        return lambda *a, **k: None


class _Missing(types.ModuleType):
    def __getattr__(self, name):
        raise ImportError(f"{self.__name__} is not installed, and this code path needs it")


def load_engine(dashboard_path):
    """dashboard.py's imports, constants and functions in a plain namespace. Top-level UI statements
    are skipped; a statement that cannot run outside the app is skipped and recorded."""
    src = open(dashboard_path, encoding="utf-8").read()
    ns = {"__name__": "dashboard_engine", "__file__": dashboard_path, "st": _St()}
    skipped = []
    keep = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef, ast.Try)
    for node in ast.parse(src).body:
        if not isinstance(node, keep):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound = [(a.asname or a.name.split(".")[0]) for a in node.names]
            if "st" in bound:
                continue
            try:
                exec(compile(ast.Module([node], []), dashboard_path, "exec"), ns)
            except ImportError:
                for b in bound:
                    ns.setdefault(b, _Missing(b))
            continue
        try:
            exec(compile(ast.Module([node], []), dashboard_path, "exec"), ns)
        except Exception as e:
            skipped.append((getattr(node, "lineno", 0), type(e).__name__))
    ns["_skipped"] = skipped
    need = ["grade_logged_pick", "_report_pick_plans", "report_first_session", "plan_aware_realized_alpha",
            "_plan_aware_ticker_bars", "_plan_aware_ticker_actions", "bars_in_asof_basis", "_horizon_to_days",
            "_parse_plan_entry", "_ticket_num", "parse_plan_levels"]
    missing = [n for n in need if n not in ns]
    if missing:
        sys.exit(f"{dashboard_path} does not define: {missing}")
    return ns


# ───────────────────────────────────── small statistics ─────────────────────────────────────
def clustered_mean_se(values, clusters):
    """(mean, standard error, n clusters): observations sharing a cluster label count as one group."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return None, None, 0
    mean = sum(vals) / n
    sums = {}
    for v, c in zip(vals, clusters):
        sums[c] = sums.get(c, 0.0) + (v - mean)
    g = len(sums)
    if g < 2:
        return mean, None, g
    return mean, (sum(s * s for s in sums.values()) / (n * n) * (g / (g - 1))) ** 0.5, g


def fmt(x, unit="%", nd=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}{unit}"


# ──────────────────────────────────────── grading ────────────────────────────────────────
def plan_shares(entry, stop):
    """The journal's sizing: floor($100 / (entry - stop)), capped so the position is <= $2,000."""
    if not entry or not stop or entry <= stop:
        return 0
    return int(min(RISK_DOLLARS // (entry - stop), POSITION_CAP // entry))


def grade_entry_pick(E, entry, ticker, ohlc, spy_tr, plans=None):
    """One row for one logged (or synthetic) pick, from the Track Record's own grading path."""
    g = E["grade_logged_pick"](entry, ticker, False, ohlc, plans=plans)
    out, plan = g.get("out") or {}, g.get("plan") or {}
    ret = out.get("realized_return")
    alpha = E["plan_aware_realized_alpha"](out, spy_tr) if ret is not None else None
    r = out.get("realized_r")
    e, s = plan.get("entry"), plan.get("stop")
    sh = plan_shares(e, s)
    dollars = sh * r * (e - s) if (r is not None and sh) else (0.0 if g.get("status") == "EXPIRED_NO_FILL" else None)
    return {"status": g.get("status"), "ret": ret, "alpha": alpha, "r": r, "entry": e, "stop": s, "shares": sh,
            "notional": sh * e if e else 0.0, "dollars": dollars, "fill_date": out.get("fill_date"),
            "exit_date": out.get("exit_date"), "ticket": g.get("ticket") or {}}


def summarize(rows, cost_bps, label):
    """Headline numbers for a set of graded picks. Return / alpha cover filled picks; R and dollars
    also count an expired, never-filled order as zero, as the engine does."""
    filled = [x for x in rows if x["ret"] is not None]
    with_alpha = [x for x in filled if x["alpha"] is not None]
    with_r = [x for x in rows if x["r"] is not None]
    c = cost_bps / 100.0
    m, se, g = clustered_mean_se([x["alpha"] for x in with_alpha], [x["report"] for x in with_alpha])
    dollars = [x["dollars"] - (x["notional"] * cost_bps / 1e4 if x["ret"] is not None else 0.0)
               for x in rows if x["dollars"] is not None]
    status = {}
    for x in rows:
        status[x["status"]] = status.get(x["status"], 0) + 1
    return {"set": label, "picks": len(rows), "reports": len({x["report"] for x in rows}), "filled": len(filled),
            "avg_return": (sum(x["ret"] for x in filled) / len(filled)) if filled else None,
            "avg_alpha": m, "alpha_se_clustered": se, "alpha_t": (m / se) if (m is not None and se) else None,
            "beat_spy": f"{sum(1 for x in with_alpha if x['alpha'] > 0)}/{len(with_alpha)}",
            "avg_return_net": (sum(x["ret"] for x in filled) / len(filled) - c) if filled else None,
            "avg_alpha_net": (m - c) if m is not None else None,
            "avg_r": (sum(x["r"] for x in with_r) / len(with_r)) if with_r else None,
            "dollars_at_100_risk_net": sum(dollars) if dollars else None, "status": status}


def print_summary(s):
    print(f"\n{s['set']}: {s['picks']} picks in {s['reports']} reports, {s['filled']} filled")
    print(f"  avg return {fmt(s['avg_return'])} (net of cost {fmt(s['avg_return_net'])})   "
          f"avg alpha vs SPY {fmt(s['avg_alpha'])} ± {fmt(s['alpha_se_clustered']).lstrip('+')} report-clustered SE"
          f"{'' if s['alpha_t'] is None else f' (t = {s['alpha_t']:.2f})'}   beat SPY {s['beat_spy']}")
    dollars = s["dollars_at_100_risk_net"]
    print(f"  avg R {fmt(s['avg_r'], 'R')}   P&L at $100 risk / $2,000 cap, net of cost: "
          + ("n/a" if dollars is None else f"${dollars:+,.0f}"))
    print("  status: " + ", ".join(f"{k} {v}" for k, v in sorted(s["status"].items(), key=lambda kv: -kv[1])))


# ─────────────────────────────────────── earnings rule ───────────────────────────────────────
def earnings_dates(ticker, cache, fetch=None):
    """Past and scheduled earnings dates for a ticker (ISO strings), or None when the feed gave none.
    None is 'unverified', never 'no print'."""
    if ticker in cache:
        return cache[ticker]
    try:
        if fetch is not None:
            dates = fetch(ticker)
        else:
            import yfinance as yf
            df = yf.Ticker(ticker).get_earnings_dates(limit=24)
            dates = sorted({pd.Timestamp(d).tz_localize(None).normalize().date().isoformat() for d in df.index}) \
                if df is not None and len(df) else None
    except Exception:
        dates = None
    cache[ticker] = dates
    return dates


def earnings_flag(dates, first, hold_days):
    """'inside' / 'clear' / 'unverified' for a hold of hold_days calendar days from the first session."""
    if not dates:
        return "unverified"
    lo, hi = pd.Timestamp(first), pd.Timestamp(first) + pd.Timedelta(days=hold_days)
    if pd.Timestamp(dates[0]) > lo:
        return "unverified"            # the feed's history does not reach back to this report
    return "inside" if any(lo <= pd.Timestamp(d) <= hi for d in dates) else "clear"


def is_hedged(pick):
    ov = (pick or {}).get("overlay") or {}
    return str(ov.get("type") or "none").strip().lower() not in ("none", "")


# ───────────────────────────────────────── placebo ─────────────────────────────────────────
_LINE_TICKER = re.compile(r"^([A-Z][A-Z0-9]{0,5}(?:[.\-][A-Z0-9]{1,2})?) \(", re.M)


def report_universe(entry, fallback):
    """Tickers on the candidate lines the model was shown (data_block); the fallback list when a
    report logged none."""
    db = entry.get("data_block")
    found = _LINE_TICKER.findall(db) if isinstance(db, str) else []
    return sorted(set(found)) if len(found) >= 20 else list(fallback)


def scale_pick(E, pick, ratio, ticker, resolved_ticket, gen_price):
    """The same plan on another name: every price level multiplied by `ratio`; entry type, trigger
    bases, ladder fractions and trail rule kept. The options overlay is dropped (R is the stock leg)."""
    q = copy.deepcopy(pick)
    q["ticker"] = ticker
    px = lambda v: None if v is None else round(v * ratio, 4)
    q["entry"] = px(E["_parse_plan_entry"](pick.get("entry"), gen_price)[0])
    q["stop"] = px(E["_ticket_num"](pick.get("stop")))
    for key in ("targets", "adds"):
        new = []
        for x in (pick.get(key) or []):
            if isinstance(x, dict):
                y = dict(x)
                y["price"] = px(E["_ticket_num"](x))
            else:
                y = px(E["_ticket_num"](x))
            new.append(y)
        q[key] = new
    tr = q.get("trail")
    if isinstance(tr, dict) and tr.get("rule") == "level" and isinstance(tr.get("value"), (int, float)):
        tr["value"] = px(tr["value"])
    q["overlay"] = None
    for k in ("entry_type", "entry_basis", "stop_basis"):          # as the grader resolved them for the real pick
        if resolved_ticket.get(k) is not None:
            q[k] = resolved_ticket[k]
    ra = resolved_ticket.get("adds") or []
    if len(ra) == len(q.get("adds") or []):
        for a, b in zip(q["adds"], ra):
            if isinstance(a, dict) and b.get("basis"):
                a["basis"] = b["basis"]
    q.pop("stop_as_written", None)
    return q


def asof_close(E, ohlc, ticker, first):
    """The name's last close before the report's first tradeable session, in the report-date share
    basis (the same rebasing grade_logged_pick applies), or None without enough history."""
    bars = E["_plan_aware_ticker_bars"](ohlc, ticker)
    if bars is None:
        return None
    splits, divs = E["_plan_aware_ticker_actions"](ohlc, ticker)
    bars, _ = E["bars_in_asof_basis"](bars, splits, divs, asof=first - pd.Timedelta(days=1))
    before = bars[bars.index.normalize() < first]
    after = bars[bars.index.normalize() >= first]
    if len(before) < 55 or after.empty:                            # the MA50 trail needs 50 sessions of history
        return None
    px = float(before["Close"].iloc[-1])
    return px if px >= 5 else None


def run_placebo(E, picks, entries, ohlc, spy_tr, universes, draws, seed):
    """picks: graded rows of real picks (with 'pick_json', 'gen_price', 'ticket', 'report', 'ticker').
    Returns (per-pick mean placebo metrics, per-draw book means)."""
    rng = np.random.default_rng(seed)
    close_cache, grade_cache = {}, {}
    eligible = {}
    for x in picks:
        rep = x["report"]
        if rep not in eligible:
            first = E["report_first_session"](entries[rep])
            own = set(entries[rep].get("recommended_tickers", []))
            names = []
            for u in universes[rep]:
                if u in own:
                    continue
                key = (rep, u)
                if key not in close_cache:
                    close_cache[key] = asof_close(E, ohlc, u, first)
                if close_cache[key]:
                    names.append(u)
            eligible[rep] = names
    usable = [x for x in picks if x["status"] not in ("NO_PLAN", "EXCLUDED") and x["gen_price"] and eligible[x["report"]]]
    per_pick = {id(x): {"ret": [], "alpha": [], "r": []} for x in usable}
    per_draw = []
    for d in range(draws):
        rets, alphas, rs = [], [], []
        for x in usable:
            rep = x["report"]
            u = eligible[rep][int(rng.integers(len(eligible[rep])))]
            key = (rep, x["ticker"], u)
            if key not in grade_cache:
                pu = close_cache[(rep, u)]
                q = scale_pick(E, x["pick_json"], pu / x["gen_price"], u, x["ticket"], x["gen_price"])
                synth = dict(entries[rep], recommended_tickers=[u], entry_prices={u: pu},
                             ai_response="```json\n" + json.dumps([q]) + "\n```")
                try:
                    grade_cache[key] = grade_entry_pick(E, synth, u, ohlc, spy_tr, plans={u: q})
                except Exception:
                    grade_cache[key] = None
            g = grade_cache[key]
            if g is None:
                continue
            for name, bucket in (("ret", rets), ("alpha", alphas), ("r", rs)):
                if g[name] is not None:
                    bucket.append(g[name])
                    per_pick[id(x)][name].append(g[name])
        per_draw.append({"ret": np.mean(rets) if rets else np.nan, "alpha": np.mean(alphas) if alphas else np.nan,
                         "r": np.mean(rs) if rs else np.nan, "filled": len(rets)})
    return usable, per_pick, per_draw


def placebo_report(usable, per_pick, per_draw):
    out = {"picks_tested": len(usable), "draws": len(per_draw)}
    for name, unit in (("ret", "%"), ("alpha", "%"), ("r", "R")):
        actual = [x[name] for x in usable if x[name] is not None]
        draws = np.array([d[name] for d in per_draw], dtype=float)
        draws = draws[~np.isnan(draws)]
        diffs, reps = [], []
        for x in usable:                                           # paired: this pick minus its own placebo mean
            pl = per_pick[id(x)][name]
            if x[name] is not None and pl:
                diffs.append(x[name] - float(np.mean(pl)))
                reps.append(x["report"])
        m, se, g = clustered_mean_se(diffs, reps)
        out[name] = {"actual_mean": float(np.mean(actual)) if actual else None,
                     "placebo_mean": float(draws.mean()) if len(draws) else None,
                     "placebo_p05_p95": [float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))] if len(draws) else None,
                     "share_of_placebo_books_at_or_above_actual":
                         float((1 + np.sum(draws >= np.mean(actual))) / (1 + len(draws))) if len(draws) and actual else None,
                     "pick_minus_placebo": m, "se_clustered": se, "reports": g, "unit": unit}
    return out


def print_placebo(p):
    print(f"\nPLACEBO: {p['picks_tested']} picks, each replaced {p['draws']} times by a random name from its "
          "report's candidate list carrying the same plan")
    for name, title in (("ret", "return"), ("alpha", "alpha vs SPY"), ("r", "R")):
        v = p[name]
        if v["actual_mean"] is None or v["placebo_mean"] is None:
            print(f"  {title}: not enough data")
            continue
        u = v["unit"]
        lo, hi = v["placebo_p05_p95"]
        print(f"  {title}: your picks {fmt(v['actual_mean'], u)} | random names, same plan {fmt(v['placebo_mean'], u)} "
              f"(5th-95th pct of books {fmt(lo, u)} to {fmt(hi, u)})")
        print(f"      selection effect (pick minus placebo) {fmt(v['pick_minus_placebo'], u)} ± "
              f"{fmt(v['se_clustered'], u).lstrip('+')} report-clustered SE; "
              f"{v['share_of_placebo_books_at_or_above_actual']:.0%} of random books did at least as well")
    print("  Reading it: a selection effect within about two standard errors of zero means the picks are not "
          "distinguishable from random names run through the same plan.")


# ───────────────────────────────────────── pipeline ─────────────────────────────────────────
def spy_total_return(ohlc):
    """SPY closes with dividends folded in (Adj Close), the basis the Track Record's alpha uses."""
    sub = ohlc["SPY"]
    s = (sub["Adj Close"] if "Adj Close" in sub.columns else sub["Close"]).dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None) if getattr(s.index, "tz", None) is not None else s.index
    return s


def run(E, log, ohlc, a, earn_fetch=None, earn_cache=None):
    spy_tr = spy_total_return(ohlc)
    earn_cache = {} if earn_cache is None else earn_cache
    entries = {e["id"]: e for e in log}
    # funds report no earnings: they pass the earnings rule (dashboard.earnings_gap_audit skips them too)
    funds = {t for cat, ts in (E.get("CATEGORIES") or {}).items() if cat in set(E.get("ETF_SECTORS") or ()) for t in ts}
    rows = []
    for e in log:
        plans = E["_report_pick_plans"](e.get("ai_response", ""))
        first = E["report_first_session"](e)
        horizon = E["_horizon_to_days"]((e.get("inputs") or {}).get("time_horizon"))
        hold = horizon if a.hold_days == "auto" else int(a.hold_days)
        for rank, t in enumerate(e.get("recommended_tickers", [])):
            gen = (e.get("entry_prices") or {}).get(t, 0)
            if not gen:
                continue
            row = grade_entry_pick(E, e, t, ohlc, spy_tr, plans=plans)
            pj = plans.get(t) or {}
            flag = ("not_checked" if not a.earnings else "clear" if t in funds
                    else earnings_flag(earnings_dates(t, earn_cache, earn_fetch), first, hold))
            row.update(report=e["id"], date=e["date"], ticker=t, rank=rank, gen_price=gen, pick_json=pj,
                       version=e.get("prompt_version", ""), prob=(e.get("pick_probabilities") or {}).get(t),
                       hold_days=hold, earnings=flag, hedged=is_hedged(pj))
            rows.append(row)
    results = {"headline": summarize(rows, a.cost_bps, "ALL LOGGED PICKS (should match the Track Record tab)")}
    print_summary(results["headline"])
    if E["_skipped"]:
        print(f"  note: {len(E['_skipped'])} top-level statement(s) of dashboard.py were skipped outside the app")
    if a.earnings:
        ok = lambda x: x["earnings"] == "clear" or (x["earnings"] == "inside" and x["hedged"])
        skipped = [x for x in rows if x["earnings"] == "inside" and not x["hedged"]]
        unver = [x for x in rows if x["earnings"] == "unverified"]
        kept, taken = [x for x in rows if ok(x)], {}
        mine = []
        for x in sorted(kept, key=lambda x: (x["date"], x["report"], x["rank"])):
            if taken.get(x["report"], 0) < MAX_PER_REPORT:
                mine.append(x)
                taken[x["report"]] = taken.get(x["report"], 0) + 1
        hold_txt = "the report's own horizon" if a.hold_days == "auto" else f"{a.hold_days} days"
        print(f"\nEARNINGS RULE over a hold of {hold_txt}: {len(kept)} pass, {len(skipped)} skipped "
              f"(print inside the hold, no hedge), {len(unver)} unverified (no earnings history from the feed)")
        for label, sub in (("YOUR RULES (earnings rule, first 3 per report)", mine),
                           ("SKIPPED BY THE EARNINGS RULE (what the rule kept you out of)", skipped),
                           ("UNVERIFIED (earnings dates unavailable; not counted either way)", unver)):
            if sub:
                results[label.split(" (")[0].lower().replace(" ", "_")] = summarize(sub, a.cost_bps, label)
                print_summary(results[label.split(" (")[0].lower().replace(" ", "_")])
    if a.draws > 0:
        fallback = sorted({t for e in log for t in report_universe(e, [])})
        if len(fallback) < 20:
            etf = set(E.get("ETF_SECTORS", ()))
            fallback = sorted({t for cat, ts in E.get("CATEGORIES", {}).items() if cat not in etf for t in ts})
        universes = {e["id"]: [u for u in report_universe(e, fallback) if u in set(a.available)] for e in log}
        usable, per_pick, per_draw = run_placebo(E, rows, entries, ohlc, spy_tr, universes, a.draws, a.seed)
        results["placebo"] = placebo_report(usable, per_pick, per_draw)
        print_placebo(results["placebo"])
    keep = ["report", "date", "version", "rank", "ticker", "status", "ret", "alpha", "r", "entry", "stop", "shares",
            "dollars", "fill_date", "exit_date", "prob", "hold_days", "earnings", "hedged"]
    return results, pd.DataFrame([{k: x.get(k) for k in keep} for x in rows])


def download(E, tickers, start, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"ohlc_{start}_{pd.Timestamp.now().date()}_{len(tickers)}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    print(f"downloading daily bars for {len(tickers)} tickers from {start} (cached for today in {cache_dir})")
    raw = E["_plan_aware_ohlc_batch"](tuple(tickers), start)
    pickle.dump(raw, open(path, "wb"))
    return raw


# ───────────────────────────────────────── self-test ─────────────────────────────────────────
def _synthetic_market(tickers, start="2026-03-02", periods=190, seed=3, drift=None):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    frames = {}
    for i, t in enumerate(tickers):
        mu = (drift or {}).get(t, 0.0003)
        close = (20 + 15 * i) * np.exp(np.cumsum(rng.normal(mu, 0.018, periods)))
        op = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, periods))
        hi, lo = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.006, periods))), np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.006, periods)))
        frames[t] = pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Adj Close": close,
                                  "Volume": 1e6, "Dividends": 0.0, "Stock Splits": 0.0}, index=idx)
    return pd.concat(frames, axis=1)


def selftest(dashboard_path):
    E = load_engine(dashboard_path)
    names = [f"T{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    good = {"TA": 0.004, "TB": 0.004, "TC": 0.004, "TD": 0.004}                    # the "picks" drift up; the rest do not
    ohlc = _synthetic_market(names + ["SPY"], drift=good)
    n = 0

    def check(cond, what):
        nonlocal n
        assert cond, "FAILED: " + what
        n += 1

    log = []
    for k, day in enumerate(("2026-08-03", "2026-08-17")):
        d = pd.Timestamp(day)
        picks, prices = [], {}
        for t in (("TA", "TB") if k == 0 else ("TC", "TD")):
            px = float(ohlc[t]["Close"][ohlc.index <= d].iloc[-1])
            prices[t] = px
            picks.append({"ticker": t, "setup": "MOMENTUM", "probability": 60, "strategy": "tactical", "entry": round(px, 2),
                          "entry_type": "immediate", "stop": round(px * 0.93, 2), "stop_basis": "close",
                          "targets": [{"price": round(px * 1.12, 2), "fraction": 0.5}, {"price": round(px * 1.2, 2), "fraction": 0.3}],
                          "adds": [], "trail": {"rule": "ma50", "value": None},
                          "overlay": {"type": "none"} if t != "TB" else {"type": "collar"}})
        lines = "\n".join(f"{t} (Name {t}) | Sector: Test | Price: $1" for t in names)
        log.append({"id": day.replace("-", "") + "-210000", "date": day, "generated_at": f"{day}T21:00:00-04:00",
                    "inputs": {"time_horizon": "3 Months"}, "recommended_tickers": [p["ticker"] for p in picks],
                    "entry_prices": prices, "pick_probabilities": {p["ticker"]: 60 for p in picks},
                    "data_block": lines, "ai_response": "PICK 1: x\n```json\n" + json.dumps(picks) + "\n```"})
    a = argparse.Namespace(hold_days="auto", earnings=True, cost_bps=10.0, draws=40, seed=1, available=names)
    earn = {"TA": ["2026-05-01", "2026-09-01"], "TB": ["2026-05-01", "2026-09-01"], "TC": ["2026-05-01", "2027-02-01"]}
    res, table = run(E, log, ohlc, a, earn_fetch=lambda t: earn.get(t))
    check(len(table) == 4 and set(table["status"]) <= {"LADDER", "LADDER_OPEN", "OPEN", "STOPPED", "TARGET", "TRAIL"},
          f"four picks graded by the real engine: {list(table['status'])}")
    check(table.loc[table.ticker == "TA", "fill_date"].iloc[0] == pd.Timestamp("2026-08-04"),
          "a report written after the close fills at the next session's open")
    flags = dict(zip(table.ticker, table.earnings))
    check(flags == {"TA": "inside", "TB": "inside", "TC": "clear", "TD": "unverified"}, f"earnings flags: {flags}")
    check(res["your_rules"]["picks"] == 2 and res["skipped_by_the_earnings_rule"]["picks"] == 1,
          "TB passes on its hedge, TA is skipped, TD is unverified and counted in neither")
    p = res["placebo"]
    check(p["picks_tested"] == 4 and p["draws"] == 40, "placebo ran for every pick")
    check(p["ret"]["pick_minus_placebo"] > 5, f"picks built to outperform beat random names with the same plan "
          f"({p['ret']['pick_minus_placebo']:+.1f}%)")
    # the scaled plan really is the same plan
    x = {"pick_json": json.loads(log[0]["ai_response"].split("```json\n")[1].split("\n```")[0])[0]}
    q = scale_pick(E, x["pick_json"], 2.0, "TZ", {"entry_type": "immediate", "stop_basis": "close", "adds": []}, 100.0)
    check(abs(q["stop"] / q["entry"] - x["pick_json"]["stop"] / x["pick_json"]["entry"]) < 1e-4
          and abs(q["targets"][0]["price"] / q["entry"] - 1.12) < 1e-3 and q["stop_basis"] == "close" and q["overlay"] is None,
          "scaling keeps the stop and target distances, the trigger basis and the ladder")
    check(plan_shares(138.38, 127.47) == 9 and plan_shares(50, 49) == 40, "sizing: 9 shares for WPM; the $2,000 cap binds on a tight stop")
    print(f"OK: {n} checks passed (engine loaded from {os.path.basename(dashboard_path)}, "
          f"{len(E['_skipped'])} non-engine statements skipped)")


# ───────────────────────────────────────────── CLI ─────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="Re-grade the growth log, apply the live rules, run the placebo.")
    ap.add_argument("cmd", nargs="?", default="run", choices=("run", "selftest"))
    ap.add_argument("--log", default=os.path.join(HERE, "research_log.json"))
    ap.add_argument("--dashboard", default=os.path.join(HERE, "dashboard.py"))
    ap.add_argument("--through", help="only reports dated on or before this day (YYYY-MM-DD)")
    ap.add_argument("--hold-days", default="auto", help="'auto' = each report's own horizon, or a number of calendar days")
    ap.add_argument("--no-earnings", dest="earnings", action="store_false")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="round-trip trading cost per filled pick")
    ap.add_argument("--draws", type=int, default=200, help="placebo books to draw (0 = skip the placebo)")
    ap.add_argument("--universe-cap", type=int, default=250, help="placebo names downloaded (random subset of the candidate lists)")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--out", default=os.path.join(HERE, "regrade_results"))
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest(a.dashboard)
    E = load_engine(a.dashboard)
    log = [e for e in json.load(open(a.log, encoding="utf-8"))
           if e.get("recommended_tickers") and e.get("type") != "discovery_scan"
           and (not a.through or str(e.get("date")) <= a.through)]
    if not log:
        sys.exit("no reports in range")
    picks = sorted({t for e in log for t in e["recommended_tickers"]})
    pool = sorted({t for e in log for t in report_universe(e, [])} - set(picks))
    if len(pool) < 20:
        etf = set(E.get("ETF_SECTORS", ()))
        pool = sorted({t for cat, ts in E.get("CATEGORIES", {}).items() if cat not in etf for t in ts} - set(picks))
    rng = np.random.default_rng(a.seed)
    extra = sorted(rng.choice(pool, min(a.universe_cap, len(pool)), replace=False).tolist()) if a.draws > 0 and pool else []
    start = (min(pd.Timestamp(str(e["date"])) for e in log) - pd.Timedelta(days=100)).strftime("%Y-%m-%d")
    os.makedirs(a.out, exist_ok=True)
    ohlc = download(E, picks + extra, start, os.path.join(a.out, "cache"))
    a.available = [t for t in extra if E["_plan_aware_ticker_bars"](ohlc, t) is not None]
    ecache_path = os.path.join(a.out, "cache", "earnings_dates.json")
    ecache = json.load(open(ecache_path)) if os.path.exists(ecache_path) else {}
    print(f"{len(log)} reports {log[0]['date']} to {log[-1]['date']}, {len(picks)} picked tickers, "
          f"{len(a.available)} placebo names with bars; cost {a.cost_bps:g} bps round trip")
    results, table = run(E, log, ohlc, a, earn_cache=ecache)
    json.dump(ecache, open(ecache_path, "w"))
    table.to_csv(os.path.join(a.out, "regrade_picks.csv"), index=False)
    json.dump(results, open(os.path.join(a.out, "regrade_summary.json"), "w"), indent=2, default=str)
    print(f"\nwrote {os.path.join(a.out, 'regrade_picks.csv')} and regrade_summary.json")


if __name__ == "__main__":
    main()
