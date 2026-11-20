# Edge Study — Pre-Registration (public edition)

> **This is a redacted public edition.** The full plan was frozen on 2026-11-05, before any forward data was collected, and its SHA-256 was anchored to the Bitcoin blockchain via OpenTimestamps (see [`plan_freeze.txt`](plan_freeze.txt) and its `.ots` proof). This edition preserves the experimental design, statistics, and governance in full, and **redacts the reproducible specifics** — the exact signal definitions, the scoring prompt, the feature/snapshot engineering, and the operational parameter values. Those are marked `[redacted — in the frozen plan]`. The complete plan will be published at readout; anyone can then confirm its hash matches the proof anchored here.

---

## 0. Purpose & honest priors

**Objective.** Determine whether the research terminal's *selection* carries real, after-cost, factor-adjusted edge. Two questions are kept strictly separate:

1. **Do the mechanical setup archetypes have edge?** Code-only, historical, no LLM — answerable in weeks.
2. **Does the LLM's selection add edge *within* an archetype's eligible pool?** LLM-only, forward, measured entirely after the model's training cutoff so the result can't be memorized.

These are independent. A zero-edge pool can still contain selection skill, and a strong archetype tells you nothing about the LLM.

**Honest priors (written down so a null can't hurt afterward):**
- P(the LLM's selection adds real, after-cost, factor-adjusted edge) ≈ **5–10%**.
- If it exists, it's likely **1–3%/yr of alpha** — economically trivial on a student-sized account.
- **A clean, pre-registered null that *rules out* a meaningful edge is a success of this program**, not a disappointment.
- A NULL means "no edge as large as the pre-registered threshold," never "the model has zero signal."

**Governing principles.**
- Code-testable questions go historical; LLM questions go forward.
- The benchmark is the same-day eligible pool, never the index.
- One primary hypothesis, one primary test, committed as code before any data. Everything else is exploratory and multiplicity-corrected.
- A null is only meaningful as "effects ≥ X ruled out," so X (the smallest effect of interest) and an equivalence test are pre-registered.
- Frozen engine, pinned model, externally timestamped logs.
- Every component that silently assumes an edge is stripped out before measurement.

---

## 1. Pre-registered design (fixed before data)

| Item | Value |
|---|---|
| **Primary hypothesis** | Mean daily **sector-neutral** rank IC between the LLM's score and realized R is > 0. Ranks are taken within the day; sector-day mean ranks are removed, so only *within-sector* ordering counts. |
| **Why sector-neutral** | In simulation, an LLM with persistent sector tastes riding sector trends pushed the raw-IC false-positive rate to ~3× nominal and collapsed power. Sector-neutral IC held near nominal. Sector tilts are factor exposure, not selection skill. |
| **Primary test** | One-sided t-test on the means of non-overlapping 20-trading-day blocks of daily IC. |
| **Why blocks (not Newey–West)** | In simulation, Newey–West (20 lags) gave 7–9% false-positive rates; block means held at the nominal ~5%. |
| **Looks & alpha** | Look 1 on the first 260 decision dates at α = 0.015. Only if Look 1 is INCONCLUSIVE, Look 2 on the first 520 at α = 0.035 (Bonferroni total ≤ 0.05). 260/520 are multiples of the block length, so no days are dropped. |
| **Readout gate** | A decision date enters the analysis only when **all** of its candidates have resolved; the code refuses to run while any is open. Slow days (which hold the long trades) are never skipped — dropping them would be selection bias. |
| **SESOI** (smallest effect of interest) | A pre-registered IC threshold. |
| **Null claim** | TOST equivalence on block means against ±SESOI, at the look's α. |
| **Universe** | A broad, liquidity-filtered US large/mid/small-cap universe, fixed by snapshot. `[exact index, filters, and thresholds redacted — in the frozen plan]` |
| **Archetypes** | Two code-defined setups (a momentum regime and a pullback-to-support regime), frozen as predicates. `[exact predicate definitions redacted — in the frozen plan]` |
| **Model** | A pinned production model, with the scoring **prompt-template SHA-256 recorded in the frozen plan**. Choosing the model was an explicit budget decision, and it must be the model the product ranks with. `[model ID and prompt text redacted — hash in the frozen plan]` |
| **Cadence** | Every US trading day, scored after the close using data as of that close; entries are for the next session. |

**No discovery/confirmation split for the primary test.** With one pre-registered hypothesis there's nothing to "discover," and splitting would halve power. The split applies only to exploratory per-archetype / per-sector findings, which must replicate out-of-sample and are Benjamini–Hochberg corrected.

---

## 2. Freeze & pin

- The plan's SHA-256 is bound to the freeze commit in the OpenTimestamps-anchored `plan_freeze.txt`, and marked by a git tag. **Any change from the freeze starts a new "arm."**
- The analysis code and the archetype code are each pinned by content hash at freeze (hashes recorded in the frozen plan).
- **Successor rule:** if the model is deprecated mid-study, the successor starts a new arm with its own prompt hash; the primary test runs on the first arm alone if it has enough decision dates, otherwise arms are pooled with arm as a stratum and the report says so. Deprecation never silently resets the clock.

---

## 3. Teardown of edge-assuming components

Before measuring, every piece that quietly presumes the strategy works is removed:
- Position sizing switched to fixed-fractional (the primary metric is in R-multiples, which are sizing-invariant, so it doesn't depend on this).
- Conviction/confidence removed from ranking and sizing — the model emits one score per candidate, nothing else.
- Product "gates" are logged and graded as *treatments* (every candidate graded before any gate fires), evaluated only as exploratory.
- Benchmark is the eligible pool, not the index.

---

## 4. Look-ahead hardening (the part most backtests get wrong)

The plan-aware P&L engine that grades every candidate was hardened against look-ahead bias, each fix shipping with a unit test on a synthetic bar that would trip the old behavior:
- Same-bar stop/target ambiguity resolved conservatively (stop assumed first).
- Limit fills require the price to trade *through* the limit, not merely touch it.
- A gap through the stop fills at the open, not the stop.
- Event dates (e.g. earnings) are stored as known on the decision date.
- Options overlays priced on the worse side of the spread, and excluded from the primary metric (R is the stock leg).
- Levels computed and checked on **unadjusted** prices as they printed.
- **Identical mechanics for the pool and the picks** — every candidate's entry/stop/target is computed by the *same* code; the LLM never sets or nudges a level. Otherwise you'd be testing "LLM levels + LLM selection," not selection.

---

## 5. Independent scoring protocol

- Each candidate is scored **independently**, in a fresh context, from a structured as-of-date snapshot — no other candidates visible, which removes position bias and cross-candidate anchoring. `[snapshot feature set and prompt redacted — in the frozen plan]`
- The order of API calls is randomized and logged.
- The snapshot contains only data timestamped on or before the decision close, and its SHA-256 is stored in each record, so anyone can verify what the model saw.
- Raw model output is stored verbatim with the parsed score; parse failures are logged as missing, never silently dropped.
- Output is score-only (a JSON score plus at most one short sentence) — the study needs no narrative, and output tokens are the costly part.

**Construct validity** is addressed explicitly: independent scoring tests whether the model can assign an absolute score that rank-correlates with outcome; a cheap exploratory "listwise" arm separately measures comparative ranking skill. The writeup must name which skill a positive result proves.

---

## 6. Tamper evidence

- The log is append-only and hash-chained — each record embeds the previous record's hash.
- **External anchoring:** a chain held alone could be rewritten end to end, so each day the chain head is timestamped with OpenTimestamps (free, anchors to Bitcoin). The repo stays private until readout; the Bitcoin proofs independently establish that each record existed at its logged time, so contemporaneous publication isn't required for the no-backfill guarantee.
- Outages and missed days are logged as missing. **No backfilling, ever.**

---

## 7. Budget discipline (the most likely way the study dies)

Independent scoring means tens of API calls per trading day, and credit exhaustion had already halted earlier work. So the full cost through the second look was projected from a measured dry run, funded in advance with headroom, guarded by an auto-alert and a live run-rate projection, and governed by a pre-registered halt rule (days lost to credits/outages are logged missing, never backfilled; a long enough outage halts and reports the arm). `[budget figures and call-count estimates redacted — in the frozen plan]`

---

## 8. Readout & decision rules (fixed in advance)

| Verdict | Condition |
|---|---|
| **EDGE** | Primary p < α at the look **and** every robustness check passes (within-sector permutation placebo; leave-one-month-out; factor-residualized IC still positive; survives realistic costs; a credible economic story for who's on the other side). |
| **NULL** | TOST rejects an effect ≥ the SESOI at the look's α. Reported in fixed words: "no selection edge as large as [SESOI]; smaller effects not excluded." |
| **INCONCLUSIVE** | Neither. After Look 1 → go to Look 2. After Look 2 → publish the interval and stop. |

A significant primary that fails any robustness check is reported as "significant but not robust," never as edge.

---

## 9. Risks to the study itself (and the mitigation)

Forking paths (archetypes frozen in code; a run ledger for the historical phase) · peeking (an outcome-blind dashboard that cannot display any outcome metric before readout) · partial resolution (the full-day readout gate, enforced in code) · running out of money (§7) · sector momentum masquerading as skill (sector-neutral primary) · self-tampering (external Bitcoin anchoring, not just a local chain) · pseudo-replication (entry-events + re-entry cooldown; block-based inference) · model deprecation (successor rule) · data integrity (snapshot hashes, unadjusted prices, no backfill).

---

*What this document is for: to show, verifiably and in advance, that the question "does this actually work?" is being answered honestly — pre-registered, timestamped, outcome-blind, and designed so a null is as publishable as a hit. The reproducible specifics are withheld here and released in full at readout, where their hash will match the proof anchored in this repository.*
