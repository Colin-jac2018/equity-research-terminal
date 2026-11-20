# Systematic Equity Research Terminal — and a pre-registered experiment to test whether it actually works

> A research tool that turns US-equity setups into fully mechanical trade plans — paired with a pre-registered, Bitcoin-timestamped, outcome-blind study built to answer one honest question: **do the picks have real edge, or am I fooling myself?**

Most trading projects stop at a backtest that flatters the author. This one is built around the opposite instinct: assume the backtest is lying until proven otherwise, and construct the machinery to catch the lie. The headline result so far is deliberately unglamorous — **no edge is proven yet** — and that honesty is the point of the project.

---

## Why I built it this way

I'm interested in event-driven and catalyst investing — the discipline of estimating the probability and payoff of a situation and sizing around it without kidding yourself about the odds. That mindset shaped every design choice here:

- **Expected value over conviction.** Every pick carries a probability and a payoff; positions clear an EV gate, not a gut call. (A real finding below: the model's stated *confidence* was inverted against outcomes — a good reminder that a number you believe is not a number that's true.)
- **Catalyst/event risk is managed explicitly.** A position is skipped or must be hedged if a binary event — an earnings print — falls inside its holding window, because a gap through the stop is the event-risk that actually hurts you. This is the same logic a merger-arb book applies to deal risk, applied to single names.
- **Not fooling yourself is the whole job.** The hard part of investing isn't the idea, it's honest evaluation. So most of the engineering here went into making the evaluation un-gameable rather than into the picking.

---

## How it works

**1. Screen to a defined setup (code, not vibes).** The universe of liquid US names is filtered to one of two code-defined archetypes — a momentum/trend regime or a pullback-to-support regime. A stock either qualifies or it doesn't.

**2. The model narrates; the code computes.** A language model writes the *reasoning* for each idea — the case and the counter-case — but is not permitted to write a single number. Every figure (entry, stop, target, risk/reward, position size) is computed by code. A pick literally cannot contain a made-up number, because the model never touches the math.

**3. Gates drop the junk.** Risk/reward ≥ 1.5, positive expected value, a fillable entry, diversification caps, structural integrity. Thin days show fewer ideas rather than padding with weak ones.

**4. Output is a complete, mechanical plan** — entry, stop (evaluated on the close), laddered targets, and a time stop — fixed at generation and not adjusted afterward. The discipline is in *not* editing it.

---

## The part I'm actually proud of: honest evaluation

Anyone can write a backtest. The work here is in the machinery that keeps it honest.

- **Look-ahead hardening.** The plan-aware P&L engine that grades past picks was audited and fixed across eight separate look-ahead bugs — the easy way to fake a great track record is to accidentally let the backtest see the future, so I hunted those down specifically.
- **A placebo test.** Each pick's exact plan is applied to *random* names from the same candidate list on the same dates. If random names run through the same plan do as well as the real picks, the result came from the plan and the period, not from selection. (They roughly did — more below.)
- **Pre-registration, timestamped into Bitcoin.** The forward study's full analysis plan — hypothesis, universe, test statistic, success threshold, the smallest effect worth caring about — was frozen *before* any outcome existed and anchored with [OpenTimestamps](https://opentimestamps.org), so the commitment date can't be faked or backdated. The analysis is chosen in advance precisely so it can't be tuned after seeing results.
- **Outcome-blind and append-only.** The study scores the candidate pool every trading day into a hash-chained, append-only log. No outcome can be read until the pre-registered readout, so there's nothing to peek at and nothing to quietly rewrite.

This is the difference between "here's a backtest" and "here's a result I can't have cheated on."

---

## Results so far — stated honestly

*Sample: ~119 growth picks over ~3 months (a small, single-regime window). Treat everything here as suggestive at best.*

- **Raw stock selection is roughly a coin flip** versus the market — about −0.26% alpha, a 48% hit rate, not distinguishable from luck.
- **Run through its own plan**, the same picks show +1.16% alpha and a 53% hit rate — but on this sample size that is **not statistically significant** (binomial p ≈ 0.28).
- **The placebo test is the honest tie-breaker.** Random names carrying the same plan did *not* make money, and the real picks beat them by a small margin at roughly 1.5 standard errors — one-sided p ≈ 0.07. That's *suggestive of a weak selection effect, and short of evidence.*
- **The model's confidence rating was inverted** against outcomes, so it's ignored entirely.

**Bottom line: no tradable edge is proven.** What the data shows is a disciplined process that protected capital on a small sample, plus an evaluation framework I trust. The pre-registered forward study — reading out after a full year of outcomes — is the only thing that can turn "suggestive, unproven" into a real answer, and it will report NULL, EDGE, or INCONCLUSIVE honestly either way.

---

## What I learned (the real takeaways)

- **Overfitting is the default, not the exception.** Twenty-plus prompt revisions judged on the same picks will find prompts that fit the sample, not an edge. Guarding against this is a discipline, not a one-time check.
- **A backtest before the model's training cutoff is contaminated** — the model has already read what happened to those stocks. This is *why* a forward, pre-registered test is the only clean one, and why it's necessarily slow.
- **Pre-registration changes your incentives.** Once the analysis is frozen and timestamped, there's no room to move the goalposts — which is the entire point.
- **A disciplined small loss is a better outcome than a win you got by overriding the rules.** The override is the lesson that empties the account later.

---

## What's next

- A **forward, pre-registered test** of the one or two signals that survived a decay-and-orthogonality screen of ~200 published anomalies (the survivors that are both alive post-publication *and* uncorrelated with what the terminal already does).
- The **first-outcome readout** of the main study, which the pre-registration dates to roughly a year out.

---

## Tech

Python · pandas / NumPy · Streamlit · an LLM for reasoning-only narration · OpenTimestamps for Bitcoin anchoring · a hash-chained append-only study log.

---

## Repository layout

- `EDGE_STUDY_PLAN.public.md` — the study's pre-registration, public (redacted) edition.
- `SETUP_BACKTEST_PREREG.md` — pre-registration for the historical, code-only setup backtest.
- `plan_freeze.txt` + `plan_freeze.txt.ots` — the frozen plan's SHA-256 and freeze commit, plus its OpenTimestamps Bitcoin proof.
- `primary_analysis.py`, `readout_checks.py`, `regrade_placebo.py`, `setup_backtest.py`, `study_anchor.py` — the evaluation and analysis code: the readout and sector-neutral IC analysis, the placebo / plan-aware regrade harness, the pre-registered setup backtest, and the OpenTimestamps anchoring helper.

**Intentionally omitted, and released at readout:** the signal engine itself — the exact archetype predicate definitions, the scoring prompt, and the feature/snapshot engineering. This is a deliberate *commit-reveal*: the proof files let you confirm **now** that a complete plan was frozen and Bitcoin-timestamped on 2026-11-05, and at readout the full plan and engine are published — at which point the full plan's SHA-256 will match the hash in `plan_freeze.txt`. It proves the pre-registration is real without handing over a copyable strategy. The `.py` files here are to read, not to run standalone; they import the omitted engine modules by design.

## Verify the timestamp yourself

`plan_freeze.txt` holds the SHA-256 of the frozen plan and the freeze commit. Its `.ots` proof anchors that file to a Bitcoin block. With the [OpenTimestamps](https://opentimestamps.org) client:

```
ots verify plan_freeze.txt.ots
```

It reports the Bitcoin block that establishes the plan existed on its logged date — before any forward data was collected, and impossible to backdate.

## Honest disclaimers

This is a research and learning project, not investment advice. Live trading is a tiny, ring-fenced amount with capped per-trade risk and a hard circuit breaker — the purpose is an honest track record and disciplined practice, not returns. No result here should be read as a claim that the strategy makes money; by design, that question is still open and being tested in the open.
