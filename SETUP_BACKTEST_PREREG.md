# Setup backtest: pre-registration of 2026-11-13

A historical, code-only test of the two setups that feed the edge study. No model is called. It is
separate from the forward study: it reads prices only, never opens the study chain, and changes no
frozen file. Stamp and commit this file before the first run (section 8).

## 0. State at signing

- SHA-256 of `setup_backtest.py` as committed: `07a90eee4068713258bad905d06e14297a143ab65e49f94a4c3b8e0e3421df61`
- Output of `python setup_backtest.py status` at signing: `no run has been started`
- Engine hash the run must match (the forward arm's): begins `50ecb1b183736c75`

No backtest of these setups, with these or any other parameters, has been run or looked at before
this file was stamped.

## 1. Question

Does entering on the first day a setup turns true, with the study's frozen levels and grader, do
better than entering the same stocks on random days of the same year with the same levels and
grader?

## 2. What is frozen

Everything that defines a trade is imported from the study's files as they stand, pinned by the
engine hash above. The run refuses to start if that hash differs.

| Part | Source |
|---|---|
| The two setups | `archetypes.ARCHETYPES` (MOMENTUM_STRONG_BUY, CONTRARIAN_SUPPORT) |
| Events and cooldown | `study_pool.generate_pools`: first day true, 10-session cooldown, prior plan resolved |
| Entry, stop, target | `study_data.compute_levels`: next open, stop 1.75 ATR below, target at the nearest named high at least 1.5R away, else 1.5R |
| Grader | `study_pool.resolve_plan`: the plan-aware engine, intraday stop, 20-session time stop, dividends credited |
| Universe screen | Close above $5 and 20-day dollar volume above $10 million, as printed on the day |

No parameter is varied. There is one run.

## 3. Data and period

- Universe: `universe_sp1500.csv`, today's S&P 1500 members. Names that were delisted are missing.
  See section 6 for what that does to the reading.
- Decision dates: 2011-01-03 to 2025-11-21, fifteen calendar years. 2026 is excluded because it
  overlaps the product's live sample and the forward study.
- Prices: daily bars from 2009-06-01, processed exactly as `study_data.load_bars` does.
- Sixty sessions of cooldown history are walked before the first counted date.

## 4. Placebo and measure

For each event, three entry dates are drawn at random from the same ticker's sessions in the same
calendar year on which it passes the universe screen. Each gets its levels from the same function
and is graded by the same grader. Draws are seeded from the ticker, setup, date and a fixed salt,
so a rerun reproduces them.

Outcome per trade: realized R net of a 20 bps round-trip cost on entry notional, which in R is
0.20% divided by the stop distance in percent.

For each setup and year: mean net R of the events, mean net R of their placebo trades, and the
difference. The primary measure is the mean of the fifteen yearly differences. Years, not trades,
are the unit of evidence.

## 5. Pass line

A setup passes only if all three hold at 20 bps:

1. The mean yearly difference is above zero at a one-sided p below 0.025 (t-test across years;
   0.05 split over the two setups).
2. The difference is positive in at least 10 of the 15 years. A setup-year with fewer than 30
   events does not count as positive.
3. The mean yearly net R of the setup itself is above zero.

Anything else is a fail. Results at 10 and 40 bps are shown as sensitivity and do not change the
verdict. The by-year table is always reported in full.

## 6. How the result will be read

- Two setups is a small family. A single marginal result is weak evidence even at 0.025.
- Survivorship: the universe is today's members, so absolute returns are flattered. The placebo
  is drawn from the same survivors, so the difference is much less affected than the level. Part 3
  of the pass line is the part survivorship helps most.
- A fail closes the question for these parameters. It is not followed by a variant.
- A pass earns the setup a place as a candidate for a forward test. It is not evidence to trade
  on, and it changes nothing in the product or in the running study.
- The model's ranking is not tested here. A setup that fails can still be part of a forward study
  in which the model's scores carry the edge, and the forward arm continues either way.

## 7. One run, counted

`setup_backtest.py run` writes `setup_backtest_results/result.json` once and logs every start in
`runs.log`. An interrupted run resumes from its saved chunks; that is not a second run. A second
completed run needs a stated reason, which is logged. Any change of setup, level, stop, target,
time stop, universe, period, cost or pass line is a new test with its own pre-registration, and
the number of tests run is kept in this folder's log.

## 8. Stamping

From the repository folder, after filling in section 0:

    python setup_backtest.py selftest
    python setup_backtest.py status
    certutil -hashfile setup_backtest.py SHA256
    python -c "import study_anchor as A; A._ots_stamp('SETUP_BACKTEST_PREREG.md')"
    git add SETUP_BACKTEST_PREREG.md SETUP_BACKTEST_PREREG.md.ots setup_backtest.py
    git commit -m "Setup backtest: pre-registration and script"

Then run it, between two study runs:

    python setup_backtest.py run

It downloads about 1,440 tickers from 2009 and takes roughly an hour to an hour and a half. The
script prints its own SHA-256 at the start; it must equal the one in section 0. Add
`setup_backtest_results/` to `.gitignore` except `result.json` and `runs.log`, which are committed
after the run.
