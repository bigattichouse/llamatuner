# Roadmap

Improvement ideas, roughly ordered by expected value. Items get checked off (and
their design notes trimmed) as they land.

## 1. Noise-aware picks

`pick_recommendations` and the Pareto frontier treat scores as exact, but with
`--quick` (1 rep) two configs 2% apart are a coin flip.

- llama-bench `-o json` already reports `stddev_ts` per test — capture it into
  the CSV as `pp_std`/`tg_std`; have the server driver keep per-rep samples and
  do the same.
- Report: flag a pick that is statistically tied with its runner-up (within
  ~2σ of the combined noise).
- Tie-breaking: among tied configs prefer more context, then lower measured
  VRAM, instead of whichever got the lucky rep.

This is the item most likely to change a real recommendation. Validate against
a real full-sweep CSV (needs per-rep spread in the data, not synthetic rows).

## 2. Predictive OOM pruning

OOM rows are correctly scored 0, but each still costs a model load + timeout —
at L125 with big models that's 20+ minutes of known-doomed runs.

- `--vram` sampling already exists. Fit a rough VRAM footprint from
  `ngl`/`kv_type`/context on the first few completed rows.
- Skip combinations certain to exceed physical VRAM, recorded as `SKIP_PRED`
  (never silently absent), with an opt-out flag.
- Must be conservative: a wrongly-skipped viable config is worse than a wasted
  OOM run. Needs live-GPU validation.

## 3. Multi-GPU factors

No `-ts` (tensor-split), `-sm` (split-mode layer/row), or `--main-gpu` in the
FACTORS registry — the biggest untuned lever on 2+-card boxes.

- Detect device count via rocm-smi/nvidia-smi; only add the factors when >1
  (same "only sweep where it varies" pattern as `numa`).
- Sensible default levels: `sm=layer,row`; `ts` around the VRAM ratio.
- Needs multi-GPU hardware to validate.

## 4. ~~Results-diff mode~~ — done

`--diff old.csv new.csv` compares two sweeps of the same factor space
(llama.cpp upgrade, driver update, quant swap): per-config tg deltas on the
factor columns both files share, status changes, and whether the old winner
still wins.

## 5. Time-to-first-token metric

The server driver times the whole request; splitting out TTFT would matter for
interactive use where great `tg` with slow prefill still *feels* worse.

- Report alongside `pp`/`tg` (timestamp of first streamed token).
- Not a new objective initially; could later back a `--score ttft`.

## 6. ~~CI for the selftest~~ — done

GitHub Action running the selftest on push/PR, plus a binding smoke test
(builds the submodule, exercises L25/L125 generation and the analyzer — the
paths the selftest deliberately skips).

## Small cleanups

- ~~`--merge-results` rows aren't deduplicated against the current pass~~ —
  done: a merged row is kept only if it beats every known measurement of that
  exact config, so the Pareto/all-runs tables don't repeat rows across passes
  (and the never-lose-an-earlier-best guarantee is preserved).
