# Design notes & background

Background and prior knowledge behind `llama-optimize`. The [README](../README.md) covers
usage and the tuning methodology; this document records *why* the factors and levels
were chosen and the heuristics we're treating as hypotheses to be confirmed by the
sweep rather than assumed.

## Origin

The goal: given a GGUF model and a `llama.cpp` build, automatically search the
command-line parameter space for the best **tokens/sec at usable context**, and
present the trade-off (fastest vs. longest-context) as ready-to-run commands —
instead of hand-tuning `-ngl`, threads, KV type, etc. by trial and error.

The search uses a Taguchi orthogonal array (via the `robust`/`taguchi` library,
included as a submodule) so we cover the space in ~25 runs instead of thousands.

## Reference hardware

The tool is hardware-agnostic, but it was designed on:

- **AMD Instinct MI50** (32 GB HBM2, gfx906), ROCm + `llama.cpp` (HIP build)
- **128 GB system RAM**
- Large GGUFs run with partial GPU offload — VRAM acts as an accelerator for a
  subset of layers while system RAM holds the rest.

On a box like this, the model rarely fits entirely in VRAM, so `-ngl` (how many
layers to offload) and the KV-cache footprint dominate both speed and the maximum
context that will load. That is exactly what the sweep is built to map.

## llama.cpp tuning heuristics (hypotheses, not assumptions)

These are the starting beliefs that shaped the factor set. The sweep exists to
*measure* them, not take them on faith:

- **Flash attention (`-fa 1`)** reduces KV-cache bandwidth and helps gfx906. It is
  also a precondition for a quantized KV cache in llama.cpp. → We fix it on and
  treat KV-quant as a factor. (A future outer-block run can quantify `-fa 0/1`.)
- **`-ngl` is the biggest lever**, but for MoE models "more layers on GPU" is not
  strictly monotonic — tensors differ. → `-ngl` is the widest-range factor.
- **Don't chase context you don't need.** Large `-c`/depth mostly burns RAM and
  slows everything. → context depth is a *factor* so we can see the cost curve, and
  the "balanced" recommendation is gated by a usable-context floor.
- **Threads ≈ physical cores.** More threads than physical cores usually hurts. →
  thread levels are auto-derived to bracket the physical-core count.
- **mmap on** unless benchmarking locked memory. → fixed `-mmp 1`.
- **KV-cache quantization** (`f16`/`q8_0` … `q4_0`) trades a little quality for
  substantially more context in the same VRAM. → a quality-gated factor: `--min-kv`
  floors it (default `q8_0`, near-lossless) so only `f16, q8_0` are swept unless you
  opt into the lossier ladder with `--min-kv any`.

## Sampling settings (out of scope for the sweep, kept for reference)

Throughput tuning is independent of sampling, but these are the recommended
generation defaults for the models in play and belong in the eventual
`llama-server` invocation:

- General: `--temp 1.0 --top-p 0.95 --min-p 0.01`
- Coding:  `--temp 0.7 --top-p 1.0 --min-p 0.01`

These typically beat llama.cpp's defaults. The tuner optimizes *speed/context*; pair
its output with the sampling settings appropriate to your workload.

## Quantization trade-off (model selection, upstream of tuning)

Choosing the quant is upstream of this tool, but the intuition that motivated a
model-agnostic tuner: smaller quants are faster and fit more context, larger quants
are higher quality. The "sweet spot" (often a mid Q3/Q4/Q5) is exactly the kind of
thing worth benchmarking per-machine rather than guessing — which is what `llama-optimize`
automates for the *parameter* axis once a quant is chosen.

| Quant tier | Speed | Quality |
|------------|-------|---------|
| smaller (Q2–Q3) | faster, more context | good |
| mid (Q4–Q5)     | balanced             | better |
| larger (Q6–Q8/F16) | slower, less context | best |

## Methodology recap

See the README for full detail. The tool evolved from a single Taguchi array into a
**DOE funnel**:

1. **Morris screen** (`--screen`, ~r·(k+1) runs) ranks every knob by importance (μ\*)
   and flags interactions (σ), dropping the ones that don't matter.
2. **Taguchi array** on the survivors — orthogonal, so main effects read cleanly.
3. **Iterative refinement** (`--iterate`) settles low-impact factors and refines the
   high-impact ones onto finer grids, converging on the optimum.
4. **Confirmation run** (`--confirm`) verifies the additive prediction (predicted vs
   actual); a large gap means interactions or thermal drift dominate.

Configs are scored by **effective throughput** for the profile's request shape
`(P+G)/(P/pp+G/tg)`, not raw decode. OOM/crash is recorded as data (the memory cliff);
the recommendation is driven off the measured **Pareto frontier**, with main-effects
used only to *rank* which knobs matter. Two measurement-validity guards proved
necessary in practice: **realistic prompts** (a repeated token inflates MTP
acceptance) and **randomized run order** (GPUs like the MI50 thermally throttle over a
long sweep, which otherwise confounds factor effects). And because some settings can
hard-reboot the machine, attempts are **journaled with fsync** so a crash-causer is
skipped on resume rather than retried into a loop.

## Emitting a command that actually loads

The whole point of the tool is a *ready-to-paste* command, so a recommendation that
OOMs on launch is a correctness bug, not a rough edge. Two defects here both violate
one invariant:

> **The emitted `-c` must never ask for more KV cache than the sweep verified will
> load.** Every recommended row was measured at some depth `d`; the driver that
> measured it sized its context to `n_prompt + d + n_gen + 256` (`server_run_one`),
> times `--parallel`. That footprint — not a prettier, larger number — is the most
> context we have evidence for.

**Defect 1 — context rounded *up* past the verified depth.** The report sized `-c` by
rounding `depth + n_prompt + n_gen` **up to the next power of two**. A row verified at
`depth=49152` (footprint ≈ 50 k tokens) was emitted as `-c 65536` — ~33 % more KV
cache than anything the sweep ran. On a 32 GB card holding a 26 GB model, that ~6 GiB
of headroom is the entire margin, so the f16 KV allocation for the inflated context
overflowed and `llama-server` segfaulted — even though the identical config had just
benchmarked fine at 49 k. Root cause: the recommendation used a *different, larger*
context formula than the driver that verified the row. Fix: emit the verified
footprint itself (mirror `server_run_one`), rounded **down** to a tidy multiple. The
usable-context floor still selects *which* row is "balanced"/"longest", and (next
section) sets how small an emitted `-c` may be — but never past verified evidence.

**Defect 2 — the fast pick hides the cheap context lever.** When `f16` KV wins on
speed (it often does; it is the only type heavier than the `q8_0` floor swept by
default), the "best" command carries `-ctk/-ctv f16`. A user who then wants more
context than the verified depth walks straight back into the OOM, not knowing that
`q8_0` KV is right there: ~half the footprint per token at near-lossless quality
(`KV_QUALITY` ranks it just below `f16`/`bf16`), i.e. roughly double the context in
the same VRAM. Fix: annotate any recommendation whose KV type is heavier than the
`q8_0` floor with a one-line `q8_0` suggestion, so the speed/context trade-off is
visible at the point of copy-paste rather than discovered by a crash.

**Fastest must mean fastest *usable*.** Mirroring the row's own footprint exactly
made a depth-0 FASTEST emit `-c 1024` — honest, but nobody deploys a server that
can't take an 8 k prompt; that's bench-number chasing. Two adjustments, both inside
the invariant. (1) The emitted `-c` is floored at the usable-context floor
(`--min-context`, default 8 k, per `--parallel` slot) — but the floor only rides as
far as *verified evidence* reaches: the deepest depth the **same launch config**
measured OK anywhere in the sweep (`verified_depth_of`; server-driver siblings
shared one session sized at the group's max depth, bench siblings each loaded their
own). No deeper evidence ⇒ the small verified `-c` stands and the report says so.
(2) FASTEST itself is picked among configs that hold the floor, falling back to the
raw fastest only when nothing can (e.g. a native context below the floor). So the
headline is the best *short-request* speed on a deployable server, while BALANCED
stays the best speed *measured at* depth ≥ the floor — speed while actually deep in
context.

Known residual: rows verified by **llama-bench** (rather than the server driver)
allocate slightly less than `llama-server` does for the same context, so a
bench-verified `-c` can still be marginally optimistic. The `q8_0` hint is the safety
valve; a VRAM-headroom check driven off `--measure-vram` peaks is the eventual
belt-and-suspenders.

## Refinement must not eat the tradeoff axis, and must right-size its runs

The `--iterate` funnel refines toward a single throughput optimum, but the report is a
*tradeoff* (fastest / balanced / longest-context). Two defects came from refinement
collapsing the design without the surrounding machinery adapting:

**Depth is the tradeoff axis, not a knob.** `refine_factors` treated `n_depth` like any
other factor and, once `ngl` dominated, settled it to a single winning value. The final
pass then held one depth, so **FASTEST / BALANCED / MAX-CONTEXT all resolved to the same
row** — three identical recommendations. Context length is the independent variable the
whole report is organized around; settling it defeats the purpose. Fix: refinement holds
`n_depth`'s full spread across every pass and tunes only the performance knobs, so the
final pass still maps the whole curve in one measurement regime.

**Don't run a 25-row array to sweep one factor.** `choose_array` sized the design on the
count of *all* factors, including the ones refinement had already pinned to a single
level. A lone 5-level `ngl` among four pinned constants still drew an L25 — 25 runs that
are really 5 configs replicated 5× (on top of the 3 internal bench reps). Constants carry
no information for an orthogonal array. Fix: size on the *active* (multi-level) factors
only; with ≤1 active factor an array is degenerate, so enumerate its levels directly (a
one-way sweep) and attach the constants to each row. Combined with keeping `n_depth`
spread, a refined pass is a real `depth × knob` design (every run a distinct data point)
rather than single-axis replication.

## Thermal drift is the dominant noise source

The two fixes above make the *accounting* honest, but on a throttling card the *numbers*
still aren't unless temperature is controlled. Measured back-to-back, an identical config
(`ngl 60 / depth 49152 / f16 / ub 1024`) ran **tg 13.5 t/s in pass 1 but 7.5 in pass 3** —
~1.8×, purely thermal (the MI50 heats over a long sweep). That nuisance swing (~80%) is
larger than most factor effects the sweep is trying to resolve (`ngl 56→60` moved eff only
~13%), so an uncontrolled search partly measures GPU temperature, and cross-pass numbers
are not comparable — a naive merge of passes would crown pass 1's cool-start outlier.
Randomized run order (already done) decorrelates drift from factors *within* a pass but
does not remove the cross-pass offset. The implemented lever is a **"wait and watch"
settle** (default on): capture the idle GPU temperature once before the sweep, then
between runs poll the sensor (`rocm-smi`/`nvidia-smi`) and block until it falls back to
within a few °C of that baseline, so every config is measured from a comparable thermal
state. It is capped and plateau-aware — where a plateau means *cooling stalled*, not any
small delta: a rising temperature (post-run heat soak) keeps the wait alive rather than
exiting at the hottest moment, while an already-settled card returns at once so idle
runs don't wait. It degrades to a fixed `--cooldown` when no sensor is present and is
disabled with `--no-thermal-wait`. The baseline is captured **once, before any GPU
work**, and handed to every `--iterate` child pass (internal `--thermal-baseline`) — a
child re-capturing "idle" at the start of pass 2 would bake a hot card into the target
and neuter the settle for that whole pass. The Morris screen (which decides what gets
*dropped*), the confirmation run (whose prediction comes from settled numbers), and the
ceiling probe all settle the same way, and every sweep row records its start
temperature (`temp_c` in the CSV) so thermal comparability is checkable after the fact
rather than assumed. This is a *cool-start* baseline — it
buys comparability (a correct winner) cheaply; it does not by itself make the absolute
t/s match sustained-hot reality. The remaining honest step is an **empirical
confirmation** of the top few candidates re-measured head-to-head, rather than an
additive-model extrapolation.
