# llama-optimize

Find good `llama.cpp` command-line parameters for a given GGUF model **on your
machine**, automatically, using statistically-designed experiments instead of a
brute-force sweep. Point it at a model; it detects the hardware, runs a small,
balanced set of benchmarks, and hands you paste-ready commands for the **fastest
(usable)**, **balanced**, and **max-context** configuration — plus the
speed-vs-context Pareto frontier. The DOE engines (Taguchi + Morris) come from the
[`robust`](https://github.com/bigattichouse/robust) suite, vendored as a submodule.

It works on **AMD (ROCm) or NVIDIA (CUDA)**, tunes **every knob llama.cpp exposes**
(see the [knob reference](#knob-reference-one-stop-shop)), can measure **MTP /
speculative decoding**, **ngram self-speculation** (`--ngram`), and
**multi-user concurrency** via a server driver, and is
**crash-safe** — a setting that reboots the box won't be retried into a loop.

```bash
# plan only — prints the experiment matrix and commands, uses NO GPU
python3 llama-optimize.py /path/to/model.gguf

# just tune it — autonomous, one command
python3 llama-optimize.py /path/to/model.gguf --run

# tune for how you actually run it (see "Use cases" below)
python3 llama-optimize.py /path/to/model.gguf --run --use-case agents

# fast screen (1 rep) vs thorough (5 reps + confirm); the array is auto-chosen
python3 llama-optimize.py /path/to/model.gguf --run --quick
python3 llama-optimize.py /path/to/model.gguf --run --full

# the full funnel: screen many knobs → refine the few that matter → confirm + report
python3 llama-optimize.py /path/to/model.gguf --run --screen --iterate 3 --html report.html
```

---

## Setup

You need three things: this repo, the `robust` DOE library (a **git submodule** —
a second repo nested inside this one), and a `llama.cpp` build for your GPU.

**1. Get the code, including the submodule.** The DOE engines live in a separate
project (`robust`) that git tracks as a *submodule* in `./robust`: a plain
`git clone` leaves that folder empty, so you must pull it in explicitly.

```bash
# if you're cloning fresh, grab the submodule in one step:
git clone --recurse-submodules https://github.com/bigattichouse/llama-optimize
cd llama-optimize

# already cloned (or downloaded a zip) without the submodule? fetch it now:
git submodule update --init
```

If `robust/` is empty, step 1 didn't run — `git submodule update --init` fixes it.

**2. Build `robust`.** It's pure C, no GPU needed, and takes a few seconds. This one
build produces everything both funnel stages need:

```bash
make -C robust           # builds libtaguchi.so (Taguchi arrays + main-effects)
                         # and the morris binary (for --screen)
```

`llama-optimize` finds the Taguchi Python binding and the `morris` binary under
`robust/` automatically by searching the submodule — no paths to set.

**3. Point it at your `llama.cpp`.** You need `llama.cpp` built for your GPU
(ROCm/HIP, CUDA, Metal, …). The tool auto-discovers the binaries in the default
workspace layout; otherwise pass **`--llama-cpp /path/to/llama.cpp`** (its root or
`build/bin` dir). It also reads `$LLAMA_CPP` and `$PATH`, or you can pass
`--llama-bench`/`--llama-server` directly. If the binaries can't be found the tool
stops with a clear error.

**4. Run it** — the commands above (e.g. `python3 llama-optimize.py model.gguf --run`).
Python 3.10+ is the only other requirement (uses `X | None` syntax, standard library
only). Verify the install with no GPU or model via `python3 llama-optimize.py --selftest`.

---

## Why designed experiments (Morris + Taguchi) instead of a full sweep?

The knobs that matter for llama.cpp throughput interact, and testing every
combination explodes fast. With 5 factors at 5 levels each a full factorial is
`5^5 = 3125` runs; add a dozen more `--factor`s and it's astronomical. `llama-optimize`
replaces the sweep with a **two-stage DOE funnel**, both stages powered by the
vendored [`robust`](https://github.com/bigattichouse/robust) suite:

**1. Morris screening (`--screen`) — "which knobs even matter?"** Elementary-effects
screening walks `R` trajectories through the factor space (~`R·(k+1)` runs) and, for
every knob, reports **μ\*** (how much it moves throughput) and **σ** (how much its
effect depends on the others — i.e. interactions/nonlinearity). Negligible knobs get
**dropped** (pinned at their best-seen level) so the expensive stage never wastes
runs on them. This is what makes it practical to throw a dozen `--factor`s at the
tool. Runs the `robust` **morris** binary.

**2. Taguchi orthogonal array — "what's the optimum among the survivors?"** A Taguchi
array (**L25** for up to 6 varying factors, **L125** beyond — auto-sized) estimates
every surviving factor's main effect in 25–125 runs instead of thousands, keeping
the levels balanced so each effect reads independently. `--iterate` then refines: it
settles the low-impact factors at their winner and re-runs the high-impact ones on a
finer grid, converging on the optimum — and the final pass folds **every** earlier
pass's measurements back into its report, so refinement can only add information.
`--confirm` measures the predicted-optimal config directly to check the additive
model held. Runs the `robust`/`taguchi` library via its Python binding.

```
many knobs ─► MORRIS (μ*, σ; ~R·(k+1) runs) ─► the few that matter
           ─► TAGUCHI L-array + --iterate ─► --confirm ─► optimum
```

Both stages are optional and compose: `--screen` alone screens, a bare `--run` goes
straight to Taguchi, and `--screen --iterate N --confirm` runs the whole funnel.
(**Sobol** variance attribution was considered and dropped — see Roadmap; Morris `σ`
already flags interactions cheaply.)

---

## What it tunes

Swept by default (auto-scaled to your hardware and model):

| Factor        | Flag             | Levels (example, Qwen3.6-27B on MI50) | Notes |
|---------------|------------------|----------------------------------------|-------|
| GPU layers    | `-ngl`           | `0, 16, 32, 48, 64`                    | biggest lever; top = model's real layer count |
| Context depth | `-d` (n-depth)   | 5 levels `0..min(native ctx, 65536)`   | KV pre-fill; the speed-vs-context axis, adaptive to the model's native context |
| CPU threads   | `-t`             | `4, 6, 8, 12, 16`                      | auto-derived around the physical-core count |
| KV cache type | `-ctk`/`-ctv`    | `f16, q8_0` (default)                  | KV precision; **floored to near-lossless** by `--min-kv q8_0` |
| Micro-batch   | `-ub`            | `128, 256, 512, 1024, 2048`            | prefill/decode balance |
| KV offload    | `-nkvo`          | `0, 1`                                 | KV cache in VRAM vs system RAM — the VRAM-vs-PCIe lever |
| CPU polling   | `--poll`         | `0, 50, 100`                           | busy-wait level for CPU-side work |
| Logical batch | `-b`             | `2048, 4096, 8192`                     | prompt chunking (levels start at max `-ub` so the `b≥ub` clamp never aliases) |
| Tensor placement | `-ot`         | `none, ffn_up_cpu, ffn_cpu`            | **dense models**: FFN tensors on CPU at full `-ngl` often beats dropping whole layers |
| MoE expert offload | `-ncmoe`   | `0 .. n_layers` (5 levels)             | **MoE models** (replaces `-ot`): how many layers keep experts on CPU |
| NUMA policy   | `--numa`         | `distribute, isolate`                  | **multi-NUMA-node boxes only** (inert on one node) |
| Prefill threads | `-tb`          | same levels as `-t`                    | **server driver**: decode vs prefill thread split |
| ngram variant  | `--spec-type <variant>` | `none, ng-simple, ng-mod, ng-map-k, ng-map-k4v` | **server driver, --ngram**: which pattern-matching variant — *screened* first (see [ngram staging](#ngram-staged-search)) |
| ngram map params | `--spec-ngram-<variant>-size-n/m, -min-hits` | `4..24 / 8..64 / 1..5`  | **--ngram tuning stage**: lookup n-gram, draft m-gram, and min hit thresholds (ngram-simple / map-k / map-k4v) |
| ngram mod params | `--spec-ngram-mod-n-match/min/max` | `8..48 / 16..96 / 32..128` | **--ngram tuning stage**: ngram-mod hasher lookup length and range |
| MTP on/off    | `--spec-type draft-mtp` | `1, 0`                          | **MTP models, server driver**: measures what speculative decoding actually buys |
| MTP draft len | `--spec-draft-n-max` / `-min` | `1..6` / `1, 2`           | **MTP models, server driver**: speculative draft lengths |
| MTP acceptance | `--spec-draft-p-min` / `-p-split` | `0.0..0.9` / `0.1..0.5` | **MTP models, server driver**: draft acceptance thresholds |

The KV factor is quality-gated: `--min-kv` (default `q8_0`, near-lossless) drops the
lossier levels so the tool never recommends a KV type that degrades output over long
context. Pass `--min-kv any` to explore the full `f16, q8_0, q5_1, q4_1, q4_0` ladder
(quantizing the KV cache buys context at some quality cost).

**Fixed** (not swept): flash-attention on (`-fa 1` — a near-certain win on gfx906 and
a *precondition* for quantized KV cache, so sweeping it would structurally fail every
`fa=0` × KV-quant row) and mmap on (bench `-mmp 1`; the server default). Sweep fa
explicitly with `--factor fa=0,1 --min-kv f16`. Also not swept by default: CPU
affinity masks (`cpu_mask`/`cpu_range` — no universal levels), RoPE/YaRN scaling
(extends context by *changing model behavior*, a quality tradeoff, not a perf knob),
and `parallel` (a property of your workload — set it via `--use-case`/`--parallel`).

Up to 6 varying factors fit an **L25** array (25 runs); the default set above
overflows that, so a bare `--run` now draws an **L125** (125 runs, a few hours).
Use `--screen` to Morris-prune the factors that don't matter first (usually
funneling the sweep back down to an L25), `--quick` for 1 rep/config, or pin
factors (`--factor threads=8`) to shrink the design.

---

## Use cases (`--use-case`) — start here

Most people don't want to think about drivers and concurrency — they know *how
they run the model*. `--use-case` is a **runbook**: one friendly name that expands
into the right bundle of lower-level flags (driver + request profile + concurrency).

| `--use-case` | Driver | Request (prompt + gen) | Streams | For |
|--------------|--------|------------------------|:-------:|-----|
| **app** | `llama-bench` | 512 + 256 | 1 | a general/embedded llama.cpp app — raw single-stream throughput |
| **single** | `llama-server` | 512 + 256 | 1 | llama-server for **one** user/worker (measures MTP too) |
| **agents** | `llama-server` | 8192 + 256 | 4 | several **autonomous agents** — long tool-use prompts, concurrent |
| **multi-user** | `llama-server` | 1024 + 256 | 8 | **many concurrent chat users** — short prompts, high concurrency |

```bash
python3 llama-optimize.py model.gguf --run --use-case app          # embedded / CLI app
python3 llama-optimize.py model.gguf --run --use-case single       # one-user server
python3 llama-optimize.py model.gguf --run --use-case agents       # 4 concurrent agents
python3 llama-optimize.py model.gguf --run --use-case multi-user   # 8 concurrent users
```

**Precedence: built-in defaults < `--use-case` < your explicit flags.** A runbook
only *fills in* the flags you didn't set, so you can tweak any single dimension
without abandoning the bundle:

```bash
# agents runbook, but pin it to 2 streams instead of 4
python3 llama-optimize.py model.gguf --run --use-case agents --parallel 2
```

### The underlying knobs (`--profile` / `--driver` / `--parallel`)

A use-case is just a named bundle of these; set them directly for full control.
The **profile** sets the representative request shape the sweep optimizes for:

| `--profile` | Request (prompt + gen) | Ctx floor | Default driver | For |
|-------------|------------------------|-----------|----------------|-----|
| **single** (default) | 512 + 256 | 8192 | bench | interactive chat/coding |
| **agents** | 8192 + 256 | 32768 | bench | big-context tool use / RAG |
| **multi** | 1024 + 256 | 8192 | server | concurrent serving (`--parallel N`) |

The objective the stats optimize is **generation throughput** (`tg` t/s) by
default: prompt-processing speed is measured and reported, but doesn't move the
fits or the picks — blending it in lets a config with slow decode but a huge
prefill number outrank one that generates far faster. If your workload really is
prefill-bound, `--score eff` switches the objective to **effective throughput**
for that request — `(P + G) / (P/pp_tps + G/tg_tps)` — which weighs prefill and
decode the way the workload experiences them.
Override the shape with `--n-prompt/--n-gen/--ctx-floor`, the engine with
`--driver bench|server`, and concurrency with `--parallel N`.

## What it measures

For each config, `llama-bench` reports two throughput numbers, both captured:

- **`tg_tps`** — token-generation t/s (decode speed; what you feel interactively).
  This is what the optimizer maximizes.
- **`pp_tps`** — prompt-processing t/s (prefill speed; matters for long-context/RAG).
  Reported alongside; `--score eff` folds it into the objective.

Runs that **OOM, crash, or time out** are recorded as data (`tg=0`, with a status of
`OOM`/`ERROR`/`TIMEOUT`) rather than aborting the sweep. High context depth at low
`-ngl` with an `f16` KV cache is *expected* to OOM — that failure is the memory cliff
we're mapping, not a bug.

---

## Auto-detection

The tool inspects the box and the model so you don't hand-tune the factor levels:

- **Physical / logical cores** — unique `(physical id, core id)` pairs from
  `/proc/cpuinfo` (fallback: `logical / 2`). Thread levels bracket the physical-core
  count, where llama.cpp throughput usually peaks.
- **VRAM** — via `rocm-smi` (AMD) or `nvidia-smi` (NVIDIA); best-effort,
  informational. Everything else is vendor-agnostic — the tool just drives
  `llama-bench`/`llama-server`, so it works on ROCm, CUDA, or Metal builds alike.
- **Model layer count** — a minimal, dependency-free GGUF metadata reader parses
  `<arch>.block_count` from the header (no tensors loaded), so `-ngl`'s top level is
  the model's real layer count.

---

## Output

The samples below are real (trimmed) output from a 3-pass sweep of a 31B Q6_K
model, so you know what to expect from each section.

**During the sweep** — one line per config as it finishes, with a running ETA:

```
[23/25] run 15: 58/60 layers on GPU, 5 threads, q8_0 KV cache, 65536-token context, ubatch 512 -> OK tg=7.4 t/s (pp=122.0) (125s)  [23/25 done, elapsed 1h41m, ETA ~8m49s]
```

**The three picks** — each with a copy-paste `llama-server` command, its `-c`
sized to what the sweep actually verified for that config, and a prefill-cost
estimate for that `-c`. With `--verify-picks` (default on) each pick also carries
a `verified: median of N measurements (spread X%)` line:

```
RESULTS: 72/72 configs succeeded

### FASTEST (max speed, usable context)
  tg=10.8 t/s  (pp=134.1)  depth=32768  ngl=60  t=6  kv=q8_0  ub=512
  suggested llama-server command:
    ./llama-server -m gemma-4-31B-it-UD-Q6_K_XL.gguf -c 33792 -fa 1 -ngl 60 \
      -t 6 -ctk q8_0 -ctv q8_0 -ub 512 -nkvo --poll 50 -b 2048
  prefill cost: a full 33792-token prompt ≈ 4m12s to first token at pp=134 t/s (an 8k prompt ≈ 1m01s)

### BALANCED (best with context >= 8192)
  tg=10.8 t/s  (pp=134.1)  depth=32768  ngl=60  t=6  kv=q8_0  ub=512
  suggested llama-server command: …

### MAX CONTEXT
  tg=7.4 t/s  (pp=122.0)  depth=65536  ngl=58  t=5  kv=q8_0  ub=512
  suggested llama-server command: …
```

**Probed ceiling** — stage 2 binary-searches how large `-c` can go before the
model fails to load. This deliberately goes *beyond* the swept depth grid, so
the t/s there is a single spot-check, not a swept measurement:

```
### PROBED CEILING (largest -c that loads — beyond the swept range)
  ~262144 tokens (the model's native limit)  tg=6.6 t/s spot-check there  ngl=58  kv=q8_0  ub=512
  suggested llama-server command (-c 235520, ~10% headroom under the ceiling): …
```

**Pareto frontier** — the speed↔context trade-off curve: the configs where no
other measured config is both deeper *and* faster:

```
### Pareto frontier (context vs generation t/s)
  depth= 32768  tg=  10.8 t/s  ngl= 60  kv=q8_0  ub= 512
  depth= 49152  tg=   8.7 t/s  ngl= 60  kv=q8_0  ub= 512
  depth= 65536  tg=   7.4 t/s  ngl= 58  kv=q8_0  ub= 512
```

**Taguchi main effects** — which knobs actually matter, ranked by impact (here
tensor-offload dominates, `ngl` is worth ~1 t/s, threads barely matter):

```
Main Effects (sorted by range, descending):
  ot                   range=  6.2910  means=[7.8980, 3.2560, 1.6070]
  ngl                  range=  1.1230  means=[4.3730, 4.5460, 4.5780, 4.9240, 5.4950]
  n_depth              range=  0.7410  means=[4.6590, 4.6410, 5.3010, 4.7540, 4.5600]
  threads              range=  0.3680  means=[4.6150, 4.9830, 4.7200]

Predicted-optimal levels: {'ngl': '60', 'n_depth': '32768', 'threads': '5', 'ot': 'none'}
```

**Confirmation run** (`--confirm`/`--full`) — re-measures the predicted-optimal
config; a large gap means factor interactions or thermal drift, so trust the
measured Pareto picks over the additive prediction:

```
### Confirmation run (predicted-optimal config)
  predicted tg: 9.3 t/s
  measured  tg: 7.7 t/s  (pp=109.6, status=OK)
  prediction error: 17%  → LARGE gap: interactions likely — trust the Pareto pick
```

The objective is **generation t/s** by default; with `--score eff` the reports
switch to blended effective throughput for the profile's request shape,
`(P+G)/(P/pp+G/tg)`. Full per-run data is written to `results.csv` (incrementally,
so it survives a crash — see below).

### Reading the results correctly

- **Trust the Pareto frontier and the raw `results.csv`** for the actual
  recommendation — those read measured numbers directly.
- **Use the main-effects table to rank which knobs matter**, not as gospel for the
  single best config. Because OOM is scored as `0 t/s` and is driven by an
  *interaction* (the `ngl × n_depth × kv_type` memory budget), the additive
  main-effects model can misattribute an interaction to a main effect. This is why
  the recommendation is driven off the Pareto, and why we keep the spare column for
  an error estimate.
- **Beware thermal drift.** GPUs (notably the MI50) throttle under sustained load,
  so throughput drifts *down* over a long sweep — the same config can measure 40%+
  slower late in the run than early. Two defaults fight it: **execution order is
  randomized** (`--seed` to reproduce, `--no-shuffle` to disable), and between runs
  the tool **waits for the GPU to fall back near its idle temperature** (baseline
  captured once, up front; `--no-thermal-wait` disables it, `--cooldown` is the
  fixed fallback when no sensor is readable). Each row records its start temp in
  the CSV (`temp_c`) so you can check comparability afterwards. For steadier
  numbers use `--full` (more reps). If `--confirm` reports a large predicted-vs-
  actual gap, suspect either interactions *or* thermal drift.

---

## Recommended workflow (staged / iterative refinement)

**Many knobs? Screen first (the funnel).** With a dozen-plus `--factor`s, run a
**Morris pre-screen** to find which knobs even matter before spending a full sweep:
```bash
python3 llama-optimize.py model.gguf --run --screen --iterate 2 \
  --factor ngl=0,64 --factor ubatch=128,2048 --factor kv_type=f16,q8_0,q4_0 \
  --factor nkvo=0,1 --factor poll=0,50 --factor ot=none,ffn_cpu
```
`--screen [R]` uses the vendored `robust` **morris** tool: ~`R·(k+1)` cheap runs to
rank every knob by **μ\*** (importance) and flag **σ** (interaction/nonlinearity).
It then **drops the negligible knobs** (fixed at their best-seen level) and continues
into the Taguchi sweep / `--iterate` on the ones that matter. That's the funnel:

```
many knobs ─► MORRIS (μ*, σ; ~R·(k+1) runs) ─► the few that matter ─► TAGUCHI/--iterate ─► optimum
```

Morris is *screening*, not optimization — it answers "which knobs matter?" cheaply,
so the expensive Taguchi sweep only spends runs where they count. (Sobol variance
attribution was considered for quantifying interaction strength and dropped — see
Roadmap; Morris `σ` already flags interactions for free.)

**Automatic:** let the tool do the staging for you —
```bash
python3 llama-optimize.py model.gguf --run --iterate 3
```
`--iterate N` runs N passes: pass 1 screens coarsely, then each pass **settles the
low-impact factors at their winner and refines the high-impact ones onto a finer
grid** around their best value, converging on the optimum (stops early if factors
converge). Each pass writes `results.passN.csv`; the final pass also runs
**pick verification** (medians over re-measurements), the **max-context probe**,
and `--confirm`/`--html` if requested. This *is* the loop below, automated.

Every pass after the first also **merges all earlier passes' rows into its
report, picks, and Pareto** (`--merge-results`, added automatically), so the
final answer is drawn from *everything* measured — a refinement pass that
wanders into a worse region (noise, drift) can no longer make the final report
worse than pass 1. Re-measured configs are deduplicated (a merged row is kept
only if it beats every known measurement of that exact config), so the tables
don't repeat rows across passes. Refinement *decisions* and the main-effects
table still use only the current pass's balanced design.

**Manual** (if you want to steer each pass yourself) — the same idea, by hand:

1. **Screen** — a quick coarse sweep to rank the knobs:
   ```bash
   python3 llama-optimize.py model.gguf --run --quick
   ```
   Read the **main-effects "impact"** ranking. Factors with a **wide window** (large
   range) are where the throughput lives — commonly `ngl`, **context** (`n_depth`),
   and, for MTP models, **`spec_n_max`**. Factors with a tiny range are settled.

2. **Refine** — a focused pass that gives the wide-window factors **more levels** and
   pins the flat ones at their winner (via `--factor`):
   ```bash
   python3 llama-optimize.py model.gguf --run --full \
     --factor ngl=56,58,60,62,64 \
     --factor n_depth=0,3072,6144,9216,12288 \
     --factor spec_n_max=1,2,3,4,5 \
     --factor kv_type=f16,q8_0 --factor threads=8
   ```
   More levels on a factor = finer resolution across its window. Repeat, zooming in
   each time (e.g. once you see `spec_n_max` peaks near 4, sweep `3,4,5,6`).

3. **Confirm** with `--confirm` (or `--full`): the tool runs the predicted-optimal
   config directly and reports predicted-vs-actual — a small gap means the additive
   model held; a large gap means interactions (or thermal drift) dominate, so trust
   the Pareto pick. Add `--html report.html` for a visual report.

4. **Revisit anytime, no GPU** — `--report-only` rebuilds the full report (and
   `--html`) from the results CSVs, including the probed ceiling and verified
   medians via their sidecars (`<results>.probe.json` / `.verify.json`); use
   `--merge-results` for the other passes. `--diff old.csv new.csv` compares two
   sweeps of the same factor space after a llama.cpp upgrade or quant swap.

**Expanding the array vs. more runs.** More *levels* on the wide-window factors (a
finer grid) is usually more informative than a bigger array. If you want raw
statistical power (replication to average out thermal/measurement noise), force a
larger array like `--array L125` — but that's a 125-run, overnight job. Randomized
order (default) plus `--full` reps already averages out most drift.

---

## CLI reference

```
python3 llama-optimize.py MODEL.gguf [options]

  --array A          orthogonal array (default: auto-picks the smallest that
                     fits your factors; advanced: force L9/L25/L27/L125/...)
  --confirm          run the predicted-optimal config to verify the additive
                     model (predicted vs actual; implied by --full)
  --cooldown SECS    fixed pause between runs so the GPU can cool — the fallback
                     when no temp sensor is readable (default: 0)
  --ctx-scan         probe the ceiling FIRST, then set the n_depth axis to fractions
                     of it (0, ¼, ½, ¾, 0.9×) so the Pareto spans your full range
  --ctx-size N, -c N tune at a FIXED context (like llama.cpp -c) = min==max==N
  --diff OLD.csv NEW.csv  compare two sweeps of the same factor space (llama.cpp
                     upgrade, driver update, quant swap): per-config tg deltas,
                     status changes (OOM -> OK, ...), and whether the old winner
                     still wins. No model or GPU needed; exits after the report
  --driver bench|server  benchmark driver (default: from profile; MTP-capable
                     models auto-switch to server). 'server' measures real
                     generation incl. MTP + concurrency
  --env NAME=v1,v2,...      sweep an environment variable as a factor (repeatable)
  --factor NAME=v1,v2,...   sweep/override a knob (repeatable; see Knob reference)
  --full             thorough: 5 reps/config (steadier, slower)
  --html PATH        also write a visual HTML report (Pareto + main effects)
  --iterate N        run N auto-refining passes (screen -> refine -> ...): settle
                     low-impact factors, refine high-impact ones on a finer grid;
                     the final report/picks merge ALL passes' results
  --llama-bench PATH path to the llama-bench binary
  --llama-cpp PATH   path to llama.cpp (root or build/bin); also $LLAMA_CPP/$PATH
  --llama-server PATH  path to the llama-server binary
  --max-context N    cap the context axis and the ceiling probe (alias: --max-depth)
  --merge-results CSV  fold rows from an earlier results CSV into this run's
                     report/picks/Pareto without re-running them (repeatable;
                     --iterate adds these automatically for earlier passes)
  --min-context N    minimum context you need: BALANCED targets it, FASTEST only
                     considers configs verified to hold it, and emitted -c is
                     floored at it where the sweep has evidence (alias: --ctx-floor)
  --min-kv TYPE      KV-cache quality floor (default q8_0, near-lossless); never
                     recommends a lossier KV. 'any' to explore all (q5/q4)
  --n-gen N          generated tokens per measurement (default: from profile)
  --n-prompt N       prompt tokens per measurement (default: from profile)
  --no-mtp           don't add draft-mtp flags to the server command
  --ngram            enable ngram self-speculative decoding (server only): a
                     screen stage measures every variant, then the top variants
                     get their parameters tuned (staged, see ngram staging below)
  --ngram-type V     pin the ngram variant and tune only its parameters in one
                     pass (ngram-simple | ngram-mod | ngram-map-k | ngram-map-k4v)
  --ngram-keep K     how many top screened variants to tune (default: 2)
  --ngram-fast       greedy: tune only the single best-screened variant (K=1)
  --no-probe         skip the max-context probe (runs by default: binary-searches
                     the physical context ceiling for the furthest-reaching config,
                     reported as PROBED CEILING with a ready command at ~90% of it)
  --no-shuffle       run in array order (default: randomized, see below)
  --no-thermal-wait  disable the default settle between runs (waits until GPU temp
                     falls back near its idle baseline; keeps runs comparable)
  --parallel N       concurrent streams for the server driver
  --profile P        workload profile: single | agents | multi (default: single)
  --quick            fast screen: 1 rep/config (noisier, ~1/3 the time)
  --report-only      rebuild the report (and --html) from an existing results
                     CSV — no GPU, no llama.cpp. Reads --results, folds in
                     --merge-results; PROBED CEILING is included when the sweep
                     saved its probe result (<results>.probe.json)
  --reps N           repetitions per config (default: 3, or --quick=1/--full=5)
  --results NAME     results CSV name inside --results-dir (default: <model>.csv)
  --results-dir DIR  directory for all output (default: results/, gitignored)
  --resume           skip runs already in --results (rows save incrementally,
                     so an interrupted sweep can be resumed)
  --retry-crashed    on resume, also retry configs that were started but never
                     finished (suspected crash/hang); default skips them
  --run              actually execute the sweep (default: plan/dry-run, no GPU)
  --score tg|eff     objective for stats/fits/picks: 'tg' (default) ranks by
                     generation speed alone (pp measured + reported, not scored);
                     'eff' ranks by blended effective t/s for the request shape
  --screen [R]       Morris pre-screen (R trajectories, default 6): rank knobs by
                     importance, drop the negligible ones, then sweep the rest
  --seed N           seed the randomized order (reproducibility)
  --selftest         run offline logic checks and exit (no GPU, no model)
  --server-start-timeout SECS  give up on a config if llama-server doesn't load
                     in this long (default: 180; also fails fast if it dies)
  --spec-draft-n-max N  draft tokens for MTP / draft-model speculative decoding
                     (default: 2). Not used by ngram (ngram variants carry their
                     own draft-length knobs)
  --thinking         tune for reasoning workloads (long decode, n_gen~2048);
                     default is non-thinking / short answers
  --timeout SECS     per-run timeout (default: 1200)
  --use-case U       runbook that bundles driver+profile+concurrency:
                     app | single | agents | multi-user (see Use cases above).
                     --driver/--profile/--parallel override the runbook.
  --verify-picks R   re-measure each pick candidate R extra times and report the
                     median of all its measurements (default: 2, --quick=0/--full=3;
                     0 disables) — guards the headline numbers against thermal /
                     run-to-run noise; medians persist to <results>.verify.json
  --vram             measure actual peak VRAM per run (polls rocm-smi/nvidia-smi);
                     records vram_mib and overlays the VRAM curve + physical
                     ceiling on the Pareto chart
```

(Flags are listed alphabetically — both here and in `--help`.)

Results are written **incrementally** (one row per run, flushed), so a crash or
timeout never loses completed runs — rerun with `--resume` to finish the rest.

**Crash-safe (reboot protection).** Some configs can hard-hang or spontaneously
reboot the machine (driver faults, OOM at the kernel level, flaky GPUs). Before
each run — and before each server model-load — the tool writes a durable,
`fsync`-ed record to `<results>.journal`. On `--resume`, any config that was
*started but never produced a result* is treated as the suspected culprit,
recorded as `CRASH`, and **skipped instead of retried** — so a bad setting can't
put you in a reboot loop. Use `--retry-crashed` to attempt those again once you've
addressed the cause. (Server *load-time* crashes blacklist the whole shared-launch
group, since the fault is in the launch config, not one request.)

> **Note on run time.** Deep-context configs at low `-ngl` prefill their KV cache
> on the CPU, which is slow (tens of seconds to minutes per run on a big model).
> That cost is inherent to measuring throughput *at* context. Use `--reps`,
> `--n-prompt/--n-gen`, and `--max-depth` to trade accuracy/coverage for speed on
> a first pass.

Run `python3 llama-optimize.py --selftest` to verify the JSON parser, OOM detection,
factor-level generation, MoE detection, and Pareto logic without a GPU or model.

## Knob reference (one-stop-shop)

Every tunable the sweep understands. **swept** = in the default design; **opt-in**
= add with `--factor NAME=v1,v2,...`. Any **environment variable** can also be a
factor via `--env NAME=v1,v2` (applied per process; the winning value is prepended
to the recommended command). Adding a new knob is one entry in the `FACTORS`
registry in `llama-optimize.py`.

| knob | flag(s) | driver | kind | when | effect |
|---|---|---|---|---|---|
| `ngl` | `-ngl` | both | num | swept | layers offloaded to GPU (dominant lever) |
| `n_depth` | `-d` | bench | num | swept | context depth (KV prefill); speed-vs-context axis |
| `threads` | `-t` | both | num | swept | CPU threads for decode |
| `kv_type` | `-ctk -ctv` | both | cat | swept | KV cache precision (buys context) |
| `ubatch` | `-ub` | both | num | swept | physical micro-batch |
| `ncmoe` | `-ncmoe` | both | num | swept¹ | MoE expert layers kept on CPU |
| `batch` | `-b` | both | num | swept | logical batch (levels ≥ max ubatch) |
| `nkvo` | `-nkvo` | both | bool | swept | keep KV in RAM vs VRAM |
| `poll` | `--poll` | both | num | swept | CPU polling level |
| `numa` | `--numa` | both | cat | swept⁴ | NUMA optimization mode |
| `cpu_mask` | `-C` | both | cat | opt-in | CPU affinity mask (hex, e.g. `0xFF`) |
| `cpu_strict` | `--cpu-strict` | both | cat | opt-in | strict CPU placement (0/1) |
| `cpu_range` | `-Cr` | server | cat | opt-in | CPU affinity range (`lo-hi`) |
| `fa` | `-fa` | both | cat | opt-in² | flash attention on/off |
| `ot` | `-ot` | both | cat | swept¹ | per-tensor placement — the VRAM-fit lever (named patterns below) |
| `threads_batch` | `-tb` | server | num | swept | CPU threads for prompt processing |
| `parallel` | `--parallel` | server | num | opt-in | concurrent request streams (multi) |
| `ngram` | `--spec-type <variant>` | server | cat | swept⁴ | ngram self-speculation variant (the gate): none, ngram-simple, ngram-mod, ngram-map-k, ngram-map-k4v |
| `ngram_size_n` | `--spec-ngram-<v>-size-n` | server | num | tuning⁴ | lookup n-gram length (ngram-simple / map-k / map-k4v — one factor, variant-spelled) |
| `ngram_size_m` | `--spec-ngram-<v>-size-m` | server | num | tuning⁴ | draft m-gram length (simple / map-k / map-k4v) |
| `ngram_min_hits` | `--spec-ngram-<v>-min-hits` | server | num | tuning⁴ | min hits to draft (simple / map-k / map-k4v) |
| `ngram_mod_n_match/min/max` | `--spec-ngram-mod-n-*` | server | num | tuning⁴ | ngram-mod hasher: lookup length, min/max range |
| `mtp` | `--spec-type draft-mtp` | server | cat | swept³ | speculative decoding via the model's MTP head, on/off |
| `spec_n_max` | `--spec-draft-n-max` | server | num | swept³ | max draft tokens (MTP / draft-model only — **not** ngram) |
| `spec_n_min` | `--spec-draft-n-min` | server | num | swept³ | MTP draft tokens (min) |
| `spec_p_min` | `--spec-draft-p-min` | server | float | swept³ | MTP acceptance-probability threshold |
| `spec_p_split` | `--spec-draft-p-split` | server | float | swept³ | MTP split probability |
| `rope_scaling` | `--rope-scaling` | server | cat | opt-in | RoPE scaling: none/linear/yarn |
| `yarn_factor` | `--yarn-ext-factor` | server | float | opt-in | YaRN extrapolation (context **beyond** native) |
¹ `ncmoe` swept for MoE models, `ot` for dense ones (the same placement lever, per
architecture).  ² fixed on unless swept (precondition for KV-quant; pair with
`--min-kv f16`).  ³ swept when the model ships an MTP/NextN head — such models also
auto-switch to the server driver so the effect is measured (`--no-mtp` disables).
⁴ with `--ngram` (server driver auto-switches), the *variant* is screened first;
each variant's tuning knobs are then swept only in that variant's tuning stage —
they never share one array with the gate (see [ngram staging](#ngram-staged-search)).
ngram needs no draft model — it pattern-matches from the token history.  Note
`spec_n_max` (`--spec-draft-n-max`) is a draft-model/MTP knob and has **no effect
on ngram**, so it is not an ngram factor.

**`-ot` named patterns** (translate to real tensor regexes): `none`, `ffn_cpu`,
`ffn_up_cpu`, `exps_cpu`, `attn_cpu`.

```bash
# offload placement + CPU polling + KV location
python3 llama-optimize.py model.gguf --run \
  --factor ngl=56,60,64 --factor ot=none,ffn_cpu --factor nkvo=0,1 --factor poll=0,50

# tune the MTP surface (server driver)
python3 llama-optimize.py model-UD.gguf --run --driver server \
  --factor spec_n_max=2,3,4,5 --factor spec_p_min=0.0,0.1,0.2

# gfx906 / ROCm environment knobs (the "10-30%")
python3 llama-optimize.py model.gguf --run \
  --env GGML_CUDA_FORCE_MMQ=0,1 --env GGML_CUDA_FORCE_CUBLAS=0,1

# search the ngram self-speculation surface (server driver, staged)
python3 llama-optimize.py model.gguf --run --driver server --ngram
# or pin one variant and tune just its knobs in a single pass
python3 llama-optimize.py model.gguf --run --driver server --ngram-type ngram-mod

**Notes.** MTP/spec and concurrency knobs need `--driver server` (llama-bench can't
do them). Keep the number of levels-per-factor uniform where you can — mixing 2- and
5-level factors forces a much larger array. `--iterate` refines numeric knobs on a
finer grid and keeps the top levels of categorical ones.

<a name="ngram-staged-search"></a>
### ngram staged search

The ngram `--spec-type` variants (`ngram-simple`, `ngram-mod`, `ngram-map-k`,
`ngram-map-k4v`) each have their **own** tuning knobs that mean nothing to the
others — a knob is only live when its variant is selected. Putting all of them in
one orthogonal array would send flags for the wrong variant, blow the run count up
(L25 → L125), and dilute every knob's measured effect (each is inert in most rows).
So `--ngram` runs a **staged search** instead:

1. **Screen** — measure every variant at its default knobs in one clean array.
2. **Keep the top `--ngram-keep` variants** (default 2; `--ngram-fast` keeps just
   the best). Only clearly-worse variants are dropped, so a variant that shines
   only once tuned still gets its chance.
3. **Tune each survivor** — one clean array per variant sweeps only *that* variant's
   knobs, with the other (non-ngram) knobs held at their screen winners.

The reported pick is the best **measured** config across all stages. To skip the
screen and tune one variant directly, use `--ngram-type <variant>`. Background and
the general mechanism for this kind of conditional parameter are in
[`docs/CONDITIONAL-FACTORS.md`](docs/CONDITIONAL-FACTORS.md) and
[`docs/ngram-design.md`](docs/ngram-design.md).

---

## Two benchmark drivers

- **`bench`** (default) — `llama-bench`. Fast, measures raw prefill/decode. Cannot
  do speculative decoding or real concurrency.
- **`server`** — launches `llama-server`, drives real generation over HTTP, and
  measures actual tokens/sec. This is the **only** way to measure **MTP /**
  **speculative decoding** (auto-adds `--spec-type draft-mtp` for models with an
  MTP/NextN head; add `--ngram` for ngram-based self-speculation) and
  **multi-user concurrency** (`--parallel N`, aggregate throughput). The `multi`
  profile selects it automatically; force it on any profile with `--driver server`
  (e.g. to measure MTP on a single-user workload).  Configs that share load-time
  params (everything except context depth) **reuse a single server** sized to the
  group's max context, so it doesn't reload the model for every run.

```bash
# measure the real MTP speedup on a single-user workload (UD quant with NextN head)
python3 llama-optimize.py model-UD.gguf --run --driver server

# tune concurrent serving throughput
python3 llama-optimize.py model.gguf --run --profile multi --parallel 8

# tune the MTP aggressiveness knob directly (server driver only)
python3 llama-optimize.py model-UD.gguf --run --driver server \
  --array auto --factor spec_n_max=1,2,3,4
```

## Implemented

- **Autonomous** — auto-detects CPU cores, VRAM (AMD *and* NVIDIA), model layers,
  MoE (`expert_count`), MTP (`nextn_predict_layers`), and native context; picks the
  array; generates sensible per-model factor levels. Zero flags required.
- **Two drivers** — `bench` (fast, raw pp/tg) and `server` (real generation:
  **measures MTP, ngram self-speculation, and multi-user concurrency**), with
  server reuse across runs.
- **Use-case runbooks** (`--use-case app|single|agents|multi-user`) that bundle
  driver + profile + concurrency, over **workload profiles** (`--profile
  single|agents|multi`) and a selectable objective (`--score`: generation t/s by
  default, or blended effective throughput that weighs prefill vs decode as the
  workload experiences them).
- **The funnel** — `--screen` (Morris: rank knobs by μ\*, flag interactions by σ,
  drop the negligible) → `--iterate N` (auto-refine the survivors) → `--confirm`
  (verify the prediction) → `--html` report.
- **One-stop knob registry** — every llama.cpp lever is sweepable (`--factor`),
  including MTP/spec dials, ngram parameters, `-ot` placement, CPU affinity,
  and env vars (`--env`); MoE/MTP/ngram knobs auto-enabled by model or --ngram.
- **Trustworthy measurement** — realistic prompts, warmup + rep averaging,
  **randomized run order** + a default **thermal settle** between runs (wait for
  the GPU to return near its idle temp; `--cooldown` as the sensorless fallback),
  with each row's start temp recorded (`temp_c`); **pick verification**
  (`--verify-picks`, default on) re-measures the final picks and reports medians
  with the observed spread, so the headline number isn't one lucky rep.
- **Robust** — incremental save + `--resume`; **crash journal** so a config that
  reboots the machine is skipped, not retried (`--retry-crashed` to override);
  clear errors; `--selftest` (no GPU); max-context probe by default (`--no-probe` to skip).

## Roadmap / ideas

The DOE funnel is feature-complete for tuning: **Morris** screens which knobs matter
and flags interactions (`σ`), **Taguchi + `--iterate`** find the optimum, and
**`--verify-picks`/`--confirm`** verify the result. Remaining improvement ideas
(per-rep noise capture, predictive OOM pruning, multi-GPU factors, measured TTFT)
are tracked in [`ROADMAP.md`](ROADMAP.md).

*Considered and dropped:* **Sobol variance attribution.** It would quantify
interaction strength precisely, but that's diagnostic, not actionable — it never
changes the deployed config. Morris `σ` already flags interactions cheaply, and the
confirmation run detects when the additive model breaks. Sobol would also need a
surrogate over a mixed continuous/categorical space, where a weak fit gives
confidently-wrong indices. Not worth it.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the background and tuning hypotheses.

## License

`llama-optimize` is released under the [MIT License](LICENSE). The bundled
[`robust`](https://github.com/bigattichouse/robust) DOE suite (the `robust/`
submodule) is dedicated to the public domain under CC0-1.0 — so the whole thing is
free to use, modify, and redistribute.
