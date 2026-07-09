# llamatune

Find good `llama.cpp` command-line parameters for a given GGUF model **on your
machine**, automatically, using a [Taguchi orthogonal-array](taguchi/) sweep
over `llama-bench`. The Taguchi engine (and its Morris/Sobol siblings) comes from
the [`robust`](https://github.com/bigattichouse/robust) DOE suite, vendored here as
a git submodule.

You point it at a model; it figures out the hardware, runs a small, statistically
designed set of benchmarks, and hands you paste-ready `llama-server` commands for
the **fastest**, the **longest-context**, and the **best-balanced** configuration —
plus the full speed-vs-context Pareto frontier.

```bash
# plan only — prints the experiment matrix and commands, uses NO GPU
python3 llamatuner.py /path/to/model.gguf

# actually run the sweep (uses the GPU)
python3 llamatuner.py /path/to/model.gguf --run

# bigger, finer sweep
python3 llamatuner.py /path/to/model.gguf --run --array L125
```

---

## Why Taguchi instead of a full sweep?

The knobs that matter for llama.cpp throughput interact, and testing every
combination explodes fast. With 5 factors at 5 levels each, a full factorial is
`5^5 = 3125` benchmark runs. A Taguchi **L25** orthogonal array estimates every
factor's main effect in **25 runs** — a >99% reduction — while keeping the levels
balanced so each factor's effect can be read independently.

We use the vendored `robust`/`taguchi` library via its Python binding to generate
the runs and to compute main effects. The `robust` suite also ships `morris` and
`sobol` binaries for the wider screening funnel (see Roadmap).

---

## What it tunes

Five factors, five levels each (auto-scaled to your hardware and model):

| Factor        | llama-bench flag | Levels (example, Qwen3.6-27B on MI50) | Notes |
|---------------|------------------|----------------------------------------|-------|
| GPU layers    | `-ngl`           | `0, 16, 32, 48, 64`                    | biggest lever; top = model's real layer count |
| Context depth | `-d` (n-depth)   | 5 levels `0..min(native ctx, 65536)`   | KV pre-fill; the speed-vs-context axis, adaptive to the model's native context |
| CPU threads   | `-t`             | `4, 6, 8, 12, 16`                      | auto-derived around the physical-core count |
| KV cache type | `-ctk`/`-ctv`    | `f16, q8_0, q5_1, q4_1, q4_0`          | quantizing the KV cache buys context |
| Micro-batch   | `-ub`            | `128, 256, 512, 1024, 2048`            | prefill/decode balance |

**Fixed** (not swept): `-fa 1` (flash-attention on — a near-certain win on gfx906
and a *precondition* for quantized KV cache), `-mmp 1` (mmap on), `-b 2048` (batch
fixed to avoid invalid `batch < ubatch` combinations).

L25 has 6 columns; using 5 factors leaves **one spare column** as an error/variance
estimate that flags when the additive main-effects model is breaking down.

---

## What it measures

For each config, `llama-bench` reports two throughput numbers, both captured:

- **`tg_tps`** — token-generation t/s (decode speed; what you feel interactively).
  This is what the optimizer maximizes.
- **`pp_tps`** — prompt-processing t/s (prefill speed; matters for long-context/RAG).
  Reported alongside.

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
- **VRAM** — via `rocm-smi` (best-effort; informational).
- **Model layer count** — a minimal, dependency-free GGUF metadata reader parses
  `<arch>.block_count` from the header (no tensors loaded), so `-ngl`'s top level is
  the model's real layer count.

---

## Output

```
### FASTEST (max t/s)
  tg=… t/s  pp=… t/s  depth=…  ngl=…  t=…  kv=…  ub=…
  suggested llama-server command:
    ./llama-server -m model.gguf -ngl … -t … -c … -ctk … -ctv … -ub … -b 2048 -fa 1

### BALANCED (max t/s with context >= 16384)
  …

### LONGEST CONTEXT
  …

### Pareto frontier (context vs t/s)
  depth=  4096  tg=…  ngl=…  kv=…  ub=…
  depth= 16384  tg=…  …
  …

### Taguchi main effects (tg t/s, higher = better)
  <per-factor level means + which factors dominate>
  Predicted-optimal levels: {…}
```

Full per-run data is written to `results.csv`.

### Reading the results correctly

- **Trust the Pareto frontier and the raw `results.csv`** for the actual
  recommendation — those read measured numbers directly.
- **Use the main-effects table to rank which knobs matter**, not as gospel for the
  single best config. Because OOM is scored as `0 t/s` and is driven by an
  *interaction* (the `ngl × n_depth × kv_type` memory budget), the additive
  main-effects model can misattribute an interaction to a main effect. This is why
  the recommendation is driven off the Pareto, and why we keep the spare column for
  an error estimate.

---

## Recommended workflow (staged)

1. **Screen** with **L25** (25 runs, ~30–40 min) to see which knobs dominate and get
   a candidate optimum.
2. **Refine** the 2–3 dominant factors with **L125** and map the context Pareto finely.
3. **Confirm** the predicted-optimal config with a direct `llama-server` run at your
   real context size (standard Taguchi discipline — verifies additivity held).

---

## Requirements

- `llama.cpp` built with ROCm/HIP for your GPU, at
  `../llama.cpp/build/bin/llama-bench` (override with `--llama-bench`).
- The `robust`/`taguchi` submodule, checked out and built:
  ```bash
  git submodule update --init          # fetch the robust DOE suite
  make -C taguchi                      # build libtaguchi.so (pure C, no GPU)
  ```
  `llamatuner` locates the Python binding automatically by searching the submodule.
- Python 3.10+ (uses `X | None` type syntax), standard library only.

---

## CLI reference

```
python3 llamatuner.py MODEL.gguf [options]

  --run              actually execute the sweep (default: plan/dry-run, no GPU)
  --array L25|L125   Taguchi array (default: L25)
  --ctx-floor N      minimum usable context for the BALANCED pick (default: 16384)
  --probe-ctx        after the sweep, binary-search the largest context that
                     loads for the fastest config (needs --run)
  --selftest         run offline logic checks and exit (no GPU, no model)
  --reps N           llama-bench repetitions per config (default: 3)
  --n-prompt N       prompt tokens per measurement (default: 512)
  --n-gen N          generated tokens per measurement (default: 128)
  --max-depth N      cap the n_depth factor levels (memory/time budget)
  --no-mtp           don't add draft-mtp flags to the server command
  --spec-draft-n-max N  MTP draft tokens for the server command (default: 2)
  --llama-bench PATH path to the llama-bench binary
  --timeout SECS     per-run timeout (default: 1200)
  --results PATH     results CSV output (default: results.csv)
```

> **Note on run time.** Deep-context configs at low `-ngl` prefill their KV cache
> on the CPU, which is slow (tens of seconds to minutes per run on a big model).
> That cost is inherent to measuring throughput *at* context. Use `--reps`,
> `--n-prompt/--n-gen`, and `--max-depth` to trade accuracy/coverage for speed on
> a first pass.

Run `python3 llamatuner.py --selftest` to verify the JSON parser, OOM detection,
factor-level generation, MoE detection, and Pareto logic without a GPU or model.

## Advanced: extra factors and environment sweeps

Beyond the default five factors, any llama-bench parameter in the `BENCH_FLAG`
map can be swept with `--factor`, and any environment variable with `--env`
(each becomes an orthogonal factor in the design). Both are opt-in, so pick an
array that fits the new factor count (`--array`, or `auto`).

```bash
# sweep KV offload location and CPU polling alongside a focused ngl/kv sweep
python3 llamatuner.py model.gguf --run --array auto \
  --factor ngl=56,60,64 --factor kv_type=f16,q4_0 \
  --factor nkvo=0,1 --factor poll=0,50

# gfx906 / ROCm environment tuning (the "10-30%" knobs) as sweepable factors
python3 llamatuner.py model.gguf --run --array auto \
  --env GGML_CUDA_FORCE_MMQ=0,1 --env GGML_CUDA_FORCE_CUBLAS=0,1
```

Env-var factors are applied to each benchmark process (not the command line),
and the winning values are prepended to the recommended `llama-server` command
as an env prefix. Sweepable llama-bench factors currently include: `ngl`,
`n_depth`, `threads`, `kv_type`, `ubatch`, `ncmoe`, `batch`, `nkvo`, `poll`.

**Note:** MTP / speculative decoding is *not* reachable from llama-bench (it has
no draft/spec support), so it can't be swept here — see the server-benchmark
roadmap item.

---

## Implemented

- **MoE awareness** — detects `<arch>.expert_count` in GGUF metadata; if the model is
  MoE, promotes `-ncmoe` (CPU-offload of experts) to a swept factor (the biggest
  RAM/VRAM lever on MoE). If dense, keeps the spare L25 column for error estimation.
- **Max-context probe** (`--probe-ctx`) — after the sweep, binary-searches the largest
  context that loads for the fastest config, capped at the model's native context.
- **Offline self-test** (`--selftest`) — verifies the core logic without a GPU.
- **MTP awareness** — detects a NextN/multi-token-prediction head (`<arch>.nextn_predict_layers`,
  present in e.g. Unsloth Dynamic quants) and appends `--spec-type draft-mtp
  --spec-draft-n-max N` to the emitted `llama-server` command. **Caveat:** the swept
  `tg` numbers do *not* include the MTP speedup — `llama-bench` can't do speculative
  decoding, so MTP is an additional multiplier on top of the measured throughput.
  See Roadmap for measuring it directly. Toggle with `--no-mtp` / `--spec-draft-n-max`.

## Roadmap / ideas

- **Server-based benchmark for MTP** — `llama-bench` can't do speculative decoding,
  so to actually *measure and optimize* MTP we'd spin up `llama-server` with each
  config, drive real generation through the API, and sweep `--spec-draft-n-max`.
  This is the only way to capture the acceptance-rate speedup rather than just
  emitting the flags.
- **Flash-attn as an outer block** — run the array twice (`-fa 0` / `-fa 1`) to
  quantify flash-attention's effect directly, mindful that quantized KV requires
  `-fa 1`.
- **Morris/Sobol pre-screen** — the vendored `robust` suite ships `morris` and `sobol`
  binaries. Use Morris to screen which factors matter (μ\* importance, σ interaction
  flag) before committing to the full Taguchi bench, and Sobol for variance/interaction
  attribution when L25's error estimate says the additive model is shaky.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the background and tuning hypotheses.
