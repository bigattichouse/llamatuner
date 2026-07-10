# Design notes & background

Background and prior knowledge behind `llamatune`. The [README](../README.md) covers
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
thing worth benchmarking per-machine rather than guessing — which is what `llamatune`
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
