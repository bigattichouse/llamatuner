#!/usr/bin/env python3
"""
llama-optimize - find good llama.cpp command-line parameters for a given GGUF model
on this machine, using a Taguchi orthogonal-array sweep over llama-bench.

Usage:
    llama-optimize.py MODEL.gguf                 # plan only: print the matrix + commands
    llama-optimize.py MODEL.gguf --run           # actually run the benchmark sweep
    llama-optimize.py MODEL.gguf --run --array L125   # bigger sweep

The tool auto-detects CPU cores, VRAM, and the model's layer count to choose
sensible factor levels, runs the sweep (one llama-bench invocation per Taguchi
run), then reports the fastest / longest-context / balanced configurations as
ready-to-paste llama-server command lines.

Nothing touches the GPU unless --run is given.
"""

import argparse
import contextlib
import csv
import io
import json
import os
import random
import re
import shlex
import shutil
import statistics
import struct
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE = PROJECT_ROOT.parent
# The Taguchi/Morris/Sobol suite is vendored as the `robust` git submodule at
# ./robust (its internal layout is nested, so locate the python binding by
# search rather than a fixed path). Older checkouts used ./taguchi; fall back to
# it so the tool keeps working before a `git submodule sync`.
SUBMODULE_DIR = next((p for p in (PROJECT_ROOT / "robust", PROJECT_ROOT / "taguchi")
                      if p.exists()), PROJECT_ROOT / "robust")


def resolve_binary(name: str, explicit: Path | None, hint: Path | None) -> Path:
    """Locate a llama.cpp binary. Search order: explicit path, --llama-cpp hint
    (root / build/bin / bin), $LLAMA_CPP, $PATH, then the sibling-workspace
    default. Returns the first existing match, else a best-guess path (whose
    non-existence is reported later)."""
    if explicit is not None:
        return explicit
    roots = []
    if hint is not None:
        roots.append(hint)
    env = os.environ.get("LLAMA_CPP")
    if env:
        roots.append(Path(env))
    cands: list[Path] = []
    for r in roots:
        cands += [r / name, r / "build" / "bin" / name, r / "bin" / name]
    on_path = shutil.which(name)
    if on_path:
        cands.append(Path(on_path))
    default = WORKSPACE / "llama.cpp" / "build" / "bin" / name
    cands.append(default)
    for c in cands:
        if c.exists():
            return c
    return default  # doesn't exist; caller validates and errors clearly


def preflight(binary: Path, timeout: int = 60):
    """Confirm the binary actually runs (not just exists) — catches a wrong build
    or missing GPU libraries. Returns (ok, reason)."""
    try:
        out = subprocess.run([str(binary), "--help"], capture_output=True,
                             text=True, timeout=timeout)
    except FileNotFoundError:
        return False, "not found / not executable"
    except subprocess.TimeoutExpired:
        return False, "hung running --help"
    except OSError as e:
        return False, f"failed to execute ({e})"
    if out.returncode != 0:
        tail = (out.stderr or out.stdout or "").strip().splitlines()[-3:]
        return False, "exited nonzero — " + " ".join(tail)
    return True, ""


def find_taguchi_binding() -> Path:
    """Return the dir to add to sys.path so `import taguchi` works."""
    hits = sorted(
        SUBMODULE_DIR.glob("**/bindings/python/taguchi/__init__.py"),
        key=lambda p: len(p.parts),
    )
    if not hits:
        raise SystemExit(
            f"taguchi python binding not found under {SUBMODULE_DIR}.\n"
            f"Run:  git submodule update --init  &&  make -C {SUBMODULE_DIR}\n"
            "(builds libtaguchi.so + the morris binary; see the README's Setup section)."
        )
    return hits[0].parents[1]  # .../bindings/python

# Fixed parameters (see design notes): flash-attn is a precondition for KV-quant
# and a near-certain win on gfx906; mmap on is the sane default; batch fixed to
# avoid invalid batch<ubatch combinations.
FIXED_FA = 1
FIXED_MMAP = 1
FIXED_BATCH = 2048

# llama-bench measurement shape per config
BENCH_N_PROMPT = 512
BENCH_N_GEN = 128
BENCH_REPS = 3

# Workload profiles. Each sets the representative request shape (prompt + gen
# tokens) that the sweep measures at and scores by, a usable-context floor, and
# the driver. Objective = effective throughput for that request shape:
#   (P + G) / (P/pp_tps + G/tg_tps)   -- combines prefill and decode as the
# workload actually experiences them. "multi" needs the server driver (real
# concurrency), which llama-bench cannot do.
PROFILES = {
    "single": {"n_prompt": 512,  "n_gen": 256, "ctx_floor": 8192,  "driver": "bench"},
    "agents": {"n_prompt": 8192, "n_gen": 256, "ctx_floor": 32768, "driver": "bench"},
    "multi":  {"n_prompt": 1024, "n_gen": 256, "ctx_floor": 8192,  "driver": "server",
               "parallel": 4},
}

# Use-cases are high-level "runbooks": a friendly name that expands into a bundle
# of lower-level flags (driver + request profile + concurrency). Precedence is
# built-in defaults < use-case < explicit flags, so `--use-case agents --parallel 2`
# keeps the agents runbook but forces 2 streams. Each entry maps to a base profile
# (request shape / objective) plus the driver and concurrency that fit the workload.
USE_CASES = {
    # name          driver     profile    parallel   what it's for
    "app":        {"driver": "bench",  "profile": "single", "parallel": 1},
    #   general llama-based app / embedded llama.cpp — raw single-stream throughput
    "single":     {"driver": "server", "profile": "single", "parallel": 1},
    #   llama-server for one user/worker — real generation incl. MTP, one slot
    "agents":     {"driver": "server", "profile": "agents", "parallel": 4},
    #   several autonomous agents — long tool-use prompts, concurrent slots
    "multi-user": {"driver": "server", "profile": "multi",  "parallel": 8},
    #   many concurrent chat users — short prompts, high concurrency
}


# ---------------------------------------------------------------------------
# Hardware / model auto-detection
# ---------------------------------------------------------------------------
def detect_logical_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def detect_physical_cores() -> int:
    """Count unique (physical id, core id) pairs from /proc/cpuinfo."""
    try:
        pairs = set()
        phys = core = None
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif line.strip() == "":
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
        if pairs:
            return len(pairs)
    except OSError:
        pass
    # Fallback: assume 2 threads/core
    return max(1, detect_logical_cores() // 2)


def detect_numa_nodes() -> int:
    """Number of NUMA nodes (1 when undetectable / not Linux)."""
    try:
        return max(1, len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))))
    except OSError:
        return 1


def detect_vram_mib() -> int | None:
    """Best-effort total VRAM in MiB. Tries AMD (rocm-smi) then NVIDIA
    (nvidia-smi); None if neither is available."""
    # AMD / ROCm
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            data = json.loads(out.stdout)
            for card in data.values():
                for k, v in card.items():
                    if "vram" in k.lower() and "total" in k.lower():
                        return int(v) // (1024 * 1024)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    # NVIDIA / CUDA
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            return int(float(out.stdout.strip().splitlines()[0]))  # already MiB
    except (OSError, ValueError):
        pass
    return None


def vram_used_mib() -> int | None:
    """Currently-used VRAM in MiB (AMD then NVIDIA); None if unavailable."""
    try:
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            for card in json.loads(out.stdout).values():
                for k, v in card.items():
                    if "vram" in k.lower() and "used" in k.lower():
                        return int(v) // (1024 * 1024)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return int(float(out.stdout.strip().splitlines()[0]))
    except (OSError, ValueError):
        pass
    return None


def gpu_temp_c() -> float | None:
    """Best-effort GPU temperature in °C (AMD rocm-smi then NVIDIA nvidia-smi);
    None if no sensor is readable. Returns the hottest sensor reported."""
    try:
        out = subprocess.run(["rocm-smi", "--showtemp", "--json"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            data = json.loads(out.stdout)
            edge, any_t = [], []
            for card in data.values():
                for k, v in card.items():
                    if "temp" not in k.lower():
                        continue
                    try:
                        t = float(v)
                    except (TypeError, ValueError):
                        continue
                    any_t.append(t)
                    if "edge" in k.lower() or "junction" in k.lower():
                        edge.append(t)
            temps = edge or any_t
            if temps:
                return max(temps)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return max(float(x) for x in out.stdout.strip().splitlines())
    except (OSError, ValueError):
        pass
    return None


# Thermal "wait and watch": between runs, block until the GPU falls back to
# within THERMAL_BAND_C of the idle baseline so each config is measured from a
# comparable thermal state (the MI50 throttles ~1.8× cool-vs-hot, a swing bigger
# than most factor effects — see docs/DESIGN.md). Capped so it can never hang.
THERMAL_BAND_C = 5.0
THERMAL_CAP_S = 120.0


def wait_until_cool(baseline_c: float | None, band: float = THERMAL_BAND_C,
                    cap_s: float = THERMAL_CAP_S, poll_s: float = 3.0) -> None:
    """Watch GPU temperature; return once it is within `band` °C of the idle
    `baseline_c`, or it plateaus (cooling stalls), or `cap_s` elapses. No-op
    when there's no baseline or no readable sensor. A *rising* temperature is
    not a plateau: right after a run the sensor often keeps climbing for a few
    seconds (heat soak), and bailing then would exit at the hottest moment."""
    if baseline_c is None:
        return
    target, t0, prev = baseline_c + band, time.time(), None
    tty = sys.stdout.isatty()
    while time.time() - t0 < cap_s:
        t = gpu_temp_c()
        if t is None:
            break
        settled = t <= target or (prev is not None and 0 <= prev - t < 0.5)
        if tty:
            print(f"\r  thermal: {t:>3.0f}°C  (settle ≤{target:.0f}°C)"
                  f"{'  ok' if settled else '  cooling…'}   ", end="", flush=True)
        if settled:
            break
        prev = t
        time.sleep(poll_s)
    if tty:
        print()


class VRAMSampler:
    """Polls used VRAM in a background thread, tracking the peak during a run.
    (rocm-smi is slow, so polling is coarse; captures the config's footprint.)"""

    def __init__(self, interval: float = 1.0):
        self.peak = 0
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        def poll():
            while not self._stop.wait(self.interval):
                v = vram_used_mib()
                if v:
                    self.peak = max(self.peak, v)
        # one immediate sample so short runs still get a reading
        v = vram_used_mib()
        if v:
            self.peak = v
        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1)
        return False


# --- Minimal GGUF metadata reader (header only, no tensor data) -------------
_GGUF_MAGIC = b"GGUF"
# value type ids
_GT_UINT8, _GT_INT8, _GT_UINT16, _GT_INT16, _GT_UINT32, _GT_INT32 = 0, 1, 2, 3, 4, 5
_GT_FLOAT32, _GT_BOOL, _GT_STRING, _GT_ARRAY = 6, 7, 8, 9
_GT_UINT64, _GT_INT64, _GT_FLOAT64 = 10, 11, 12
_GT_FMT = {
    _GT_UINT8: "<B", _GT_INT8: "<b", _GT_UINT16: "<H", _GT_INT16: "<h",
    _GT_UINT32: "<I", _GT_INT32: "<i", _GT_FLOAT32: "<f", _GT_BOOL: "<?",
    _GT_UINT64: "<Q", _GT_INT64: "<q", _GT_FLOAT64: "<d",
}


class _GGUFReader:
    def __init__(self, f):
        self.f = f

    def _read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("unexpected EOF reading GGUF header")
        return b

    def u32(self):
        return struct.unpack("<I", self._read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self._read(8))[0]

    def string(self):
        n = self.u64()
        return self._read(n).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype in _GT_FMT:
            fmt = _GT_FMT[vtype]
            return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]
        if vtype == _GT_STRING:
            return self.string()
        if vtype == _GT_ARRAY:
            elem_type = self.u32()
            count = self.u64()
            return [self.value(elem_type) for _ in range(count)]
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_metadata(path: Path) -> dict:
    """Parse GGUF metadata key/values only. Returns {} on any failure."""
    try:
        with open(path, "rb") as f:
            r = _GGUFReader(f)
            if r._read(4) != _GGUF_MAGIC:
                return {}
            version = r.u32()
            if version < 2:
                return {}
            _tensor_count = r.u64()
            kv_count = r.u64()
            meta = {}
            for _ in range(kv_count):
                key = r.string()
                vtype = r.u32()
                meta[key] = r.value(vtype)
            return meta
    except (OSError, EOFError, ValueError, struct.error):
        return {}


def _meta_int(meta: dict, suffix: str) -> int | None:
    for k, v in meta.items():
        if k.endswith(suffix):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    return None


def model_block_count(meta: dict) -> int | None:
    return _meta_int(meta, ".block_count")


def model_expert_count(meta: dict) -> int:
    """Number of MoE experts; 0 (or missing) means a dense model."""
    return _meta_int(meta, ".expert_count") or 0


def model_context_length(meta: dict) -> int | None:
    """Native max context the model was trained for."""
    return _meta_int(meta, ".context_length")


def model_nextn_layers(meta: dict) -> int:
    """MTP (multi-token-prediction / NextN) head layers; 0 means no MTP head.
    Present in e.g. Unsloth Dynamic quants that support draft-mtp speculative
    decoding in llama-server."""
    return _meta_int(meta, ".nextn_predict_layers") or 0


# ---------------------------------------------------------------------------
# Factor-level generation
# ---------------------------------------------------------------------------
def five_levels_span(lo: int, hi: int) -> list[int]:
    """Five roughly evenly spaced distinct integer levels in [lo, hi]."""
    if hi <= lo:
        return [lo]
    raw = [round(lo + (hi - lo) * i / 4) for i in range(5)]
    out = sorted(set(raw))
    # pad toward hi if collisions removed levels
    i = lo
    while len(out) < 5 and i <= hi:
        if i not in out:
            out.append(i)
        i += 1
    return sorted(out)[:5]


def thread_levels(phys: int, logical: int) -> list[int]:
    cand = {
        max(1, phys // 2),
        max(1, phys * 3 // 4),
        phys,
        (phys + logical + 1) // 2,
        logical,
    }
    levels = sorted(c for c in cand if c >= 1)
    # ensure exactly 5 distinct levels where possible
    n = 1
    while len(levels) < 5 and phys + n <= logical:
        levels = sorted(set(levels) | {phys + n})
        n += 1
    return levels[:5] if len(levels) >= 5 else levels


def ngl_levels(n_layers: int | None) -> list[int]:
    top = n_layers if n_layers else 99
    # Always include 0 (pure CPU) and the top (all layers). 99 = "all" is safe
    # since llama.cpp clamps to the real layer count.
    mids = five_levels_span(0, top)
    lv = sorted(set([0] + mids + [top]))
    # trim to 5, keeping endpoints
    if len(lv) > 5:
        keep = {0, top}
        inner = [x for x in lv if x not in keep]
        step = max(1, len(inner) // 3)
        lv = sorted(keep | set(inner[::step]))[:5]
    return lv


# Don't probe deeper than this by default, even if the model's native context is
# larger — deep contexts on CPU (low -ngl) prefill very slowly and burn memory.
DEFAULT_MAX_DEPTH = 65536
DEFAULT_KV_LEVELS = ["f16", "q8_0", "q5_1", "q4_1", "q4_0"]
DEFAULT_UBATCH_LEVELS = [128, 256, 512, 1024, 2048]

# KV cache types ordered best-quality -> lossiest. q8_0 is near-lossless; below it
# quality degrades and errors compound over context. --min-kv floors this.
KV_QUALITY = ["f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"]


def kv_at_or_above(levels: list, floor: str) -> list:
    """Keep only KV types at least as high-quality as `floor` (order in
    KV_QUALITY). floor 'any'/'none'/'' disables the filter."""
    if not floor or floor.lower() in ("any", "none"):
        return list(levels)
    fi = KV_QUALITY.index(floor) if floor in KV_QUALITY else len(KV_QUALITY) - 1
    kept = [l for l in levels
            if l not in KV_QUALITY or KV_QUALITY.index(l) <= fi]
    return kept or [floor]  # never empty


def depth_levels(n_ctx_train: int | None, override_max: int | None = None,
                 floor: int = 0) -> list[int]:
    """Five n_depth levels spanning floor..min(native ctx, cap). Adaptive so we
    never test beyond the model's native context.  The floor raises the minimum
    depth so the sweep only tests context sizes the user actually needs (--ctx-floor)."""
    top = min(n_ctx_train or DEFAULT_MAX_DEPTH, DEFAULT_MAX_DEPTH)
    if override_max is not None:
        top = min(top, override_max)
    if floor >= top:
        return [top]
    return five_levels_span(floor, top)


def ncmoe_levels(n_layers: int | None) -> list[int]:
    """Levels for -ncmoe (how many layers keep their MoE experts on CPU)."""
    return five_levels_span(0, n_layers if n_layers else 64)


@dataclass
class Config:
    model: Path
    llama_bench: Path
    array: str
    ctx_floor: int
    llama_server: Path = field(
        default_factory=lambda: WORKSPACE / "llama.cpp" / "build" / "bin" / "llama-server")
    reps: int = BENCH_REPS
    n_prompt: int = BENCH_N_PROMPT
    n_gen: int = BENCH_N_GEN
    max_depth: int | None = None  # cap n_depth levels (memory/time budget)
    emit_mtp: bool = True         # add draft-mtp flags to server cmd if supported
    ngram: bool = False           # enable ngram self-speculative decoding sweep
    ngram_type: str | None = None # pin the ngram variant (tuning stage / --ngram-type)
    ngram_keep: int = 2           # variants carried from screen into tuning (top-K)
    spec_draft_n_max: int = 2     # --spec-draft-n-max for MTP
    profile: str = "single"       # workload profile (see PROFILES)
    driver: str = "bench"         # "bench" (llama-bench) or "server" (llama-server)
    parallel: int = 1             # concurrent request streams (server driver)
    server_start_timeout: int = 180  # max seconds to wait for llama-server to load
    measure_vram: bool = False       # sample peak VRAM used during each run
    score: str = "tg"             # objective: "tg" (decode only) or "eff" (blend pp+tg)
    factors: dict = field(default_factory=dict)
    hw: dict = field(default_factory=dict)
    env_factor_names: set = field(default_factory=set)  # factors that set env vars


def effective_tps(n_prompt: int, n_gen: int, pp: float, tg: float) -> float:
    """Throughput for a representative request of n_prompt prompt + n_gen gen
    tokens: total tokens / (prefill time + decode time)."""
    if pp <= 0 or tg <= 0:
        return 0.0
    return (n_prompt + n_gen) / (n_prompt / pp + n_gen / tg)


def objective_tps(cfg: Config, pp: float, tg: float) -> float:
    """The score a run contributes to fits/stats/picks (stored as eff_tps).
    Default (--score tg): pure generation speed — pp is still measured and
    reported, but a huge prefill number can't make a slow generator outrank a
    fast one. --score eff blends both into effective request throughput."""
    if cfg.score == "eff":
        return effective_tps(cfg.n_prompt, cfg.n_gen, pp, tg)
    return tg if tg > 0 else 0.0


def build_factors(cfg: Config):
    reg_errors = validate_factor_registry()
    assert not reg_errors, "FACTORS registry invalid: " + "; ".join(reg_errors)
    phys = cfg.hw["phys"]
    logical = cfg.hw["logical"]
    n_layers = cfg.hw.get("n_layers")
    depths = depth_levels(cfg.hw.get("n_ctx_train"), cfg.max_depth, floor=cfg.ctx_floor)
    factors = {
        "ngl": [str(x) for x in ngl_levels(n_layers)],
        "n_depth": [str(x) for x in depths],
        "threads": [str(x) for x in thread_levels(phys, logical)],
        "kv_type": list(DEFAULT_KV_LEVELS),
        "ubatch": [str(x) for x in DEFAULT_UBATCH_LEVELS],
        # KV offload (-nkvo) is the VRAM-vs-bandwidth lever: keeping the KV
        # cache in system RAM frees VRAM for layers at a PCIe cost that only a
        # measurement can price on a given box. (fa stays fixed: flash-attn is
        # a precondition for quantized KV, so sweeping it would just fail every
        # fa=0 × KV-quant row — sweep it via --factor fa=0,1 --min-kv f16.)
        "nkvo": ["0", "1"],
        # An L125 fits 31 factors at the same 125 runs, so the remaining clean
        # knobs ride along free. batch levels start at the largest ubatch level
        # so the -b >= -ub clamp never fires (clamping would alias low batch
        # levels with high ubatch ones and muddy both estimates).
        "poll": ["0", "50", "100"],
        "batch": ["2048", "4096", "8192"],
    }
    if cfg.driver == "server":
        # decode threads (-t) and prefill threads (-tb) can want different
        # counts; only llama-server exposes the split
        factors["threads_batch"] = [str(x) for x in thread_levels(phys, logical)]
    # only sweep NUMA policy on a machine that actually has multiple nodes
    # (on a single node it's an inert column)
    if cfg.hw.get("numa_nodes", 1) > 1:
        factors["numa"] = ["distribute", "isolate"]
    # For MoE models, expert CPU-offload (-ncmoe) is the biggest RAM/VRAM lever,
    # so promote it to a swept factor. For dense models the equivalent lever is
    # tensor placement (-ot): keeping FFN tensors on CPU at full -ngl often
    # beats dropping whole layers (attn_cpu is left out — attention on CPU
    # kills decode; exps_cpu is inert without experts).
    if cfg.hw.get("n_experts", 0) > 0:
        factors["ncmoe"] = [str(x) for x in ncmoe_levels(n_layers)]
    else:
        factors["ot"] = ["none", "ffn_up_cpu", "ffn_cpu"]
    # A model with an MTP/NextN head on the server driver: sweep the whole
    # speculative-decoding surface — on/off, draft lengths, and acceptance
    # thresholds — so the report MEASURES what MTP buys instead of assuming it.
    # Bench can't speculate (server-only knobs), and --no-mtp opts out.
    if cfg.driver == "server" and cfg.emit_mtp and cfg.hw.get("n_nextn", 0) > 0:
        factors["mtp"] = ["1", "0"]
        factors["spec_n_max"] = ["1", "2", "3", "4", "6"]
        factors["spec_n_min"] = ["1", "2"]
        factors["spec_p_min"] = ["0.0", "0.25", "0.5", "0.75", "0.9"]
        factors["spec_p_split"] = ["0.1", "0.3", "0.5"]
    # ngram self-speculative decoding (server only). The --spec-type variant is a
    # gate; its tuning knobs are CONDITIONAL (active_when — docs/CONDITIONAL-FACTORS.md)
    # and must not share a flat array with the gate (that inflates the design and
    # dilutes their effects). So build_factors emits only the GATE here — the
    # Stage-0 "screen" that measures each variant at its default knobs. A variant's
    # knobs enter the design only once the gate is pinned to it (a tuning stage,
    # cfg.ngram_type — driven by run_ngram_stages or --ngram-type). ngram needs no
    # draft model and complements MTP when both are active (--spec-type values
    # combine). spec_n_max is NOT added: --spec-draft-n-max is a draft-model/MTP
    # knob with no effect on ngram.
    if cfg.driver == "server" and cfg.ngram:
        if cfg.ngram_type:                         # tuning stage: gate pinned
            factors["ngram"] = [cfg.ngram_type]
            factors.update(ngram_child_levels(cfg.ngram_type))
        else:                                      # screen stage: variants only
            factors["ngram"] = ["none", "ngram-simple", "ngram-mod",
                                "ngram-map-k", "ngram-map-k4v"]
    return factors


# Orthogonal-array capacity: name -> number of columns (factors) it holds.
_ARRAY_TABLES = {
    2: [("L4", 3), ("L8", 7), ("L16", 15), ("L32", 31), ("L64", 63), ("L128", 127)],
    3: [("L9", 4), ("L27", 13), ("L81", 40), ("L243", 121)],
    5: [("L25", 6), ("L125", 31), ("L625", 156)],
}


def choose_array(factors: dict) -> str | None:
    """Pick the smallest orthogonal array that fits the factor set, based on the
    factors' level counts. Returns an array name, or None to let the binding
    auto-select. Fixes the binding auto-selecting a 5-level array for 3-level
    factors.

    Only factors that actually vary (>1 level) count: a factor refinement has
    already pinned to a single level is a constant and carries no information
    for an orthogonal array. Sizing on all factors instead would draw a full
    25-run L25 to sweep a lone 5-level factor left among four pinned ones. With
    <=1 varying factor an array is meaningless (return None ⇒ direct sweep)."""
    counts = [len(v) for v in factors.values() if len(v) > 1]
    if len(counts) <= 1:
        return None
    nf, mx = len(counts), max(counts)
    # Mixed level counts ride on the array of the largest base (a 2-level factor
    # maps onto a 5-level column with a modulo imbalance the level means absorb).
    # The binding only ships pure 2/3/5-level arrays — no mixed L18/L36/L50.
    base = 2 if mx <= 2 else 3 if mx == 3 else 5 if mx <= 5 else None
    if base is None:
        return None
    for name, cap in _ARRAY_TABLES[base]:
        if cap >= nf:
            return name
    return None


# ---------------------------------------------------------------------------
# Taguchi run generation (via the python binding)
# ---------------------------------------------------------------------------
def generate_runs(factors: dict, array: str | None):
    # Split settled (single-level) factors out of the design. They are constants:
    # feeding them to the orthogonal array adds no information but inflates the
    # run count (a lone 5-level factor among four pinned ones would draw a 25-run
    # L25 — 5 real configs each replicated 5×). Build over the ACTIVE factors and
    # re-attach the constants to every generated run so downstream command
    # builders still see a complete factor set.
    active = {k: v for k, v in factors.items() if len(v) > 1}
    const = {k: v[0] for k, v in factors.items() if len(v) <= 1}

    if len(active) <= 1:
        # 0 or 1 varying factor: an orthogonal array is degenerate — enumerate
        # the level(s) directly (a one-way sweep), no wasted replicate rows and
        # no dependency on the array binding.
        if not active:
            return None, [{"run_id": 1, "factors": dict(const)}]
        (name, levels), = active.items()
        return None, [{"run_id": i + 1, "factors": {**const, name: lvl}}
                      for i, lvl in enumerate(levels)]

    sys.path.insert(0, str(find_taguchi_binding()))
    from taguchi import Experiment  # noqa: E402

    if array and array.lower() == "auto":
        array = None
    # The binding takes the array in the constructor; None => auto-select.
    exp = Experiment(array_type=array)
    for name, levels in active.items():
        exp.add_factor(name, levels)
    runs = exp.generate()
    for run in runs:                       # constants ride along on every row
        run["factors"].update(const)
    return exp, runs


# ---------------------------------------------------------------------------
# Staged designs for conditional factors — docs/CONDITIONAL-FACTORS.md.
# A flat orthogonal array cannot hold conditional (active_when) factors without
# inert columns (F1–F4). plan_stages decomposes a full factor set into a sequence
# of flat designs, each with every factor active in every row (invariant I1):
#   - one "screen" stage over the unconditional + gate factors (children excluded,
#     each variant measured at its default knobs), then
#   - one "tune:<gate>=<value>" stage per candidate gate value that has children,
#     pinning the gate to that value and sweeping exactly its children.
# The executor runs the screen, keeps the surviving gate values (top-K), and runs
# their tuning stages; the winner is the best MEASURED config across all stages.
# ---------------------------------------------------------------------------
def _active_when(name: str):
    return FACTORS.get(name, {}).get("active_when")


def plan_stages(factors: dict) -> list[dict]:
    """Ordered list of staged flat designs for `factors` (name → levels). Each
    stage is {name, factors, pin, gate, value}: `factors` is that stage's OA
    design, `pin` the gate constants that make its conditional children live. The
    screen stage has pin={} (only unconditional + gate factors). Tuning stages are
    ordered parent-gate-first so a nested gate is pinned only after the stage that
    selects it. Every factor in a stage is active in every row (I1)."""
    # children grouped by the (gate, value) that makes them live
    kids: dict = {}
    for n in factors:
        aw = _active_when(n)
        if aw:
            gate, live = aw
            for v in live:
                if gate in factors and v in factors[gate]:
                    kids.setdefault((gate, v), []).append(n)

    screen = {n: factors[n] for n in factors if not _active_when(n)}
    stages = [{"name": "screen", "factors": screen, "pin": {},
               "gate": None, "value": None}]

    def gate_depth(g: str) -> int:                 # nesting depth for ordering
        d, seen, aw = 0, set(), _active_when(g)
        while aw and aw[0] not in seen:
            seen.add(aw[0]); d += 1; aw = _active_when(aw[0])
        return d

    for gate, v in sorted(kids, key=lambda gv: (gate_depth(gv[0]), gv[0],
                                                factors[gv[0]].index(gv[1]))):
        fset = {gate: [v]}                          # gate pinned to one value
        for c in kids[(gate, v)]:
            fset[c] = factors[c]
        stages.append({"name": f"tune:{gate}={v}", "factors": fset,
                       "pin": {gate: v}, "gate": gate, "value": v})
    return stages


# ---------------------------------------------------------------------------
# Command building + execution
# ---------------------------------------------------------------------------
# Named -override-tensor patterns → real llama.cpp tensor regex. "none" emits
# nothing. These place whole tensor classes on CPU to free VRAM for more layers.
OT_PATTERNS = {
    "none": "",
    "ffn_cpu": r"\.ffn_(gate|up|down)\.weight=CPU",   # all FFN on CPU (dense)
    "ffn_up_cpu": r"\.ffn_up\.weight=CPU",
    "exps_cpu": r"\.ffn_.*_exps\.=CPU",               # MoE experts on CPU
    "attn_cpu": r"\.attn_.*=CPU",
}

# ngram --spec-type variants that share one {size-n, size-m, min-hits} knob
# structure (differing only in the flag's variant token). ngram-mod has its own
# {n-match, n-min, n-max}. See docs/ngram-design.md.
NGRAM_MAP_VARIANTS = frozenset({"ngram-simple", "ngram-map-k", "ngram-map-k4v"})

# Conditional-child level sets (levels bracket llama.cpp's defaults within the
# documented bounds). These enter the design only in a variant's tuning stage —
# see plan_stages / run_ngram_stages, docs/CONDITIONAL-FACTORS.md.
NGRAM_MAP_LEVELS = {                      # ngram-simple / map-k / map-k4v
    "ngram_size_n":   ["4", "8", "12", "16", "24"],   # default 12
    "ngram_size_m":   ["8", "16", "32", "48", "64"],  # default 48
    "ngram_min_hits": ["1", "2", "3", "5"],           # default 1
}
NGRAM_MOD_LEVELS = {                      # ngram-mod
    "ngram_mod_n_match": ["8", "16", "24", "32", "48"],   # default 24
    "ngram_mod_n_min":   ["16", "32", "48", "64", "96"],  # default 48
    "ngram_mod_n_max":   ["32", "48", "64", "96", "128"], # default 64
}


def ngram_child_levels(variant: str) -> dict:
    """The tuning-knob level sets for one ngram variant (empty for none/unknown)."""
    if variant in NGRAM_MAP_VARIANTS:
        return dict(NGRAM_MAP_LEVELS)
    if variant == "ngram-mod":
        return dict(NGRAM_MOD_LEVELS)
    return {}

# ---------------------------------------------------------------------------
# Unified knob registry — the one place to add a tunable. Each factor declares
# how it maps onto each driver.
#   bench/server : flag tuple for that driver, or None if unsupported there
#   kind         : "num"   integer, refined onto a finer grid between passes
#                  "float" real value, refined by keeping top levels
#                  "cat"   categorical, refined by keeping top levels
#                  "bool"  0/1; bench takes the value, server emits a bare flag
#   server_only  : only meaningful with the server driver
#   request      : request-time (n_depth) — not a server launch arg
#   translate    : map named level -> real value ("" ⇒ omit the flag)
#   active_when  : (gate_factor, {live values}) — a CONDITIONAL factor that only
#                  participates when factor `gate_factor` resolves to one of the
#                  live values (e.g. an ngram-mod knob is live only when the
#                  `ngram` gate == "ngram-mod"). Outside its live set the factor is
#                  inert: no flag is emitted (I2) and it is not scored (I3). See
#                  docs/CONDITIONAL-FACTORS.md.
#   flag_for     : gate_value -> flag tuple. For a conditional factor whose flag
#                  spelling depends on the live gate value (collapses several
#                  same-shaped knobs into one, e.g. ngram size-n across variants).
#                  Resolved at emission via the row's gate value; takes the place
#                  of a static `server`/`bench` tuple.
# ---------------------------------------------------------------------------
FACTORS = {
    # --- offload / placement ---
    "ngl":          {"bench": ("-ngl",), "server": ("-ngl",), "kind": "num"},
    "ncmoe":        {"bench": ("-ncmoe",), "server": ("-ncmoe",), "kind": "num"},
    "ot":           {"bench": ("-ot",), "server": ("-ot",), "kind": "cat",
                     "translate": OT_PATTERNS},
    "nkvo":         {"bench": ("-nkvo",), "server": ("-nkvo",), "kind": "bool"},
    # --- batching ---
    "batch":        {"bench": ("-b",), "server": ("-b",), "kind": "num"},
    "ubatch":       {"bench": ("-ub",), "server": ("-ub",), "kind": "num"},
    # --- KV cache ---
    "kv_type":      {"bench": ("-ctk", "-ctv"), "server": ("-ctk", "-ctv"), "kind": "cat"},
    # --- CPU / threads ---
    "threads":      {"bench": ("-t",), "server": ("-t",), "kind": "num"},
    "threads_batch": {"bench": None, "server": ("-tb",), "kind": "num", "server_only": True},
    "poll":         {"bench": ("--poll",), "server": ("--poll",), "kind": "num"},
    "numa":         {"bench": ("--numa",), "server": ("--numa",), "kind": "cat"},
    "cpu_mask":     {"bench": ("-C",), "server": ("-C",), "kind": "cat"},        # hex affinity mask
    "cpu_strict":   {"bench": ("--cpu-strict",), "server": ("--cpu-strict",), "kind": "cat"},  # 0/1
    "cpu_range":    {"bench": None, "server": ("-Cr",), "kind": "cat", "server_only": True},  # lo-hi
    # --- attention ---
    "fa":           {"bench": ("-fa",), "server": ("-fa",), "kind": "cat"},
    # --- context (request-time) ---
    "n_depth":      {"bench": ("-d",), "server": None, "kind": "num", "request": True},
    # --- speculative decoding / MTP (server only) ---
    "mtp":          {"bench": None, "server": ("--spec-type",), "kind": "cat", "server_only": True,
                     "translate": {"1": "draft-mtp", "0": ""}},   # on/off: "" omits the flag
    "spec_n_max":   {"bench": None, "server": ("--spec-draft-n-max",), "kind": "num", "server_only": True},
    "spec_n_min":   {"bench": None, "server": ("--spec-draft-n-min",), "kind": "num", "server_only": True},
    "spec_p_min":   {"bench": None, "server": ("--spec-draft-p-min",), "kind": "float", "server_only": True},
    "spec_p_split": {"bench": None, "server": ("--spec-draft-p-split",), "kind": "float", "server_only": True},
    "ngram":             {"bench": None, "server": ("--spec-type",), "kind": "cat", "server_only": True,
                          "translate": {"none": "", "ngram-simple": "ngram-simple", "ngram-mod": "ngram-mod",
                                        "ngram-map-k": "ngram-map-k", "ngram-map-k4v": "ngram-map-k4v"}},
    # ngram tuning knobs are CONDITIONAL on the --spec-type variant (active_when):
    # emitted and scored only in rows whose `ngram` gate selects the owning
    # variant. The map/simple variants share one {size-n, size-m, min-hits}
    # structure differing only in the flag's variant token, so they collapse to
    # one factor each via flag_for. ngram-mod carries its own {n-match, n-min,
    # n-max}. Defaults/bounds: llama.cpp common/common.h (see docs/ngram-design.md).
    # NOTE: --spec-draft-n-max (`spec_n_max`) is a draft-model/MTP knob and has NO
    # effect on ngram — it is deliberately not an ngram factor.
    "ngram_size_n":   {"bench": None, "kind": "num", "server_only": True,
                       "active_when": ("ngram", NGRAM_MAP_VARIANTS),
                       "flag_for": lambda v: (f"--spec-{v}-size-n",)},
    "ngram_size_m":   {"bench": None, "kind": "num", "server_only": True,
                       "active_when": ("ngram", NGRAM_MAP_VARIANTS),
                       "flag_for": lambda v: (f"--spec-{v}-size-m",)},
    "ngram_min_hits": {"bench": None, "kind": "num", "server_only": True,
                       "active_when": ("ngram", NGRAM_MAP_VARIANTS),
                       "flag_for": lambda v: (f"--spec-{v}-min-hits",)},
    "ngram_mod_n_match": {"bench": None, "server": ("--spec-ngram-mod-n-match",), "kind": "num",
                          "server_only": True, "active_when": ("ngram", {"ngram-mod"})},
    "ngram_mod_n_min":   {"bench": None, "server": ("--spec-ngram-mod-n-min",), "kind": "num",
                          "server_only": True, "active_when": ("ngram", {"ngram-mod"})},
    "ngram_mod_n_max":   {"bench": None, "server": ("--spec-ngram-mod-n-max",), "kind": "num",
                          "server_only": True, "active_when": ("ngram", {"ngram-mod"})},
    # --- concurrency (server only) ---
    "parallel":     {"bench": None, "server": ("--parallel",), "kind": "num", "server_only": True},
    # --- context extension / capability (server only) ---
    "rope_scaling": {"bench": None, "server": ("--rope-scaling",), "kind": "cat", "server_only": True},
    "yarn_factor":  {"bench": None, "server": ("--yarn-ext-factor",), "kind": "float", "server_only": True},
}


# ---------------------------------------------------------------------------
# Conditional (nested) factors — see docs/CONDITIONAL-FACTORS.md.
# A factor with `active_when: (gate, {values})` participates only when factor
# `gate` resolves to one of `values` in the assignment (a run row or a pinned
# config). `is_active` is the single source of truth for emission (I2) and
# effect-estimation (I3); the stage planner uses it to keep every OA column live
# (I1). A factor without `active_when` is unconditional (always active).
# ---------------------------------------------------------------------------
def is_active(name: str, assignment: dict) -> bool:
    """Whether factor `name` participates under `assignment`. Unconditional
    factors (and unknown names) are always active; a conditional factor is active
    iff its gate is present in the assignment with a value in its live set."""
    spec = FACTORS.get(name)
    if not spec:
        return True
    cond = spec.get("active_when")
    if cond is None:
        return True
    gate, live = cond
    return assignment.get(gate) in live


def active_factors(names, assignment: dict) -> list:
    """The subset of `names` active under `assignment` (a gate value map)."""
    return [n for n in names if is_active(n, assignment)]


def conditional_flags(name: str, gate_value) -> tuple | None:
    """Server flag tuple for a conditional factor, resolved from the live gate
    value via `flag_for` when the flag spelling is variant-dependent. Falls back
    to the static `server` tuple. None if the factor emits nothing here."""
    spec = FACTORS.get(name, {})
    ff = spec.get("flag_for")
    if ff is not None:
        return ff(gate_value)
    return spec.get("server")


def validate_factor_registry(factors: dict = FACTORS) -> list:
    """Return a list of registry errors (empty ⇒ valid). Checks that every
    conditional factor's gate exists, its live values are real gate levels (when
    the gate enumerates them via `translate`), `flag_for` is total over the live
    set, the factor can emit a flag, and the gate graph is acyclic. Run in
    selftest and asserted at build_factors time so a bad registry fails fast."""
    errors: list[str] = []
    edges: dict[str, str] = {}          # factor -> its gate (for cycle check)
    for name, spec in factors.items():
        cond = spec.get("active_when")
        if cond is None:
            continue
        try:
            gate, live = cond
            live = set(live)
        except (TypeError, ValueError):
            errors.append(f"{name}: active_when must be (gate, {{values}})")
            continue
        edges[name] = gate
        if gate not in factors:
            errors.append(f"{name}: active_when gate '{gate}' is not a factor")
        else:
            tr = factors[gate].get("translate")
            if tr is not None:
                unknown = live - set(tr)
                if unknown:
                    errors.append(f"{name}: active_when values "
                                  f"{sorted(unknown)} are not levels of gate '{gate}'")
        ff = spec.get("flag_for")
        if ff is not None:
            for v in sorted(live):
                try:
                    out = ff(v)
                except Exception as e:                       # noqa: BLE001
                    errors.append(f"{name}: flag_for({v!r}) raised {e!r}")
                    continue
                if not out or not all(isinstance(x, str) and x for x in out):
                    errors.append(f"{name}: flag_for({v!r}) must return a "
                                  f"non-empty flag tuple, got {out!r}")
        elif spec.get("server") is None and spec.get("bench") is None:
            errors.append(f"{name}: conditional factor has no flag mapping "
                          f"(need server/bench tuple or flag_for)")
    # cycle detection over factor -> gate edges (must be a DAG)
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n: WHITE for n in edges}

    def visit(n: str) -> bool:                               # True ⇒ cycle found
        color[n] = GREY
        g = edges.get(n)
        if g in edges:
            if color[g] == GREY or (color[g] == WHITE and visit(g)):
                return True
        color[n] = BLACK
        return False

    for n in edges:
        if color[n] == WHITE and visit(n):
            errors.append(f"active_when gate graph has a cycle through '{n}'")
            break
    return errors


def factor_flags(cfg: Config, f: dict, driver: str, ub: int) -> list[list[str]]:
    """Argument groups for the sweepable factors in `f` on the given driver, e.g.
    [["-ngl","64"], ["-nkvo"], ["-ctk","f16"]]. Skips env factors, request-time
    factors (n_depth), factors unsupported on the driver, and conditional factors
    whose gate isn't selected in this row (I2). Handles kv (two flags), booleans,
    batch clamp, named -ot patterns, and variant-dependent flag_for spellings."""
    groups = []
    for name, val in f.items():
        spec = FACTORS.get(name)
        if spec is None or name in cfg.env_factor_names or spec.get("request"):
            continue
        if not is_active(name, f):                 # conditional factor inactive here
            continue
        if spec.get("flag_for") is not None:       # variant-dependent flag spelling
            if driver != "server":                 # flag_for factors are server-only
                continue
            flags = conditional_flags(name, f.get(spec["active_when"][0]))
        else:
            flags = spec.get(driver)
        if flags is None:
            continue
        if spec.get("translate") is not None:
            val = spec["translate"].get(str(val), str(val))
            if val == "":
                continue
        if name == "batch":
            val = str(max(int(val), ub))
        if spec["kind"] == "bool":
            if driver == "server":
                if str(val) in ("1", "on", "true", "True"):
                    groups.append([flags[0]])      # server: bare flag when enabled
            else:
                groups.append([flags[0], str(val)])  # bench: -flag 0|1
        else:
            for fl in flags:
                groups.append([fl, str(val)])
    return groups


def _flat(groups: list[list[str]]) -> list[str]:
    return [tok for g in groups for tok in g]


def is_server_only(name: str) -> bool:
    return bool(FACTORS.get(name, {}).get("server_only"))


def bench_command(cfg: Config, f: dict) -> list[str]:
    cmd = [
        str(cfg.llama_bench),
        "-m", str(cfg.model),
        "-mmp", str(FIXED_MMAP),
        "-p", str(cfg.n_prompt),
        "-n", str(cfg.n_gen),
        "-r", str(cfg.reps),
        "-o", "json",
    ]
    if "fa" not in f:                              # flash-attn fixed unless swept
        cmd += ["-fa", str(FIXED_FA)]
    ub = int(f.get("ubatch", 512))
    if "batch" not in f:                           # batch fixed; needs -b >= -ub
        cmd += ["-b", str(max(FIXED_BATCH, ub))]
    cmd += _flat(factor_flags(cfg, f, "bench", ub))
    return cmd


def run_env(cfg: Config, f: dict) -> dict:
    """Process environment for a run: base env plus any env-factor values."""
    env = dict(os.environ)
    for name in cfg.env_factor_names:
        if name in f:
            env[name] = f[name]
    return env




def _merge_spec_type_parts(parts: list[str]) -> list[str]:
    """Combine multiple --spec-type <value> entries in a string-parts list
    into a single comma-separated --spec-type <v1,v2> entry. llama.cpp accepts
    comma-separated spec types and the orthogonal array may emit one --spec-type
    from the mtp factor and another from the ngram factor simultaneously."""
    kept: list[str] = []
    vals: list[str] = []
    for p in parts:
        if p.startswith("--spec-type "):
            v = p.split(None, 1)[1].strip()
            if v:
                vals.append(v)
        else:
            kept.append(p)
    if vals:
        kept.append("--spec-type " + ",".join(vals))
    return kept


def _merge_spec_type_args(args: list[str]) -> list[str]:
    """Combine multiple --spec-type / value flag-pairs in a flat arg list
    into a single --spec-type / v1,v2 pair."""
    result: list[str] = []
    vals: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--spec-type" and i + 1 < len(args):
            v = args[i + 1]
            if v:
                vals.append(v)
            i += 2
        else:
            result.append(args[i])
            i += 1
    if vals:
        result.append("--spec-type")
        result.append(",".join(vals))
    return result


def server_command(cfg: Config, f: dict, ctx: int) -> str:
    ub = int(f.get("ubatch", 512))
    parts = [f"-m {cfg.model.name}", f"-c {ctx}"]
    if not FIXED_MMAP:
        parts.append("--no-mmap")
    if "fa" not in f:
        parts.append(f"-fa {FIXED_FA}")
    if "batch" not in f:
        parts.append(f"-b {max(FIXED_BATCH, ub)}")
    parts += [" ".join(shlex.quote(t) for t in g)
              for g in factor_flags(cfg, f, "server", ub)]
    if "parallel" not in f and cfg.parallel > 1:
        parts.append(f"--parallel {cfg.parallel}")
    # Multi-token prediction: if the model ships a NextN/MTP head, enable
    # draft-mtp speculative decoding for extra generation throughput. With the
    # server driver this speedup IS measured; with llama-bench it is NOT (bench
    # can't do speculative decoding) and stacks on top of the reported t/s.
    if cfg.emit_mtp and cfg.hw.get("n_nextn", 0) > 0:
        if "mtp" not in f:                    # MTP fixed on unless swept
            parts.append("--spec-type draft-mtp")
        if "spec_n_max" not in f:
            parts.append(f"--spec-draft-n-max {cfg.spec_draft_n_max}")
    # ngram self-speculation: pattern-matching decoding that needs no draft model.
    # When enabled but not swept, activate a sensible default variant (ngram-mod);
    # its own draft length is ngram-mod's default, not --spec-draft-n-max (a
    # draft-model/MTP knob that has no effect on ngram).
    if cfg.ngram and "ngram" not in f:
        parts.append("--spec-type ngram-mod")
    # MTP + ngram can both emit --spec-type; merge into one comma-separated value.
    parts = _merge_spec_type_parts(parts)
    cmd = " \\\n    ".join(["./llama-server"] + parts)
    # prepend any winning env-var factor values as an env prefix
    env_prefix = " ".join(f"{n}={f[n]}" for n in sorted(cfg.env_factor_names) if n in f)
    return (env_prefix + " \\\n  " + cmd) if env_prefix else cmd


_OOM_PAT = re.compile(
    r"out of memory|failed to allocate|ROCm error|hipErrorOutOfMemory|"
    r"cudaErrorMemoryAllocation|ggml_backend_.*failed",
    re.IGNORECASE,
)


def parse_bench_json(stdout: str):
    """Return (pp_tps, tg_tps) from llama-bench JSON output, or (None, None)."""
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError:
        return None, None
    pp = tg = None
    for row in rows:
        n_gen = row.get("n_gen", 0)
        n_prompt = row.get("n_prompt", 0)
        ts = row.get("avg_ts")
        if ts is None:
            continue
        if n_gen and not n_prompt:
            tg = ts
        elif n_prompt and not n_gen:
            pp = ts
    return pp, tg


def fmt_dur(secs: float) -> str:
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m{secs % 60:02d}s"
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}m"


def with_ticker(prefix: str, timeout: int, fn):
    """Run fn() showing a live elapsed ticker on a TTY; plain start line if the
    output is redirected (keeps logs one-line-in/one-line-out)."""
    if not sys.stdout.isatty():
        print("trying " + prefix + " ...", flush=True)
        return fn()
    stop = threading.Event()
    t0 = time.time()

    def tick():
        while not stop.wait(1.0):
            sys.stdout.write(f"\rtrying {prefix} ... {fmt_dur(time.time() - t0)} "
                             f"(timeout {fmt_dur(timeout)})   ")
            sys.stdout.flush()

    th = threading.Thread(target=tick, daemon=True)
    th.start()
    try:
        return fn()
    finally:
        stop.set()
        th.join(timeout=0.2)
        sys.stdout.write("\r" + " " * (len(prefix) + 56) + "\r")  # clear line
        sys.stdout.flush()


def run_with_progress(cfg: Config, f: dict, timeout: int, prefix: str):
    return with_ticker(prefix, timeout, lambda: drive_one(cfg, f, timeout))


def run_one(cfg: Config, f: dict, timeout: int):
    cmd = bench_command(cfg, f)
    t0 = time.time()
    status, pp, tg = "OK", 0.0, 0.0
    sampler = VRAMSampler().__enter__() if cfg.measure_vram else None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=run_env(cfg, f))
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0 or _OOM_PAT.search(combined):
            status = "OOM" if _OOM_PAT.search(combined) else "ERROR"
        else:
            pp_, tg_ = parse_bench_json(proc.stdout)
            if tg_ is None:
                status = "PARSE_FAIL"
            else:
                pp, tg = pp_ or 0.0, tg_
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
    finally:
        if sampler:
            sampler.__exit__()
    return {"status": status, "pp_tps": pp, "tg_tps": tg,
            "secs": time.time() - t0, "vram_mib": sampler.peak if sampler else 0}


# ---------------------------------------------------------------------------
# Server driver: launch llama-server and drive real generation (measures MTP /
# speculative decoding and real concurrency, which llama-bench cannot).
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_server_args(cfg: Config, f: dict, port: int, n_ctx: int) -> list[str]:
    ub = int(f.get("ubatch", 512))
    args = [
        str(cfg.llama_server), "-m", str(cfg.model),
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(n_ctx),                          # mmap on is the server default
    ]
    if not FIXED_MMAP:
        args.append("--no-mmap")                   # llama-server flag (NOT -mmp)
    if "fa" not in f:                              # flash-attn fixed unless swept
        args += ["-fa", str(FIXED_FA)]
    if "batch" not in f:
        args += ["-b", str(max(FIXED_BATCH, ub))]
    args += _flat(factor_flags(cfg, f, "server", ub))
    if "parallel" not in f and cfg.parallel > 1:   # concurrency (fixed) if not swept
        args += ["--parallel", str(cfg.parallel)]
    if cfg.emit_mtp and cfg.hw.get("n_nextn", 0) > 0:
        if "mtp" not in f:                         # MTP fixed on unless swept
            args += ["--spec-type", "draft-mtp"]
        if "spec_n_max" not in f:                  # default n_max if not swept
            args += ["--spec-draft-n-max", str(cfg.spec_draft_n_max)]
    # ngram self-speculation: enable ngram-mod by default when ngram is on but not
    # swept. ngram's draft length is ngram-mod's own default, not --spec-draft-n-max
    # (a draft-model/MTP knob with no effect on ngram).
    if cfg.ngram and "ngram" not in f:
        args += ["--spec-type", "ngram-mod"]
    # Merge duplicate --spec-type entries (MTP + ngram) into one comma-sep value
    args = _merge_spec_type_args(args)
    return args


def _wait_health(port: int, deadline: float, proc=None) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False  # server process exited (died during load) — stop waiting
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


# Varied prose so speculative-decoding acceptance is realistic. A repeated single
# token is trivially predictable and would inflate the measured MTP speedup.
_CORPUS = (
    "The history of computing spans centuries, from mechanical calculators to "
    "modern processors running billions of operations per second. In distributed "
    "systems, consensus algorithms such as Raft and Paxos let unreliable machines "
    "agree on a single value despite failures. Photosynthesis converts sunlight, "
    "water, and carbon dioxide into glucose and oxygen. The novel opens in a quiet "
    "coastal town where the protagonist, returning after many years, confronts an "
    "old rival and a buried secret. Interest-rate decisions ripple through global "
    "markets as traders reprice risk and adjust their portfolios accordingly. A "
    "recipe for bread needs flour, water, salt, and time for the dough to rise. "
)


def _realistic_prompt(n_tokens: int) -> str:
    """A varied-text prompt of roughly n_tokens tokens (~4 chars/token)."""
    approx_chars = max(1, n_tokens) * 4
    reps = approx_chars // len(_CORPUS) + 1
    return (_CORPUS * reps)[:approx_chars]


def _completion(port: int, prompt, n_gen: int, timeout: int, cache: bool = False) -> dict:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_gen,
        "temperature": 0,
        "cache_prompt": cache,
        "timings_per_token": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _measure_round(port: int, prompt, n_gen: int, par: int, timeout: int, cache=False):
    """One round of `par` concurrent completions; returns (responses, wall_s)."""
    with ThreadPoolExecutor(max_workers=par) as ex:
        w0 = time.time()
        res = list(ex.map(
            lambda _: _completion(port, prompt, n_gen, timeout, cache), range(par)))
        return res, time.time() - w0


class ServerSession:
    """A running llama-server, reusable across runs that share load-time params
    (only the request — prompt length via n_depth — varies). Launch once, issue
    many measurements, close once."""

    def __init__(self, cfg: Config, launch_f: dict, n_ctx: int, timeout: int):
        self.cfg = cfg
        self.ok = False
        self.err = ""
        self.port = _free_port()
        args = build_server_args(cfg, launch_f, self.port, n_ctx)
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True,
                                     env=run_env(cfg, launch_f))
        # Wait for the server to come up, but give up fast if the process dies
        # (crashed on load) or exceeds the startup budget — don't hang for the
        # whole per-run timeout.
        start_deadline = time.time() + cfg.server_start_timeout
        if _wait_health(self.port, start_deadline, self.proc):
            self.ok = True
        else:
            died = self.proc.poll() is not None
            self.proc.terminate()
            try:
                _, err = self.proc.communicate(timeout=10)
                self.err = err or ""
            except subprocess.TimeoutExpired:
                self.proc.kill()
            if not self.err:
                self.err = (f"server exited during load" if died else
                            f"server not healthy within {cfg.server_start_timeout}s")

    def measure(self, prompt_len, n_gen, par, reps, timeout):
        prompt = _realistic_prompt(prompt_len)
        if par == 1:
            # Single stream: prefill once (warm request → real pp + primes the KV
            # cache), then reuse the cached prompt so each rep measures pure decode
            # (tg) without re-prefilling — much faster at high context, and a
            # cleaner decode number.
            warm = _completion(self.port, prompt, n_gen, timeout, cache=True)
            pp = warm.get("timings", {}).get("prompt_per_second", 0.0) or 0.0
            tps = []
            for _ in range(max(1, reps)):
                r = _completion(self.port, prompt, n_gen, timeout, cache=True)
                tps.append(r.get("timings", {}).get("predicted_per_second", 0.0) or 0.0)
            return pp, sum(tps) / len(tps)
        # Concurrency: realistic serving — every request prefills; aggregate over
        # the streams. Warmup once, then average per-round throughput over reps.
        _measure_round(self.port, prompt, n_gen, par, timeout)  # warmup, discard
        pps, tps = [], []
        for _ in range(max(1, reps)):
            res, wall = _measure_round(self.port, prompt, n_gen, par, timeout)
            tg_tok = sum(r.get("timings", {}).get("predicted_n", 0) for r in res)
            pp_tok = sum(r.get("timings", {}).get("prompt_n", 0) for r in res)
            tps.append(tg_tok / wall if wall > 0 else 0.0)
            pps.append(pp_tok / wall if wall > 0 else 0.0)
        return sum(pps) / len(pps), sum(tps) / len(tps)

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def load_key(cfg: Config, f: dict):
    """Server launch identity: every factor except the request-time n_depth."""
    return tuple((k, f.get(k)) for k in cfg.factors if k != "n_depth")


def load_key_str(cfg: Config, f: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in load_key(cfg, f))


# ---------------------------------------------------------------------------
# Crash journal: a config that hard-hangs or reboots the machine writes no
# result. We record "about to try X" and fsync it to disk BEFORE launching, so
# on restart a started-but-never-finished config is a suspected killer and is
# skipped instead of retried (which would reboot again). Two risk phases:
# "load" (server model load into VRAM, per group) and "run" (a measurement).
# ---------------------------------------------------------------------------
def journal_write(jh, *fields):
    jh.write("\t".join(str(x) for x in fields) + "\n")
    jh.flush()
    os.fsync(jh.fileno())  # durable before the risky operation begins


def read_journal(path: Path):
    """Return (tried_load{key:cfg}, ok_load{keys}, tried_run{run_id:cfg})."""
    tried_load, ok_load, tried_run = {}, set(), {}
    if not path.exists():
        return tried_load, ok_load, tried_run
    for line in path.read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 3 and p[0] == "TRY" and p[1] == "load":
            try:
                tried_load[p[2]] = json.loads(p[3]) if len(p) > 3 else {}
            except json.JSONDecodeError:
                tried_load[p[2]] = {}
        elif len(p) >= 3 and p[0] == "OK" and p[1] == "load":
            ok_load.add(p[2])
        elif len(p) >= 3 and p[0] == "TRY" and p[1] == "run":
            try:
                tried_run[p[2]] = json.loads(p[3]) if len(p) > 3 else {}
            except json.JSONDecodeError:
                tried_run[p[2]] = {}
    return tried_load, ok_load, tried_run


def measure_in_session(cfg: Config, f: dict, session, timeout: int) -> dict:
    """Measure one config against an (already-launched) server session."""
    t0 = time.time()
    if session is None or not session.ok:
        err = session.err if session else ""
        status = "OOM" if _OOM_PAT.search(err or "") else "ERROR"
        return {"status": status, "pp_tps": 0.0, "tg_tps": 0.0, "secs": 0.0,
                "vram_mib": 0}
    prompt_len = cfg.n_prompt + int(f.get("n_depth", 0))
    par = int(f.get("parallel", cfg.parallel))
    sampler = VRAMSampler().__enter__() if cfg.measure_vram else None
    status, pp, tg = "OK", 0.0, 0.0
    try:
        pp, tg = session.measure(prompt_len, cfg.n_gen, par, cfg.reps, timeout)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        status = "ERROR"
    finally:
        if sampler:
            sampler.__exit__()
    return {"status": status, "pp_tps": pp, "tg_tps": tg,
            "secs": time.time() - t0, "vram_mib": sampler.peak if sampler else 0}


def server_run_one(cfg: Config, f: dict, timeout: int):
    """Standalone server measurement for one config (own session). Used by the
    context probe; the sweep groups configs to reuse sessions instead."""
    par = int(f.get("parallel", cfg.parallel))
    prompt_len = cfg.n_prompt + int(f.get("n_depth", 0))
    n_ctx = prompt_len + cfg.n_gen + 256
    if par > 1:
        n_ctx *= par
    session = ServerSession(cfg, f, n_ctx, timeout)
    try:
        return measure_in_session(cfg, f, session, timeout)
    finally:
        session.close()


def drive_one(cfg: Config, f: dict, timeout: int):
    """Dispatch to the configured driver (standalone, one process per run)."""
    if cfg.driver == "server":
        return server_run_one(cfg, f, timeout)
    return run_one(cfg, f, timeout)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def score_of(r: dict) -> float:
    """Primary objective score for a run (effective throughput if present)."""
    return float(r.get("eff_tps", r.get("tg_tps", 0.0)))


def pareto_frontier(rows: list[dict]) -> list[dict]:
    """Non-dominated set maximizing (context depth, objective score) among OK."""
    ok = [r for r in rows if r["status"] == "OK" and score_of(r) > 0]
    frontier = []
    for r in ok:
        depth, s = int(r["n_depth"]), score_of(r)
        dominated = any(
            int(o["n_depth"]) >= depth and score_of(o) >= s and o is not r
            and (int(o["n_depth"]) > depth or score_of(o) > s)
            for o in ok
        )
        if not dominated:
            frontier.append(r)
    return sorted(frontier, key=lambda r: int(r["n_depth"]))


def verified_depth_of(cfg: Config, rows: list[dict], r: dict) -> int:
    """Deepest n_depth among OK rows sharing r's launch factors (everything but
    n_depth) — the most context this exact config is *known* to load. With the
    server driver those siblings shared one session sized at the group's max
    depth; with bench they each loaded at least their own depth."""
    def key(row):
        return tuple(str(row.get(k)) for k in cfg.factors if k != "n_depth")
    mine = key(r)
    ds = [int(o["n_depth"]) for o in rows
          if o.get("status") == "OK" and key(o) == mine]
    return max(ds, default=int(r["n_depth"]))


def recommended_ctx(cfg: Config, r: dict, verified_depth: int | None = None) -> int:
    """Context (`-c`) to emit for a winning row.

    Base: the footprint the sweep actually verified for this row — the server
    driver sizes its session at ``n_prompt + n_depth + n_gen + 256`` (times
    ``--parallel``) in ``server_run_one`` — rounded *down* to a tidy multiple;
    rounding up would inflate the KV cache past the verified point and can OOM
    at launch.

    Floor: a server with a tiny context isn't worth pasting (a depth-0 winner
    would emit ``-c 1024``), so the result is raised to at least ``ctx_floor``
    per slot — but never past the footprint of ``verified_depth`` (the deepest
    this launch config measured OK; see ``verified_depth_of``). The floor rides
    on evidence, it never outruns it: with no deeper sibling this returns the
    row's own verified footprint unchanged.
    """
    par = max(1, int(r.get("parallel", cfg.parallel)))

    def footprint(d: int) -> int:
        v = cfg.n_prompt + d + cfg.n_gen + 256
        return v * par if par > 1 else v

    d = int(r["n_depth"])
    want = max(footprint(d), cfg.ctx_floor * par)
    cap = footprint(d if verified_depth is None else max(d, verified_depth))
    return max(256, (min(want, cap) // 256) * 256)


def ctx_floor_note(cfg: Config, r: dict, ctx: int) -> str | None:
    """One-line note when the emitted -c had to stay below the usable floor
    because the sweep holds no deeper evidence for this config."""
    par = max(1, int(r.get("parallel", cfg.parallel)))
    if ctx >= (cfg.ctx_floor * par) // 256 * 256:
        return None
    return (f"note: -c {ctx} is below the {cfg.ctx_floor} usable-context floor — "
            "the sweep never verified this config any deeper.")


def pick_recommendations(cfg: Config, rows: list[dict]):
    """The report's three picks: (fastest, balanced, longest).

    FASTEST means fastest *usable* — best score among configs whose emitted
    command can hold the usable-context floor (verified evidence, not hope) —
    so the headline is never a config that only shines with an empty KV cache.
    Falls back to the raw fastest when nothing meets the floor (e.g. a model
    whose native context is below it); the floor note flags that. BALANCED is
    the best score *measured at* depth >= the floor (speed while actually deep
    in context); LONGEST is the deepest OK row.
    """
    ok = [r for r in rows if r["status"] == "OK"]
    if not ok:
        return None, None, None

    def holds_floor(r):
        ctx = recommended_ctx(cfg, r, verified_depth_of(cfg, rows, r))
        return ctx_floor_note(cfg, r, ctx) is None

    pool = [r for r in ok if holds_floor(r)]
    fastest = max(pool or ok, key=score_of)
    deep = [r for r in ok if int(r["n_depth"]) >= cfg.ctx_floor]
    balanced = max(deep, key=score_of) if deep else None
    longest = max(ok, key=lambda r: (int(r["n_depth"]), score_of(r)))
    return fastest, balanced, longest


def kv_downgrade_hint(r: dict) -> str | None:
    """One-line `q8_0` suggestion for rows whose KV cache is heavier than the
    near-lossless `q8_0` floor (i.e. `f32`/`f16`/`bf16`). q8_0 ~halves the KV
    footprint per token for roughly double the context in the same VRAM, so it
    is the cheapest lever back from a memory cliff — surfaced at copy-paste time
    rather than discovered by an OOM. Returns None when nothing is to be gained
    (KV already q8_0 or lossier)."""
    kv = str(r.get("kv_type", ""))
    if kv not in KV_QUALITY or KV_QUALITY.index(kv) >= KV_QUALITY.index("q8_0"):
        return None
    return ("tip: KV cache is " + kv + " — set -ctk q8_0 -ctv q8_0 to ~halve KV "
            "memory (near-lossless) for roughly double the context / more OOM "
            "headroom, at a small decode-speed cost.")


def ttft_note(r: dict, ctx: int) -> str | None:
    """One-line prefill-cost estimate for the emitted -c: filling the context
    at this config's measured pp speed. A giant -c can mean many minutes before
    the first token — great tg alone doesn't make a config interactive — so the
    cost is stated right where the command gets copied. Prefill slows somewhat
    with depth, so treat the estimate as optimistic."""
    try:
        pp = float(r.get("pp_tps") or 0)
    except (TypeError, ValueError):
        return None
    if pp <= 0 or ctx <= 0:
        return None
    note = (f"prefill cost: a full {ctx}-token prompt ≈ {fmt_dur(ctx / pp)} "
            f"to first token at pp={pp:.0f} t/s")
    if ctx > 16384:
        note += f" (an 8k prompt ≈ {fmt_dur(8192 / pp)})"
    return note


def report(cfg: Config, rows: list[dict], probe: dict | None = None):
    ok = [r for r in rows if r["status"] == "OK"]
    print("\n" + "=" * 70)
    print(f"RESULTS: {len(ok)}/{len(rows)} configs succeeded")
    print("=" * 70)
    if not ok:
        bad = {}
        for r in rows:
            bad[r["status"]] = bad.get(r["status"], 0) + 1
        print("No successful runs. Status breakdown:", bad)
        return

    if cfg.emit_mtp and cfg.hw.get("n_nextn", 0) > 0:
        if cfg.driver == "server":
            print("NOTE: model has an MTP head — these numbers INCLUDE the "
                  "draft-mtp speculative-decoding speedup (server driver).")
        else:
            print("NOTE: model has an MTP head — the server commands enable "
                  "draft-mtp, but the measured t/s below does NOT include that "
                  "boost (bench can't do spec decoding). Use --driver server to "
                  "measure it.")

    if cfg.score == "eff":
        print(f"objective  : effective t/s for a {cfg.profile} request "
              f"({cfg.n_prompt} prompt + {cfg.n_gen} gen tokens)")
    else:
        print("objective  : generation t/s (decode only; pp reported but not "
              "scored — --score eff to blend prefill in)")

    fastest, balanced, longest = pick_recommendations(cfg, rows)

    def show(title, r):
        if not r:
            print(f"\n### {title}: none met the constraint")
            return
        print(f"\n### {title}")
        raw = ((f"tg={r['tg_tps']:.1f}  " if cfg.score == "eff" else "")
               + f"pp={r['pp_tps']:.1f}")
        print(f"  {cfg.score}={score_of(r):.1f} t/s  ({raw})  "
              f"depth={r['n_depth']}  ngl={r['ngl']}  "
              f"t={r['threads']}  kv={r['kv_type']}  ub={r['ubatch']}")
        if r.get("verify_n"):
            print(f"  verified: median of {r['verify_n']} measurements "
                  f"(spread {r['spread_pct']:.0f}%)")
        # size context to what the sweep verified for this config, floored at
        # the usable floor where evidence allows (see recommended_ctx); a
        # bigger -c can OOM at launch.
        ctx = recommended_ctx(cfg, r, verified_depth_of(cfg, rows, r))
        print("  suggested llama-server command:")
        print("    " + server_command(cfg, r, ctx))
        for extra in (kv_downgrade_hint(r), ctx_floor_note(cfg, r, ctx),
                      ttft_note(r, ctx)):
            if extra:
                print("  " + extra)

    show("FASTEST (max speed, usable context)", fastest)
    show(f"BALANCED (best with context >= {cfg.ctx_floor})", balanced)
    show("MAX CONTEXT", longest)

    if probe:
        r, depth, tps = probe["row"], probe["depth"], probe["tg_tps"]
        kind = ("the model's native limit" if probe["at_cap"]
                else "an OOM boundary")
        print("\n### PROBED CEILING (largest -c that loads — beyond the "
              "swept range)")
        print(f"  ~{depth} tokens ({kind})"
              + (f"  tg={tps:.1f} t/s spot-check there" if tps else "")
              + f"  ngl={r['ngl']}  kv={r['kv_type']}  ub={r['ubatch']}")
        print(f"  suggested llama-server command (-c {probe['safe_ctx']}, "
              "~10% headroom under the ceiling):")
        print("    " + server_command(cfg, r, probe["safe_ctx"]))
        for extra in (kv_downgrade_hint(r), ttft_note(r, probe["safe_ctx"])):
            if extra:
                print("  " + extra)

    kind = "effective" if cfg.score == "eff" else "generation"
    print(f"\n### Pareto frontier (context vs {kind} t/s)")
    for r in pareto_frontier(rows):
        raw = f"(tg={r['tg_tps']:5.1f})  " if cfg.score == "eff" else ""
        print(f"  depth={int(r['n_depth']):>6}  {cfg.score}={score_of(r):6.1f} t/s  "
              f"{raw}ngl={r['ngl']:>3}  "
              f"kv={r['kv_type']:>4}  ub={r['ubatch']:>4}")


def factor_level_means(rows: list[dict], factor: str) -> dict:
    """Mean objective score per level of a factor (OK runs only) — the Taguchi
    main effect for a balanced design. A conditional factor (active_when) is
    scored only over rows where it was active (I3): rows where its gate selected
    a different variant are inert and carry no information about its effect."""
    ok = [r for r in rows if r["status"] == "OK" and is_active(factor, r)]
    means = {}
    levels = sorted(set(str(r[factor]) for r in ok if factor in r),
                    key=lambda x: (len(x), x))
    for lvl in levels:
        vals = [score_of(r) for r in ok if str(r.get(factor)) == lvl]
        if vals:
            means[lvl] = sum(vals) / len(vals)
    return means


def refine_numeric(vals: list[int], best: int) -> list[str]:
    """Finer grid of levels bracketing `best` (its neighbours in the current
    grid), for the next refinement pass."""
    vals = sorted(set(vals))
    if len(vals) <= 1:
        return [str(v) for v in vals] or [str(best)]
    i = vals.index(best) if best in vals else min(range(len(vals)),
                                                  key=lambda k: abs(vals[k] - best))
    lo = vals[i - 1] if i > 0 else vals[i]
    hi = vals[i + 1] if i < len(vals) - 1 else vals[i]
    if lo == hi:
        step = max(1, (vals[-1] - vals[0]) // len(vals))
        lo, hi = max(0, best - step), best + step
    return [str(x) for x in five_levels_span(lo, hi)]


def refine_factors(cfg: Config, rows: list[dict]) -> dict:
    """Produce the next pass's factor levels: settle low-impact factors at their
    winning level, and refine high-impact factors onto a finer grid around their
    best value (numeric) or their top levels (categorical)."""
    ranges, bests = {}, {}
    for name in cfg.factors:
        means = factor_level_means(rows, name)
        if not means:
            ranges[name], bests[name] = 0.0, cfg.factors[name][0]
            continue
        bests[name] = max(means, key=means.get)
        ranges[name] = max(means.values()) - min(means.values()) if len(means) > 1 else 0.0
    max_range = max(ranges.values(), default=0.0) or 1.0

    new = {}
    for name, cur in cfg.factors.items():
        # n_depth is the report's tradeoff axis (speed vs context), not a knob to
        # optimize to one value. Keep its full spread across passes so the final
        # pass still maps the whole curve — otherwise FASTEST/BALANCED/MAX-CONTEXT
        # collapse to a single depth and the three recommendations become identical.
        if name == "n_depth":
            new[name] = cur
            continue
        rng, best = ranges[name], bests[name]
        active = len(cur) > 1 and rng >= 0.25 * max_range
        kind = FACTORS.get(name, {}).get("kind", "cat")
        numeric = kind == "num" and name not in cfg.env_factor_names
        if not active:
            new[name] = [str(best)]                       # settle at the winner
        elif numeric:
            new[name] = refine_numeric([int(x) for x in cur], int(best))  # finer grid
        else:                                             # cat/float/env: keep top few
            means = factor_level_means(rows, name)
            ranked = sorted(means, key=means.get, reverse=True)
            new[name] = ranked[:3] if len(ranked) >= 3 else ranked
    return new


def _svg_pareto(rows: list[dict], vram_total: float = 0,
                ylabel: str = "effective t/s") -> str:
    """Inline SVG: effective t/s (left y) vs context (x), all OK runs as faint dots,
    the Pareto frontier as a highlighted line. If runs carry measured VRAM, overlay
    the VRAM curve on a right-hand axis plus the physical-ceiling line. Theme-neutral."""
    ok = [r for r in rows if r["status"] == "OK" and score_of(r) > 0]
    if len(ok) < 2:
        return ""
    pts = [(int(r["n_depth"]), score_of(r)) for r in ok]
    front = [(int(r["n_depth"]), score_of(r)) for r in pareto_frontier(rows)]
    # measured VRAM per run (only if --vram was used and values are present)
    vpts = sorted((int(r["n_depth"]), float(r.get("vram_mib") or 0))
                  for r in ok if float(r.get("vram_mib") or 0) > 0)
    have_vram = len(vpts) >= 2
    W, H, ml, mt, mb = 680, 340, 62, 14, 46
    mr = 58 if have_vram else 16
    # zoom the x-axis to the data range too (same treatment as y below, clamped
    # at 0 since negative context is meaningless): fixed 0..max scaling stacked
    # every point of a single-depth run on the right edge, and rounded-up ticks
    # could land outside the viewBox.
    xs = [x for x, _ in pts]
    xpad = (max(xs) - min(xs)) * 0.1 or max(max(xs) * 0.05, 1)
    xlo, xhi = max(0, min(xs) - xpad), max(xs) + xpad
    ys = [y for _, y in pts]
    # zoom the left y-axis to 10% of the data *range* beyond each end (NOT 10% of the
    # value, and NOT from 0) so the actual variation fills ~80% of the height —
    # otherwise a high-baseline low-spread curve (e.g. the 270M model, ~450 t/s with
    # ~20 spread) still looks like a flat horizontal line.
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.1 or max(hi * 0.05, 1)
    ylo, yhi = lo - pad, hi + pad
    # right VRAM axis: from 0 to the physical ceiling (or headroom above peak)
    vhi = max([v for _, v in vpts] + [vram_total]) * 1.05 if have_vram else 1

    def sx(x):
        return ml + ((x - xlo) / (xhi - xlo)) * (W - ml - mr)

    def sy(y):
        return H - mb - ((y - ylo) / (yhi - ylo)) * (H - mt - mb)

    def sv(v):                               # right axis -> pixels
        return H - mb - (v / vhi) * (H - mt - mb)

    g = []
    for i in range(5):                       # 5 ticks across the zoomed x-range
        x = xlo + (xhi - xlo) * i / 4
        gx = sx(x)
        g.append(f"<line x1='{gx:.0f}' y1='{mt}' x2='{gx:.0f}' y2='{H - mb}' class='grid'/>")
        g.append(f"<text x='{gx:.0f}' y='{H - mb + 16}' class='ax xt'>{int(x)}</text>")
    for i in range(5):                       # 5 ticks across the zoomed y-range
        y = ylo + (yhi - ylo) * i / 4
        gy = sy(y)
        g.append(f"<line x1='{ml}' y1='{gy:.0f}' x2='{W - mr}' y2='{gy:.0f}' class='grid'/>")
        g.append(f"<text x='{ml - 6}' y='{gy + 4:.0f}' class='ax yt'>{y:.0f}</text>")
    dots = "".join(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3' class='dot'/>"
                   for x, y in pts)
    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in front)
    fdots = "".join(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='4' class='fdot'/>"
                    for x, y in front)
    vram_svg = ""
    if have_vram:
        for i in range(5):                   # right-axis ticks in GiB
            v = vhi * i / 4
            g.append(f"<text x='{W - mr + 6}' y='{sv(v) + 4:.0f}' class='ax vt'>"
                     f"{v / 1024:.1f}</text>")
        vpoly = " ".join(f"{sx(x):.1f},{sv(v):.1f}" for x, v in vpts)
        vdots = "".join(f"<circle cx='{sx(x):.1f}' cy='{sv(v):.1f}' r='3' class='vdot'/>"
                        for x, v in vpts)
        vram_svg = f"<polyline points='{vpoly}' class='vline'/>{vdots}"
        if vram_total > 0:                   # physical VRAM ceiling
            cy0 = sv(vram_total)
            vram_svg += (
                f"<line x1='{ml}' y1='{cy0:.0f}' x2='{W - mr}' y2='{cy0:.0f}' "
                f"class='ceil'/><text x='{ml + 6}' y='{cy0 - 5:.0f}' class='clbl'>"
                f"VRAM ceiling {vram_total / 1024:.0f} GiB</text>")
    cy = (mt + H - mb) / 2
    vaxis = (f"<text x='{W - 14}' y='{cy:.0f}' class='albl vaxlbl' "
             f"transform='rotate(90 {W - 14} {cy:.0f})'>VRAM used (GiB)</text>"
             if have_vram else "")
    return (
        f"<svg viewBox='0 0 {W} {H}' class='chart' role='img' "
        f"aria-label='effective throughput and VRAM versus context'>"
        "<style>.chart .grid{stroke:#8883} .chart .ax{fill:#888;font:11px system-ui}"
        ".chart .xt{text-anchor:middle} .chart .yt{text-anchor:end} .chart .vt{text-anchor:start}"
        ".chart .dot{fill:#8887} .chart .fline{fill:none;stroke:#2ca88f;stroke-width:2.5}"
        ".chart .fdot{fill:#2ca88f} .chart .albl{fill:#888;font:12px system-ui;"
        "text-anchor:middle}"
        ".chart .vline{fill:none;stroke:#e0993e;stroke-width:2;stroke-dasharray:1}"
        ".chart .vdot{fill:#e0993e} .chart .vaxlbl{fill:#e0993e}"
        ".chart .ceil{stroke:#d1495b;stroke-width:1.5;stroke-dasharray:5 4}"
        ".chart .clbl{fill:#d1495b;font:11px system-ui}</style>"
        f"{''.join(g)}<polyline points='{poly}' class='fline'/>{dots}{fdots}{vram_svg}"
        f"<text x='{(ml + W - mr) / 2:.0f}' y='{H - 6}' class='albl'>context (tokens)</text>"
        f"<text x='16' y='{cy:.0f}' class='albl' transform='rotate(-90 16 {cy:.0f})'>"
        f"{ylabel}</text>{vaxis}</svg>")


def write_html_report(cfg: Config, rows: list[dict], path: Path,
                      probe: dict | None = None):
    import html as _html

    def esc(x):
        return _html.escape(str(x))

    ok = [r for r in rows if r["status"] == "OK"]
    best, balanced, longest = pick_recommendations(cfg, rows)
    pareto = pareto_frontier(rows)

    def card(title, r):
        if not r:
            return f"<div class=card><h3>{esc(title)}</h3><p class=muted>none met the constraint</p></div>"
        ctx = recommended_ctx(cfg, r, verified_depth_of(cfg, rows, r))
        cmd = esc(server_command(cfg, r, ctx))
        ver = (f"verified: median of {r['verify_n']} measurements "
               f"(spread {r['spread_pct']:.0f}%)" if r.get("verify_n") else None)
        tip = "".join(f"<div class=muted>{esc(x)}</div>"
                      for x in (ver, kv_downgrade_hint(r),
                                ctx_floor_note(cfg, r, ctx), ttft_note(r, ctx)) if x)
        raw = ((f"tg {r['tg_tps']:.1f} · " if cfg.score == "eff" else "")
               + f"pp {r['pp_tps']:.1f}")
        return (f"<div class=card><h3>{esc(title)}</h3>"
                f"<div class=big>{score_of(r):.1f} <span class=unit>{cfg.score} t/s</span></div>"
                f"<div class=muted>{raw} · "
                f"depth {esc(r['n_depth'])} · ngl {esc(r['ngl'])} · kv {esc(r['kv_type'])} · "
                f"ub {esc(r['ubatch'])} · t {esc(r['threads'])}</div>"
                f"<pre>{cmd}</pre>{tip}</div>")

    probe_card = ""
    if probe:
        r = probe["row"]
        cmd = esc(server_command(cfg, r, probe["safe_ctx"]))
        kind = "model's native limit" if probe["at_cap"] else "OOM boundary"
        tps = probe["tg_tps"]
        probe_card = (
            f"<div class=card><h3>Probed ceiling (beyond swept range)</h3>"
            f"<div class=big>{probe['depth']:,} <span class=unit>tokens</span></div>"
            f"<div class=muted>{kind}"
            + (f" · tg {tps:.1f} t/s spot-check" if tps else "")
            + f" · ngl {esc(r['ngl'])} · kv {esc(r['kv_type'])} · "
            f"ub {esc(r['ubatch'])} · t {esc(r['threads'])}</div>"
            f"<pre>{cmd}</pre>"
            f"<div class=muted>largest context that loads (binary search); "
            f"speed there is a single measurement, not swept</div>"
            + "".join(f"<div class=muted>{esc(x)}</div>"
                      for x in (ttft_note(r, probe["safe_ctx"]),) if x)
            + "</div>")

    # main-effects bars, factors ordered by range (impact) descending
    effects = []
    for name in cfg.factors:
        means = factor_level_means(rows, name)
        if len(means) >= 2:
            effects.append((max(means.values()) - min(means.values()), name, means))
    effects.sort(reverse=True)
    gmax = max((rng for rng, _, _ in effects), default=1) or 1
    fx_html = []
    for rng, name, means in effects:
        vmax = max(means.values()) or 1
        bars = "".join(
            f"<div class=lvl><span class=ll>{esc(l)}</span>"
            f"<span class=bar style='width:{max(2, v / vmax * 100):.0f}%'></span>"
            f"<span class=lv>{v:.1f}</span></div>"
            for l, v in means.items())
        impact = rng / gmax * 100
        fx_html.append(
            f"<div class=fx><div class=fxh><b>{esc(name)}</b>"
            f"<span class=muted>impact {rng:.1f} ({impact:.0f}%)</span></div>{bars}</div>")

    # pareto table (the score IS tg under --score tg: no separate tg column)
    blend = cfg.score == "eff"
    par_rows = "".join(
        f"<tr><td>{int(r['n_depth'])}</td><td>{score_of(r):.1f}</td>"
        + (f"<td>{r['tg_tps']:.1f}</td>" if blend else "")
        + f"<td>{esc(r['ngl'])}</td><td>{esc(r['kv_type'])}</td>"
        f"<td>{esc(r['ubatch'])}</td></tr>" for r in pareto)
    par_head = (f"<th>context</th><th>{cfg.score} t/s</th>"
                + ("<th>tg</th>" if blend else "")
                + "<th>ngl</th><th>kv</th><th>ubatch</th>")

    # all-runs table
    fcols = list(cfg.factors.keys())
    head = "".join(f"<th>{esc(c)}</th>" for c in
                   ["run", *fcols, *(["eff"] if blend else []), "tg", "pp", "status"])
    body = ""
    for r in sorted(rows, key=lambda r: score_of(r), reverse=True):
        cells = "".join(f"<td>{esc(r.get(c, ''))}</td>" for c in fcols)
        cls = "" if r["status"] == "OK" else " class=bad"
        body += (f"<tr{cls}><td>{esc(r.get('run_id',''))}</td>{cells}"
                 + (f"<td>{score_of(r):.1f}</td>" if blend else "")
                 + f"<td>{float(r['tg_tps']):.1f}</td>"
                 f"<td>{float(r['pp_tps']):.1f}</td><td>{esc(r['status'])}</td></tr>")

    meta = (f"{esc(cfg.model.name)} · {cfg.hw.get('n_layers','?')} layers · "
            f"{cfg.hw.get('phys')}c/{cfg.hw.get('logical')}t · "
            f"{cfg.hw.get('vram','?')} MiB VRAM · profile {esc(cfg.profile)} · "
            f"driver {esc(cfg.driver)} · array {esc(cfg.array)}")
    doc = f"""<!doctype html><meta charset=utf-8>
<title>llama-optimize — {esc(cfg.model.name)}</title>
<style>
:root{{color-scheme:light dark}}
body{{font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px;max-width:1100px;
 margin:auto;background:Canvas;color:CanvasText}}
h1{{margin:0 0 4px}} .meta{{color:#888;margin-bottom:20px;font-size:13px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:26px}}
.card{{flex:1;min-width:280px;border:1px solid #8883;border-radius:10px;padding:14px}}
.card h3{{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:#888}}
.big{{font-size:30px;font-weight:700}} .unit{{font-size:14px;color:#888;font-weight:400}}
.muted{{color:#888;font-size:13px}}
pre{{background:#8881;padding:10px;border-radius:8px;overflow-x:auto;font-size:12px;margin:10px 0 0}}
h2{{margin:26px 0 12px;font-size:16px;border-bottom:1px solid #8883;padding-bottom:6px}}
.fx{{margin-bottom:14px}} .fxh{{display:flex;justify-content:space-between;margin-bottom:4px}}
.lvl{{display:flex;align-items:center;gap:8px;margin:2px 0}}
.ll{{width:64px;text-align:right;font-size:12px;color:#888}}
.lv{{width:56px;font-size:12px;font-variant-numeric:tabular-nums}}
.bar{{height:14px;background:linear-gradient(90deg,#4a9,#6cf);border-radius:3px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{text-align:left;padding:4px 8px;border-bottom:1px solid #8882;font-variant-numeric:tabular-nums}}
th{{color:#888;font-weight:600}} tr.bad{{opacity:.5}}
.chart{{max-width:100%;height:auto;display:block;margin:6px 0 18px}}
</style>
<h1>llama-optimize report</h1>
<div class=meta>{meta}<br>objective: {"effective t/s" if blend else
 "generation t/s (pp reported, not scored)"} for a {esc(cfg.profile)} request
 ({cfg.n_prompt} prompt + {cfg.n_gen} gen tokens) — {len(ok)}/{len(rows)} configs OK</div>
<div class=cards>{card('Fastest (usable)', best)}{card(f'Balanced (≥{cfg.ctx_floor})', balanced)}{card('Max context', longest)}{probe_card}</div>
<h2>What matters (main effects, by impact)</h2>{''.join(fx_html)}
<h2>Pareto frontier (context vs {"effective" if blend else "generation"} t/s)</h2>
{_svg_pareto(rows, cfg.hw.get("vram", 0) if cfg.measure_vram else 0,
             ylabel=("effective t/s" if blend else "generation t/s"))}
<table><tr>{par_head}</tr>{par_rows}</table>
<h2>All runs</h2><table><tr>{head}</tr>{body}</table>
"""
    path.write_text(doc)
    print(f"\nwrote HTML report: {path}")


def taguchi_effects(cfg: Config, exp, rows: list[dict]):
    """Main-effects on the objective. Returns (optimal_levels, predicted_score)
    or (None, None)."""
    if exp is None:
        # direct one-way sweep: no array to analyze (and nothing to confirm —
        # every level was measured directly; the rows ARE the answer)
        print("\n(direct sweep: main-effects analysis and confirmation don't "
              "apply — every level was measured directly)")
        return None, None
    try:
        from taguchi import Analyzer
    except Exception:
        return None, None
    # Feed EVERY run: failed configs (OOM/TIMEOUT/ERROR) carry a 0 score as a
    # penalty. The analyzer requires a complete design, and scoring failures as 0
    # is the intended "failure is data" behaviour. (Caveat: a 0 from a timeout is
    # a censored value, so trust the Pareto for the pick and use main-effects only
    # to rank which factors matter — see README.)
    results = {int(r["run_id"]): score_of(r) for r in rows}
    n_failed = sum(1 for r in rows if r["status"] != "OK")
    if len(results) < 3:
        print("\n(not enough runs for main-effects analysis)")
        return None, None
    try:
        with Analyzer(exp, metric_name="eff_tps") as an:
            an.add_results_from_dict(results)
            kind = "effective" if cfg.score == "eff" else "generation"
            print(f"\n### Taguchi main effects ({kind} t/s, higher = better)")
            if n_failed:
                print(f"(note: {n_failed} failed run(s) scored as 0 t/s)")
            print(an.summary())
            opt = an.recommend_optimal(higher_is_better=True)
            print("Predicted-optimal levels:", opt)
            predicted = None
            try:
                p = an.predict_response(opt)
                if isinstance(p, (int, float)):
                    predicted = float(p)
                elif isinstance(p, dict):
                    nums = [v for v in p.values() if isinstance(v, (int, float))]
                    predicted = float(nums[0]) if nums else None
            except Exception:
                pass
            return opt, predicted
    except Exception as e:
        print(f"\n(main-effects analysis skipped: {e})")
        return None, None


# ---------------------------------------------------------------------------
# Stage-2: max-context probe
# ---------------------------------------------------------------------------
def probe_max_context(cfg: Config, base_f: dict, timeout: int, cap: int,
                      baseline_c: float | None = None):
    """Binary-search the largest n_depth that loads (no OOM) for base_f.

    Returns (max_depth, tg_tps_at_max) or None if even depth 0 fails.
    """
    def try_depth(d):
        wait_until_cool(baseline_c)   # each attempt measures; keep it comparable
        f = dict(base_f)
        f["n_depth"] = str(d)
        r = drive_one(cfg, f, timeout)
        return r["status"] == "OK", r

    good, _ = try_depth(0)
    if not good:
        return None
    good, r = try_depth(cap)
    if good:
        return cap, r["tg_tps"]

    lo, hi, best_tps = 0, cap, None
    while hi - lo > 2048:
        mid = ((lo + hi) // 2 // 1024) * 1024
        if mid <= lo:
            break
        good, r = try_depth(mid)
        if good:
            lo, best_tps = mid, r["tg_tps"]
        else:
            hi = mid
    return lo, best_tps


def run_probe_stage(cfg: Config, all_rows: list[dict], timeout: int,
                    thermal_baseline: float | None):
    """Run the max-context probe from the deepest-reaching OK config and package
    the result for the report: dict(row, depth, tg_tps, safe_ctx, at_cap) or
    None. The probe searches BEYOND the swept depth grid (up to the model's
    native context / --max-context), so its ceiling is an extrapolation the
    sweep never benchmarked — the report labels it as such."""
    ok = [r for r in all_rows if r["status"] == "OK"]
    if not ok:
        return None
    # Probe the config that reaches FURTHEST (max measured depth, then
    # fastest) — the memory-lightest good config — so this is the true
    # physical ceiling, not the fastest config's (which may fit less).
    base_row = max(ok, key=lambda r: (int(r["n_depth"]), score_of(r)))
    cap = cfg.hw.get("n_ctx_train") or 131072
    if cfg.max_depth:                          # --max-context caps the search
        cap = min(cap, cfg.max_depth)
    base = {k: base_row[k] for k in cfg.factors}
    print(f"\n### Max-context probe  (config: ngl={base_row.get('ngl')} "
          f"kv={base_row.get('kv_type')} ub={base_row.get('ubatch')}, "
          f"cap={cap})")
    res = probe_max_context(cfg, base, timeout, cap, thermal_baseline)
    if not res:
        print("  even depth 0 failed to load — check the config")
        return None
    depth, tps = res
    print(f"  largest context that loads: ~{depth} tokens"
          + (f"  (tg={tps:.1f} t/s there)" if tps else ""))
    # Turn the ceiling into a usable command: run just under it so there's
    # headroom for runtime allocation/fragmentation (living at the exact edge
    # risks an OOM mid-session), rounded to a tidy size.
    safe = max(4096, int(depth * 0.9) // 1024 * 1024)
    return {"row": base_row, "depth": depth, "tg_tps": tps,
            "safe_ctx": safe, "at_cap": depth >= cap}


# ---------------------------------------------------------------------------
# Stage-3: pick verification (medians beat lucky reps)
# ---------------------------------------------------------------------------
def config_key(cfg: Config, r: dict) -> str:
    """A config's identity across rows and sidecars: its factor values joined
    into one string (JSON-object-key friendly)."""
    return "|".join(f"{k}={r.get(k, '')}" for k in cfg.factors)


def verify_sidecar(results: Path) -> Path:
    """Where a sweep persists its pick-verification medians (JSON) so
    --report-only can re-apply them without a GPU."""
    return Path(str(results) + ".verify.json")


def verify_picks(cfg: Config, all_rows: list[dict], reps: int, timeout: int,
                 thermal_baseline: float | None) -> dict | None:
    """Re-measure each pick candidate `reps` extra times and settle on the
    median of all its measurements. Sweep rows are single measurements carrying
    thermal/run-to-run noise (±25% observed on hot GPUs), and picks made from
    exact scores crown whichever config got the lucky rep — medians make the
    headline numbers reproducible. Returns {config_key: {tg_tps, pp_tps, n,
    spread_pct}} for apply_verification(), or None if nothing to verify."""
    fastest, balanced, longest = pick_recommendations(cfg, all_rows)
    cands, seen = [], set()
    for r in (fastest, balanced, longest):
        if r is None:
            continue
        k = config_key(cfg, r)
        if k not in seen:
            seen.add(k)
            cands.append((k, r))
    if not cands:
        return None
    print(f"\n### Verifying picks  ({len(cands)} config(s) x {reps} extra "
          f"measurement(s); the median becomes the reported number)")
    out = {}
    for k, r in cands:
        f = {name: str(r[name]) for name in cfg.factors}
        tgs, pps = [float(r["tg_tps"])], [float(r["pp_tps"])]
        for i in range(reps):
            wait_until_cool(thermal_baseline)
            prefix = (f"verify ngl={f.get('ngl', '?')} "
                      f"depth={f.get('n_depth', '?')} [{i + 1}/{reps}]")
            res = with_ticker(prefix, timeout,
                              lambda ff=f: drive_one(cfg, ff, timeout))
            if res["status"] == "OK" and res["tg_tps"] > 0:
                tgs.append(res["tg_tps"])
                pps.append(res["pp_tps"])
            else:
                print(f"  (verify rep failed: {res['status']} — keeping the "
                      "other measurements)")
        med_tg = statistics.median(tgs)
        spread = (max(tgs) - min(tgs)) / med_tg * 100 if med_tg > 0 else 0.0
        print(f"  ngl={f.get('ngl', '?')} depth={f.get('n_depth', '?')}: "
              f"tg {float(r['tg_tps']):.1f} -> median {med_tg:.1f} t/s "
              f"({len(tgs)} measurements, spread {spread:.0f}%)")
        out[k] = {"tg_tps": med_tg, "pp_tps": statistics.median(pps),
                  "n": len(tgs), "spread_pct": round(spread, 1)}
    return out


def apply_verification(cfg: Config, rows: list[dict], verify: dict):
    """Overwrite each verified config's measured numbers with its median (and
    tag the row: verify_n / spread_pct) so the picks, Pareto, and report all
    speak the verified value. The results CSV keeps the raw single
    measurements; the medians live in the .verify.json sidecar."""
    for r in rows:
        v = verify.get(config_key(cfg, r))
        if not v:
            continue
        r["tg_tps"], r["pp_tps"] = v["tg_tps"], v["pp_tps"]
        r["eff_tps"] = objective_tps(cfg, r["pp_tps"], r["tg_tps"])
        r["verify_n"], r["spread_pct"] = v["n"], v["spread_pct"]


# ---------------------------------------------------------------------------
# Offline self-test (no GPU): exercises parsing / analysis / factor logic
# ---------------------------------------------------------------------------
def selftest() -> bool:
    try:
        # llama-bench JSON parsing (schema per llama-bench.cpp get_fields())
        sample = json.dumps([
            {"n_prompt": 512, "n_gen": 0, "n_depth": 0, "avg_ts": 123.4},
            {"n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": 45.6},
        ])
        assert parse_bench_json(sample) == (123.4, 45.6)
        assert parse_bench_json("not json") == (None, None)

        # OOM detection
        assert _OOM_PAT.search("ggml_backend_alloc failed: out of memory")
        assert _OOM_PAT.search("ROCm error: hipErrorOutOfMemory")

        # thread + depth level generation
        assert len(thread_levels(8, 16)) >= 3
        # depth_levels respects ctx_floor
        assert depth_levels(65536, floor=0)[0] == 0
        assert depth_levels(65536, floor=32768)[0] >= 32768
        assert depth_levels(65536, floor=32768)[-1] == 65536
        assert depth_levels(65536, floor=999999) == [65536]  # floor > top → single

        # factor-level generation
        assert five_levels_span(0, 64) == [0, 16, 32, 48, 64]
        assert ngl_levels(64)[0] == 0 and ngl_levels(64)[-1] == 64
        assert len(thread_levels(8, 16)) >= 3

        # --- conditional-factor core (docs/CONDITIONAL-FACTORS.md) ---
        # the real registry is valid
        assert validate_factor_registry() == [], validate_factor_registry()
        # is_active: unconditional always active; conditional gated on its gate
        assert is_active("ngl", {}) is True                  # unconditional
        assert is_active("does_not_exist", {}) is True       # unknown → active
        synth = {
            "G":     {"server": ("--g",), "kind": "cat", "translate": {"a": "a", "b": "b", "c": "c"}},
            "A":     {"server": ("--a",), "kind": "num"},                    # unconditional
            "Pa":    {"server": ("--pa",), "kind": "num", "active_when": ("G", {"a"})},
            "Pshare":{"kind": "num", "active_when": ("G", {"a", "b"}),
                      "flag_for": lambda v: (f"--p-{v}",)},
        }
        _saved = dict(FACTORS)
        FACTORS.clear(); FACTORS.update(synth)
        try:
            assert is_active("Pa", {"G": "a"}) is True
            assert is_active("Pa", {"G": "b"}) is False       # gate ≠ live value
            assert is_active("Pa", {}) is False               # gate absent
            assert is_active("Pshare", {"G": "b"}) is True
            assert is_active("Pshare", {"G": "c"}) is False
            assert active_factors(["A", "Pa", "Pshare"], {"G": "a"}) == ["A", "Pa", "Pshare"]
            assert active_factors(["A", "Pa", "Pshare"], {"G": "b"}) == ["A", "Pshare"]
            # flag_for resolves the variant-dependent spelling
            assert conditional_flags("Pshare", "b") == ("--p-b",)
            assert conditional_flags("Pa", "a") == ("--pa",)   # static fallback
            assert validate_factor_registry() == []            # synth is valid
            # validator catches: missing gate, bad live value, flag_for gap, cycle
            bad = {"X": {"kind": "num", "active_when": ("Nope", {"z"})}}
            assert validate_factor_registry(bad)
            badval = {"G": synth["G"], "X": {"server": ("--x",), "kind": "num",
                                             "active_when": ("G", {"zzz"})}}
            assert any("not levels" in e for e in validate_factor_registry(badval))
            badff = {"G": synth["G"], "X": {"kind": "num", "active_when": ("G", {"a"}),
                                            "flag_for": lambda v: ()}}
            assert any("non-empty" in e for e in validate_factor_registry(badff))
            cyc = {"G": {"server": ("--g",), "kind": "cat", "active_when": ("H", {"x"})},
                   "H": {"server": ("--h",), "kind": "cat", "active_when": ("G", {"y"})}}
            assert any("cycle" in e for e in validate_factor_registry(cyc))
        finally:
            FACTORS.clear(); FACTORS.update(_saved)

        # MoE metadata
        assert model_expert_count({"llama.expert_count": 8}) == 8
        assert model_expert_count({}) == 0

        # Pareto frontier (maximize depth and tg_tps)
        rows = [
            {"status": "OK", "n_depth": "0", "tg_tps": 50.0},
            {"status": "OK", "n_depth": "16384", "tg_tps": 40.0},
            {"status": "OK", "n_depth": "16384", "tg_tps": 30.0},  # dominated
            {"status": "OK", "n_depth": "4096", "tg_tps": 20.0},   # dominated
            {"status": "OOM", "n_depth": "65536", "tg_tps": 0.0},  # excluded
        ]
        depths = sorted(int(r["n_depth"]) for r in pareto_frontier(rows))
        assert depths == [0, 16384], depths

        # pareto SVG: x-axis zooms to the data range (like y) — a single-depth
        # run must not stack every point on the right edge (regression: a final
        # refinement pass at one depth drew all 25 dots at x=right-margin), and
        # every tick must land on-plot (rounding ticks up past xmax drew them
        # outside the viewBox).
        svg1 = _svg_pareto([{"status": "OK", "n_depth": "49152", "tg_tps": t}
                            for t in (10.0, 12.0, 14.0)])
        cxs = [float(m) for m in re.findall(r"circle cx='([\d.]+)'", svg1)]
        assert cxs and all(200 < c < 500 for c in cxs), cxs   # centered, not edge
        svg2 = _svg_pareto([{"status": "OK", "n_depth": d, "tg_tps": t} for d, t in
                            [("0", 50.0), ("16384", 40.0), ("65536", 20.0)]])
        ticks = [float(m) for m in
                 re.findall(r"<text x='([\d.-]+)' y='\d+' class='ax xt'", svg2)]
        assert ticks and all(62 <= t <= 664 for t in ticks), ticks  # on-plot

        # command builder: flag map, env split, batch clamp
        cfg = Config(model=Path("m.gguf"), llama_bench=Path("lb"), array="L25",
                     ctx_floor=16384, env_factor_names={"GGML_CUDA_FORCE_MMQ"})
        f = {"ngl": "64", "threads": "8", "kv_type": "q4_0", "ubatch": "2048",
             "n_depth": "0", "nkvo": "1", "poll": "50", "GGML_CUDA_FORCE_MMQ": "1"}
        cmd = bench_command(cfg, f)
        assert "-nkvo" in cmd and "--poll" in cmd
        assert "-ctk" in cmd and "-ctv" in cmd
        assert "GGML_CUDA_FORCE_MMQ" not in " ".join(cmd)  # env not on cmdline
        assert run_env(cfg, f)["GGML_CUDA_FORCE_MMQ"] == "1"
        cmd2 = bench_command(cfg, {"ubatch": "2048", "batch": "512"})  # clamp b>=ub
        assert cmd2[cmd2.index("-b") + 1] == "2048"

        # effective throughput objective
        assert effective_tps(512, 256, 0, 100) == 0.0
        assert abs(effective_tps(512, 256, 1000.0, 100.0)
                   - 768 / (512 / 1000 + 256 / 100)) < 1e-6
        assert set(PROFILES) == {"single", "agents", "multi"}

        # scoring: tg-only by default — a huge pp can't mask slow decode;
        # --score eff restores the blended request throughput
        cfg_o = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192, n_prompt=512, n_gen=256)
        assert cfg_o.score == "tg"
        assert objective_tps(cfg_o, 5000.0, 8.4) == 8.4
        assert objective_tps(cfg_o, 0.0, 100.0) == 100.0  # pp glitch ≠ failed run
        assert objective_tps(cfg_o, 100.0, 0.0) == 0.0
        cfg_o.score = "eff"
        assert abs(objective_tps(cfg_o, 1000.0, 100.0)
                   - effective_tps(512, 256, 1000.0, 100.0)) < 1e-9
        assert objective_tps(cfg_o, 0.0, 100.0) == 0.0    # blend needs both

        # --diff and --merge-results dedup (temp CSVs; offline, no binding)
        with tempfile.TemporaryDirectory() as td:
            cols = ["run_id", "ngl", "n_depth", "pp_tps", "tg_tps", "eff_tps",
                    "status", "secs", "temp_c"]

            def wcsv(name, rws):
                p = Path(td) / name
                with open(p, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=cols)
                    w.writeheader()
                    w.writerows(rws)
                return p

            def rrow(i, ngl, tg, status="OK"):
                return {"run_id": i, "ngl": ngl, "n_depth": "0", "pp_tps": 500.0,
                        "tg_tps": tg, "eff_tps": tg, "status": status,
                        "secs": 5, "temp_c": ""}

            old = wcsv("run.pass1.csv", [rrow(1, "32", 40.0), rrow(2, "16", 30.0),
                                         rrow(3, "48", 0.0, "OOM")])
            new = wcsv("new.csv", [rrow(1, "32", 44.0), rrow(2, "16", 29.0),
                                   rrow(3, "48", 50.0)])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                s = diff_results(old, new)
                assert s["matched"] == 3 and s["compared"] == 2
                assert s["status_changes"] == 1             # ngl=48: OOM -> OK
                assert s["old_winner_still_wins"] is False  # 48 took the lead
                assert abs(s["new_best_tg"] - 50.0) < 1e-9
                assert diff_results(old, Path(td) / "missing.csv") is None

                # merge dedup: a merged row is kept only if it beats every
                # known measurement of that exact config
                cfg_r = Config(model=Path("m"), llama_bench=Path("b"),
                               array="auto", ctx_floor=8192)
                cfg_r.factors = {"ngl": ["16", "32", "48"], "n_depth": ["0"]}
                out = merge_result_rows(cfg_r, [rrow(9, "32", 42.0)], [old])
                assert len(out) == 3                    # 32 deduped; 16+48 added
                got = {r["ngl"]: r for r in out}
                assert got["32"]["tg_tps"] == 42.0      # current row won the dup
                assert got["16"]["run_id"] == "pass1:2"  # tagged with its pass
                assert got["48"]["status"] == "OOM"     # failures still carried
                # a FASTER earlier measurement does get in (never lose a best)
                out = merge_result_rows(cfg_r, [rrow(9, "32", 39.0)], [old])
                assert {r["tg_tps"] for r in out
                        if r["ngl"] == "32"} == {39.0, 40.0}
                # no merge files: rows pass through untouched (no dedup of the
                # array's own intentional replicates)
                reps = [rrow(1, "32", 40.0), rrow(2, "32", 41.0)]
                assert merge_result_rows(cfg_r, reps, []) is reps
            assert "NEW winner: ngl=48" in buf.getvalue()

        # recommended -c never exceeds the verified footprint (regression:
        # a row measured at depth 49152 must NOT be emitted as -c 65536).
        cfgc = Config(model=Path("m.gguf"), llama_bench=Path("lb"), array="L25",
                      n_prompt=512, n_gen=256, ctx_floor=8192)
        assert recommended_ctx(cfgc, {"n_depth": "49152"}) == 50176
        for d in (0, 4096, 49152, 131072):
            ctx = recommended_ctx(cfgc, {"n_depth": str(d)})
            assert ctx <= cfgc.n_prompt + d + cfgc.n_gen + 256      # not inflated
            assert ctx >= d + cfgc.n_prompt + cfgc.n_gen            # covers request
            assert ctx % 256 == 0
        # parallel multiplies the verified session context
        cfgp = Config(model=Path("m.gguf"), llama_bench=Path("lb"), array="L25",
                      ctx_floor=8192, n_prompt=512, n_gen=256, parallel=4)
        assert recommended_ctx(cfgp, {"n_depth": "8192"}) == (512 + 8192 + 256 + 256) * 4

        # usable floor on the emitted -c: raised to ctx_floor when (and only as
        # far as) a sibling verified deeper; explicit lower floors are honored;
        # with no deeper evidence the row's own footprint stands, plus a note.
        assert recommended_ctx(cfgc, {"n_depth": "0"}, verified_depth=49152) == 8192
        assert recommended_ctx(cfgc, {"n_depth": "0"}, verified_depth=4096) == 5120
        assert recommended_ctx(cfgc, {"n_depth": "0"}) == 1024
        cfg_lo = Config(model=Path("m.gguf"), llama_bench=Path("lb"), array="L25",
                        n_prompt=512, n_gen=256, ctx_floor=2048)
        assert recommended_ctx(cfg_lo, {"n_depth": "0"}, verified_depth=49152) == 2048
        assert ctx_floor_note(cfgc, {"n_depth": "0"}, 1024)           # capped => note
        assert ctx_floor_note(cfgc, {"n_depth": "0"}, 8192) is None   # floor met

        # verified_depth_of: deepest OK sibling sharing the launch factors
        cfg_v = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192)
        cfg_v.factors = {"ngl": ["60", "99"], "n_depth": ["0", "16384", "49152"]}
        rows_v = [{"status": "OK", "ngl": "60", "n_depth": "0", "eff_tps": 40.},
                  {"status": "OK", "ngl": "60", "n_depth": "49152", "eff_tps": 30.},
                  {"status": "OOM", "ngl": "60", "n_depth": "16384", "eff_tps": 0.},
                  {"status": "OK", "ngl": "99", "n_depth": "0", "eff_tps": 50.}]
        assert verified_depth_of(cfg_v, rows_v, rows_v[0]) == 49152
        assert verified_depth_of(cfg_v, rows_v, rows_v[3]) == 0   # no deep sibling

        # FASTEST = fastest *usable*: the raw-fastest config (ngl 99, never
        # verified past depth 0) is skipped for the fastest one that holds the
        # floor — no bench-number chasing; falls back when nothing qualifies.
        fast, bal, lng = pick_recommendations(cfg_v, rows_v)
        assert fast is rows_v[0] and bal is rows_v[1] and lng is rows_v[1]
        f2, b2, _ = pick_recommendations(cfg_v, [rows_v[3]])
        assert f2 is rows_v[3] and b2 is None
        assert pick_recommendations(cfg_v, []) == (None, None, None)

        # thermal wait-and-watch: no baseline => immediate no-op (never blocks)
        assert wait_until_cool(None) is None
        # a RISING temperature (post-run heat soak) is not a plateau (regression:
        # `prev - t < 0.5` was true for negatives, exiting at the hottest moment);
        # a genuine stall (cooling < 0.5°C/poll) or reaching baseline+band settles.
        real_temp, calls = gpu_temp_c, []

        def fake_temp(seq):
            it = iter(seq)
            return lambda: (calls.append(1), next(it, seq[-1]))[1]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                globals()["gpu_temp_c"] = fake_temp([60.0, 62.0, 61.9, 55.0])
                wait_until_cool(40.0, band=5.0, cap_s=99, poll_s=0)
                assert len(calls) == 3   # rode out the rise; exited on the stall
                calls.clear()
                globals()["gpu_temp_c"] = fake_temp([60.0, 50.0, 44.0])
                wait_until_cool(40.0, band=5.0, cap_s=99, poll_s=0)
                assert len(calls) == 3   # cooled to <= baseline+band and settled
        finally:
            globals()["gpu_temp_c"] = real_temp

        # q8_0 downgrade hint fires only for KV heavier than the q8_0 floor
        assert kv_downgrade_hint({"kv_type": "f16"})
        assert kv_downgrade_hint({"kv_type": "f32"})
        assert kv_downgrade_hint({"kv_type": "q8_0"}) is None
        assert kv_downgrade_hint({"kv_type": "q4_0"}) is None

        # realistic prompt: varied text of ~n*4 chars, not a repeated token
        assert len(_realistic_prompt(100)) == 400
        assert len(set(_realistic_prompt(200))) > 20  # genuinely varied

        # array auto-selection by factor levels
        assert choose_array({f"f{i}": ["a", "b", "c", "d", "e"] for i in range(5)}) == "L25"
        assert choose_array({f"f{i}": ["a", "b", "c"] for i in range(6)}) == "L27"
        assert choose_array({f"f{i}": ["0", "1"] for i in range(7)}) == "L8"
        # mixed levels ride the largest base's array (the binding has no L18):
        # a 2-level factor among 3-level ones maps onto a 3-level column
        assert choose_array({"a": ["0", "1"], "b": ["x", "y", "z"],
                             "c": ["p", "q", "r"]}) == "L9"
        # 7 varying factors overflow L25's 6 columns -> the 125-run array
        assert choose_array({f"f{i}": ["a", "b", "c", "d", "e"]
                             for i in range(7)}) == "L125"
        # a fixed (1-level) factor among 3-level ones still picks a 3-level array
        assert choose_array({"t": ["8"], "a": ["1", "2", "3"],
                             "b": ["1", "2", "3"]}) == "L9"
        # settled constants don't count: one lone varying factor => no array
        # (direct sweep), not a 25-run L25 to replicate 5 configs 5×.
        assert choose_array({"ngl": ["56", "57", "58", "59", "60"],
                             "d": ["49152"], "t": ["8"], "kv": ["f16"]}) is None
        assert choose_array({}) is None

        # generate_runs: <=1 active factor enumerates directly (N runs, not N×N),
        # constants attached to every row; needs no array binding.
        exp0, runs0 = generate_runs({"ngl": ["56", "58", "60"], "d": ["49152"],
                                     "kv": ["f16"]}, "auto")
        assert exp0 is None and len(runs0) == 3            # one run per ngl level
        assert all(r["factors"]["d"] == "49152" and r["factors"]["kv"] == "f16"
                   for r in runs0)                         # constants ride along
        assert [r["factors"]["ngl"] for r in runs0] == ["56", "58", "60"]
        _, runs1 = generate_runs({"ngl": ["60"], "kv": ["f16"]}, "auto")
        assert len(runs1) == 1                             # 0 active => single config

        # refinement: settle the flat factor, refine the high-impact one
        assert refine_numeric([20, 40, 60], 60) == ["40", "45", "50", "55", "60"]
        cfg_r = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192)
        cfg_r.factors = {"ngl": ["20", "40", "60"], "kv_type": ["f16", "q8_0"]}
        rr = [{"status": "OK", "ngl": a, "kv_type": k, "eff_tps": e} for a, k, e in
              [("20", "f16", 10.), ("40", "q8_0", 50.), ("60", "f16", 90.),
               ("20", "q8_0", 11.), ("60", "q8_0", 89.), ("40", "f16", 49.)]]
        ref = refine_factors(cfg_r, rr)
        assert ref["kv_type"] == ["q8_0"]          # flat factor settled at winner
        assert ref["ngl"] == ["40", "45", "50", "55", "60"]  # refined near best (60)
        # n_depth is the tradeoff axis: kept spread across passes, never settled,
        # so the final pass still maps the whole speed/context curve.
        cfg_d = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192)
        cfg_d.factors = {"ngl": ["20", "40", "60"],
                         "n_depth": ["0", "16384", "32768", "49152", "65536"]}
        rd = [{"status": "OK", "ngl": "60", "n_depth": d, "eff_tps": e}
              for d, e in [("0", 30.), ("16384", 25.), ("32768", 20.),
                           ("49152", 15.), ("65536", 10.)]]
        assert refine_factors(cfg_d, rd)["n_depth"] == cfg_d.factors["n_depth"]

        # unified registry: driver mapping, server-only, -ot translation, bools
        assert is_server_only("spec_p_min") and not is_server_only("ngl")
        assert FACTORS["threads_batch"]["bench"] is None      # server-only flag
        cfg_s = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192, driver="server")
        assert factor_flags(cfg_s, {"ot": "none"}, "bench", 512) == []   # none omits
        assert factor_flags(cfg_s, {"ot": "exps_cpu"}, "bench", 512)[0][0] == "-ot"
        assert factor_flags(cfg_s, {"fa": "0"}, "bench", 512) == [["-fa", "0"]]
        assert factor_flags(cfg_s, {"nkvo": "1"}, "server", 512) == [["-nkvo"]]  # bare
        assert factor_flags(cfg_s, {"nkvo": "1"}, "bench", 512) == [["-nkvo", "1"]]

        # MTP as a swept factor: on/off via translate ("" omits), server-only
        assert factor_flags(cfg_s, {"mtp": "1"}, "server", 512) == \
            [["--spec-type", "draft-mtp"]]
        assert factor_flags(cfg_s, {"mtp": "0"}, "server", 512) == []
        assert factor_flags(cfg_s, {"mtp": "1"}, "bench", 512) == []
        cfg_m = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192, driver="server",
                       hw={"phys": 8, "logical": 16, "n_layers": 32,
                           "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 1})
        sa = build_server_args(cfg_m, {"mtp": "0", "ubatch": "512"}, 8080, 4096)
        assert "--spec-type" not in sa      # swept off: automatic flag yields
        sa = build_server_args(cfg_m, {"ubatch": "512"}, 8080, 4096)
        assert "--spec-type" in sa and "draft-mtp" in sa  # fixed on if not swept
        # default factor set: nkvo/poll/batch always; ot for dense / ncmoe for
        # MoE; threads_batch + the MTP surface only on the server driver;
        # numa only on a multi-node box
        fs = build_factors(cfg_m)
        assert all(k in fs for k in ("nkvo", "poll", "batch", "threads_batch",
                                     "mtp", "spec_n_max", "spec_n_min",
                                     "spec_p_min", "spec_p_split"))
        assert "ot" in fs and "ncmoe" not in fs            # dense
        assert "numa" not in fs                            # single NUMA node
        cfg_m.hw["numa_nodes"] = 2
        assert "numa" in build_factors(cfg_m)
        cfg_m.hw["numa_nodes"] = 1
        cfg_m.driver = "bench"
        fs = build_factors(cfg_m)
        assert "nkvo" in fs and "poll" in fs and "batch" in fs
        assert all(k not in fs for k in ("mtp", "spec_n_max", "spec_n_min",
                                         "spec_p_min", "spec_p_split",
                                         "threads_batch"))  # server-only

        # ngram as a swept factor: translate maps variant names, "none" omits
        assert factor_flags(cfg_s, {"ngram": "ngram-mod"}, "server", 512) == \
            [["--spec-type", "ngram-mod"]]
        assert factor_flags(cfg_s, {"ngram": "none"}, "server", 512) == []
        assert factor_flags(cfg_s, {"ngram": "ngram-simple"}, "server", 512) == \
            [["--spec-type", "ngram-simple"]]
        assert factor_flags(cfg_s, {"ngram": "ngram-mod"}, "bench", 512) == []
        # all ngram factors are server-only
        for nf in ("ngram", "ngram_size_n", "ngram_size_m", "ngram_min_hits",
                   "ngram_mod_n_match", "ngram_mod_n_min", "ngram_mod_n_max"):
            assert FACTORS[nf].get("server_only"), f"{nf} should be server_only"

        # conditional ngram knobs: emitted only for their active variant (I2), and
        # the collapsed shared knob resolves the variant-specific flag via flag_for.
        gated = _flat(factor_flags(cfg_s, {"ngram": "ngram-mod",
                                           "ngram_mod_n_match": "16",
                                           "ngram_size_n": "8"}, "server", 512))
        assert "--spec-ngram-mod-n-match" in gated, f"got {gated}"
        assert "--spec-ngram-simple-size-n" not in gated, f"got {gated}"
        # same shared knob, different variant → different flag spelling
        for variant in ("ngram-simple", "ngram-map-k", "ngram-map-k4v"):
            g = _flat(factor_flags(cfg_s, {"ngram": variant, "ngram_size_m": "32",
                                           "ngram_mod_n_max": "64"}, "server", 512))
            assert f"--spec-{variant}-size-m" in g, f"{variant}: {g}"
            assert "--spec-ngram-mod-n-max" not in g          # mod knob inactive here
        # ngram=none → variant flag omitted AND every conditional knob suppressed
        assert factor_flags(cfg_s, {"ngram": "none", "ngram_mod_n_match": "16",
                                    "ngram_size_n": "8"}, "server", 512) == []

        # ngram screen build: the gate only (no conditional children — those enter
        # a variant's tuning stage), and NOT spec_n_max (a draft-model knob).
        cfg_n = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                       ctx_floor=8192, driver="server", ngram=True,
                       hw={"phys": 8, "logical": 16, "n_layers": 32,
                           "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 0})
        nfs = build_factors(cfg_n)
        assert "ngram" in nfs and len(nfs["ngram"]) == 5, f"ngram not in {list(nfs)}"
        assert "spec_n_max" not in nfs, "spec_n_max must not be an ngram factor"
        assert not any(k.startswith("ngram_") and k != "ngram" for k in nfs), \
            "screen build must exclude conditional children"
        # a pinned variant (tuning stage) includes ONLY its own knobs
        vf_mod = build_factors(replace(cfg_n, ngram_type="ngram-mod"))
        assert vf_mod["ngram"] == ["ngram-mod"]
        assert "ngram_mod_n_max" in vf_mod and "ngram_size_n" not in vf_mod
        vf_simple = build_factors(replace(cfg_n, ngram_type="ngram-simple"))
        assert "ngram_size_n" in vf_simple and "ngram_mod_n_max" not in vf_simple

        # I3: a conditional knob is scored only over rows where it was active.
        # ngram_size_n really matters within ngram-simple (best at "24"); rows on
        # a different variant carry a DECOY best ("4") that must be ignored.
        cond_rows = (
            [{"status": "OK", "ngram": "ngram-simple", "ngram_size_n": sz,
              "pp_tps": 100.0, "tg_tps": v, "n_prompt": 0, "n_gen": 128}
             for sz, v in (("4", 10.0), ("12", 20.0), ("24", 30.0))] +
            [{"status": "OK", "ngram": "ngram-mod", "ngram_size_n": sz,
              "pp_tps": 100.0, "tg_tps": v, "n_prompt": 0, "n_gen": 128}
             for sz, v in (("4", 90.0), ("12", 5.0), ("24", 5.0))]
        )
        cond_means = factor_level_means(cond_rows, "ngram_size_n")
        assert max(cond_means, key=cond_means.get) == "24", cond_means   # true best
        assert "90.0" not in [f"{v}" for v in cond_means.values()]        # decoy row excluded

        # --- stage planner (docs/CONDITIONAL-FACTORS.md) ---
        # feed the planner the FULL factor set (screen gate + every child) it would
        # decompose; build_factors only ever emits one stage's worth at a time.
        full = dict(nfs); full.update(NGRAM_MAP_LEVELS); full.update(NGRAM_MOD_LEVELS)
        stages = plan_stages(full)
        assert stages[0]["name"] == "screen"
        # screen sweeps the gate at full spread but excludes conditional children
        assert stages[0]["factors"]["ngram"] == full["ngram"]
        assert not any(_active_when(f) for f in stages[0]["factors"])
        # one tuning stage per variant that has children (3 map + mod)
        tune = {s["value"]: s for s in stages if s["gate"] == "ngram"}
        assert set(tune) == {"ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod"}
        # a map variant tunes the 3 collapsed knobs; mod tunes its own 3; gate pinned
        assert set(tune["ngram-simple"]["factors"]) == \
            {"ngram", "ngram_size_n", "ngram_size_m", "ngram_min_hits"}
        assert set(tune["ngram-mod"]["factors"]) == \
            {"ngram", "ngram_mod_n_match", "ngram_mod_n_min", "ngram_mod_n_max"}
        assert tune["ngram-simple"]["factors"]["ngram"] == ["ngram-simple"]
        # I1 liveness: every factor in every stage is active under the stage's pin
        for s in stages:
            assert all(is_active(f, s["pin"]) for f in s["factors"]), s["name"]
        # F2: each tuning stage is small (gate + 3 knobs = 4 factors → fits L25),
        # versus the flat design's L125
        assert all(len(s["factors"]) <= 4 for s in stages if s["gate"])
        # a subset factor set plans just the screen + the one live variant's stage
        ps = plan_stages({"ngram": ["none", "ngram-mod"],
                          "ngram_mod_n_max": ["32", "64"], "ngl": ["0", "64"]})
        assert [s["name"] for s in ps] == ["screen", "tune:ngram=ngram-mod"]
        assert "ngl" in ps[0]["factors"] and "ngram_mod_n_max" not in ps[0]["factors"]

        # --- ngram staging helpers (run_ngram_stages) ---
        screen_rows = [
            {"status": "OK", "ngram": "ngram-mod", "pp_tps": 100.0, "tg_tps": 30.0,
             "ngl": "64", "n_depth": "0", "n_prompt": 0, "n_gen": 128},
            {"status": "OK", "ngram": "ngram-simple", "pp_tps": 100.0, "tg_tps": 25.0,
             "ngl": "64", "n_depth": "0", "n_prompt": 0, "n_gen": 128},
            {"status": "OK", "ngram": "none", "pp_tps": 100.0, "tg_tps": 20.0,
             "ngl": "32", "n_depth": "0", "n_prompt": 0, "n_gen": 128},
            {"status": "OOM", "ngram": "ngram-map-k", "pp_tps": 0.0, "tg_tps": 0.0,
             "ngl": "64", "n_depth": "0", "n_prompt": 0, "n_gen": 128},
        ]
        ranked = rank_gate_values(screen_rows, "ngram")
        assert [v for v, _ in ranked][:2] == ["ngram-mod", "ngram-simple"]  # best first
        assert keep_top_gate_values(ranked, 2) == ["ngram-mod", "ngram-simple"]
        assert keep_top_gate_values(ranked, 1) == ["ngram-mod"]             # --ngram-fast
        assert "none" not in keep_top_gate_values(ranked, 5)                # never tune "off"
        held = screen_base_winners(cfg_n, {"ngl": ["32", "64"], "n_depth": ["0", "4096"],
                                           "ngram": nfs["ngram"]}, screen_rows)
        assert "ngram" not in held                                         # gate pinned separately
        assert held["ngl"] == ["64"]                                       # settled at winner
        assert held["n_depth"] == ["0", "4096"]                            # tradeoff axis kept
        # child argv carries --ngram-type for a pinned tuning stage
        class _A:  # minimal args stand-in for build_child_argv
            model = Path("m"); no_mtp = no_shuffle = no_thermal_wait = False
            thermal_baseline = seed = max_depth = None; timeout = 60; cooldown = 0
            confirm = full_ = html = None; no_probe = False; verify_picks = 2
            merge_results = []
        _av = build_child_argv(_A, replace(cfg_n, ngram_type="ngram-mod"),
                               {"ngl": ["64"]}, Path("r.csv"), False, [])
        assert "--ngram-type" in _av and _av[_av.index("--ngram-type") + 1] == "ngram-mod"

        # spec-type merging: mtp + ngram both present → comma-separated
        merged = _merge_spec_type_parts(["--spec-type draft-mtp", "--spec-type ngram-mod"])
        assert merged == ["--spec-type draft-mtp,ngram-mod"], f"got {merged}"
        merged = _merge_spec_type_args(["--spec-type", "draft-mtp", "--spec-type", "ngram-mod"])
        assert merged == ["--spec-type", "draft-mtp,ngram-mod"], f"got {merged}"
        # only one spec-type → unchanged
        assert _merge_spec_type_parts(["--spec-type draft-mtp"]) == ["--spec-type draft-mtp"]
        assert _merge_spec_type_args(["--spec-type", "draft-mtp"]) == ["--spec-type", "draft-mtp"]
        # no spec-type → no change
        assert _merge_spec_type_parts(["-ngl 64"]) == ["-ngl 64"]
        assert _merge_spec_type_args(["-ngl", "64"]) == ["-ngl", "64"]

        # build_server_args with ngram: when ngram swept off, no spec-type;
        cfg_ns = Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                        ctx_floor=8192, driver="server",
                        hw={"phys": 8, "logical": 16, "n_layers": 32,
                            "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 0},
                        ngram=True)
        sa = build_server_args(cfg_ns, {"ngram": "none", "ubatch": "512"}, 8080, 4096)
        assert "--spec-type" not in sa, f"ngram=none should suppress spec-type, got {sa}"
        sa = build_server_args(cfg_ns, {"ubatch": "512"}, 8080, 4096)
        # ngram fixed on (not swept): should emit --spec-type ngram-mod
        assert "--spec-type" in sa
        idx = sa.index("--spec-type")
        val = sa[idx + 1]
        assert val == "ngram-mod", f"expected ngram-mod, got {val}"
        cfg_m.hw["n_experts"] = 64
        fs = build_factors(cfg_m)
        assert "ncmoe" in fs and "ot" not in fs            # MoE

        # KV quality floor
        assert kv_at_or_above(["f16", "q8_0", "q5_1", "q4_0"], "q8_0") == ["f16", "q8_0"]
        assert kv_at_or_above(["f16", "q8_0", "q4_0"], "any") == ["f16", "q8_0", "q4_0"]
        assert kv_at_or_above(["q4_0"], "q8_0") == ["q8_0"]  # never empty -> floor

        # morris analyze table parsing
        mtxt = ("Factor  mu*  sigma  note\n------  ----  -----  ----\n"
                "ubatch  96  0\nngl  32  0.001  \n\nRanked by mu* ...\n")
        assert parse_morris_analyze(mtxt) == [("ubatch", 96.0, 0.0), ("ngl", 32.0, 0.001)]

        # ttft note: prefill-cost estimate on emitted commands
        n = ttft_note({"pp_tps": 100.0}, 235520)
        assert n and "39m" in n and "8k prompt" in n
        assert "8k prompt" not in ttft_note({"pp_tps": 100.0}, 8192)
        assert ttft_note({"pp_tps": 0}, 8192) is None

        # sidecar naming (probe / verify persist next to the results CSV)
        assert probe_sidecar(Path("r/x.csv")).name == "x.csv.probe.json"
        assert verify_sidecar(Path("r/x.csv")).name == "x.csv.verify.json"

        # pick verification: medians overwrite the row, keyed by config
        cfg_pv = Config(model=Path("m.gguf"), llama_bench=Path("lb"), array="L25",
                        ctx_floor=8192,
                        factors={"ngl": ["58"], "n_depth": ["32768"],
                                 "threads": ["5"], "kv_type": ["q8_0"],
                                 "ubatch": ["512"]})
        pv_rows = [{"run_id": "1", "ngl": "58", "n_depth": "32768",
                    "threads": "5", "kv_type": "q8_0", "ubatch": "512",
                    "status": "OK", "pp_tps": 120.0, "tg_tps": 10.8,
                    "eff_tps": 10.8}]
        apply_verification(cfg_pv, pv_rows,
                           {config_key(cfg_pv, pv_rows[0]):
                            {"tg_tps": 8.9, "pp_tps": 118.0,
                             "n": 3, "spread_pct": 25.0}})
        assert pv_rows[0]["tg_tps"] == 8.9 and pv_rows[0]["verify_n"] == 3
        assert pv_rows[0]["eff_tps"] == 8.9      # re-scored under --score tg

        # probe + verification render in the terminal report and the HTML card
        pv_probe = {"row": pv_rows[0], "depth": 262144, "tg_tps": 6.6,
                    "safe_ctx": 235520, "at_cap": True}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report(cfg_pv, pv_rows, pv_probe)
        rout = buf.getvalue()
        assert "PROBED CEILING" in rout and "-c 235520" in rout
        assert "median of 3 measurements" in rout and "prefill cost" in rout
        with tempfile.TemporaryDirectory() as td:
            hp = Path(td) / "r.html"
            with contextlib.redirect_stdout(io.StringIO()):
                write_html_report(cfg_pv, pv_rows, hp, pv_probe)
                h = hp.read_text()
                assert "Probed ceiling" in h and "262,144" in h
                write_html_report(cfg_pv, pv_rows, hp)
            assert "Probed ceiling" not in hp.read_text()
    except AssertionError as e:
        print(f"selftest FAILED: {e}")
        return False
    print("selftest: all checks passed")
    return True


# ---------------------------------------------------------------------------
# Iterative refinement: run N passes, each a subprocess of the single-pass tool,
# refining the factor set between passes. Keeps the tested execution path intact.
# ---------------------------------------------------------------------------
def load_results_csv(path: Path, factors: dict) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            for c in ("pp_tps", "tg_tps", "eff_tps", "secs"):
                try:
                    r[c] = float(r[c])
                except (KeyError, ValueError, TypeError):
                    r[c] = 0.0
            rows.append(r)
    return rows


# results-CSV columns that are measurements/bookkeeping; everything else in a
# header is a factor column (must match the writer's `cols` in main)
RESULT_COLS = {"run_id", "pp_tps", "tg_tps", "eff_tps", "status", "secs",
               "temp_c", "vram_mib"}


def merge_result_rows(cfg: Config, rows: list[dict],
                      merge_paths: list[Path]) -> list[dict]:
    """Fold rows from earlier results CSVs (--merge-results, e.g. previous
    --iterate passes) into the row set the report/picks/Pareto/probe see, so
    every config ever measured is considered — the final answer can then never
    be worse than an earlier pass's best. A merged row is kept only if it beats
    every already-known measurement of that exact config (same value for every
    factor), so re-measured configs don't pad the Pareto/all-runs tables with
    duplicates. Main-effects/confirm stay on this run's own rows (the analyzer
    needs the balanced array structure; merged rows would skew it)."""
    if not merge_paths:
        return rows

    def key(r):
        return tuple(str(r.get(k, "")) for k in cfg.factors)

    known: dict[tuple, float] = {}
    for r in rows:
        k = key(r)
        known[k] = max(known.get(k, 0.0), score_of(r))
    merged: dict[tuple, dict] = {}
    for mi, mpath in enumerate(merge_paths, 1):
        m = re.search(r"\.(pass\d+)$", mpath.stem)
        tag = m.group(1) if m else f"merge{mi}"
        folded = 0
        for r in load_results_csv(mpath, cfg.factors):
            if not all(str(r.get(k, "")) != "" for k in cfg.factors):
                continue               # foreign CSV missing a factor column
            # re-score under the CURRENT --score mode (same rule as --resume)
            r["eff_tps"] = objective_tps(cfg, r["pp_tps"], r["tg_tps"])
            r["run_id"] = f"{tag}:{r.get('run_id', '')}"
            k = key(r)
            if k in known and known[k] >= score_of(r):
                continue               # this config already measured >= as fast
            if k in merged and score_of(merged[k]) >= score_of(r):
                continue
            merged[k] = r
            folded += 1
        print(f"merged {folded} row(s) from {mpath.name} into the report"
              if folded else
              f"merge: no new rows from {mpath.name} (duplicates or unusable)")
    return rows + list(merged.values())


def probe_sidecar(results: Path) -> Path:
    """Where a sweep persists its max-context probe result (JSON) so
    --report-only can re-show the PROBED CEILING section without a GPU."""
    return Path(str(results) + ".probe.json")


def report_from_results(cfg: Config, args, ap):
    """--report-only: rebuild the terminal report (and --html) from a finished
    sweep's results CSV — no GPU and no llama.cpp binaries. The factor columns
    are whatever the CSV header carries beyond the measurement columns
    (RESULT_COLS), so the picks/Pareto see exactly what the sweep recorded;
    --merge-results folds in further CSVs (e.g. other --iterate passes). The
    PROBED CEILING section appears if the sweep saved a probe sidecar
    (<results>.probe.json); probing anew needs the GPU."""
    path = args.results
    if not path.exists():
        ap.error(f"--report-only: results file not found: {path}")
    rows = load_results_csv(path, {})
    if not rows:
        ap.error(f"--report-only: no rows in {path}")
    names = [c for c in rows[0] if c not in RESULT_COLS]
    if not names:
        ap.error(f"--report-only: no factor columns in {path}")

    def lvl_key(v):
        try:
            return (0, int(v), "")
        except ValueError:
            return (1, 0, v)

    cfg.factors = {n: sorted({str(r.get(n, "")) for r in rows}, key=lvl_key)
                   for n in names}
    cfg.measure_vram = "vram_mib" in rows[0]
    for r in rows:      # re-score under the CURRENT --score mode
        r["eff_tps"] = objective_tps(cfg, r["pp_tps"], r["tg_tps"])
    all_rows = merge_result_rows(cfg, rows, args.merge_results)
    vf = verify_sidecar(path)
    if vf.exists():
        try:
            apply_verification(cfg, all_rows, json.loads(vf.read_text()))
            print(f"applied pick-verification medians from {vf.name}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"(ignoring unreadable {vf.name}: {e})")
    probe = None
    pf = probe_sidecar(path)
    if pf.exists():
        try:
            probe = json.loads(pf.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"(ignoring unreadable {pf.name}: {e})")
    print(f"report-only: {len(rows)} row(s) from {path} (no GPU used)"
          + ("" if probe else " — no probe sidecar, PROBED CEILING omitted"))
    report(cfg, all_rows, probe)
    if args.html:
        write_html_report(cfg, all_rows, args.html, probe)


def diff_results(old_path: Path, new_path: Path) -> dict | None:
    """Compare two results CSVs of the same factor space (llama.cpp upgrade,
    driver update, quant swap): match configs on the factor columns both files
    share, then report per-config deltas, status changes, and whether the old
    winner still wins. Compares raw tg_tps (pp alongside) — eff_tps depends on
    the --score mode and request shape a sweep ran with, so it isn't comparable
    across files. Returns a summary dict (for the selftest), None if the files
    can't be compared. Needs no model, GPU, or submodule."""
    def load(path):
        rows = load_results_csv(path, {})
        cols = [c for c in (rows[0].keys() if rows else []) if c not in RESULT_COLS]
        return rows, cols

    old_rows, old_cols = load(old_path)
    new_rows, new_cols = load(new_path)
    for path, rows in ((old_path, old_rows), (new_path, new_rows)):
        if not rows:
            print(f"diff: no rows in {path}")
            return None
    common = [c for c in old_cols if c in new_cols]
    if not common:
        print(f"diff: {old_path.name} and {new_path.name} share no factor columns")
        return None
    ignored = sorted(set(old_cols).symmetric_difference(new_cols))

    def okrow(r):
        return r.get("status") == "OK" and r["tg_tps"] > 0

    def index(rows):
        # several rows can share a key (replicates; or a factor column only the
        # other file has) — keep each config's best measurement
        d: dict[tuple, dict] = {}
        for r in rows:
            k = tuple(str(r.get(c, "")) for c in common)
            if k not in d or r["tg_tps"] > d[k]["tg_tps"]:
                d[k] = r
        return d

    def cfgstr(k):
        return " ".join(f"{c}={v}" for c, v in zip(common, k))

    old_i, new_i = index(old_rows), index(new_rows)
    both = [k for k in old_i if k in new_i]

    print(f"diff: {old_path.name} (old) -> {new_path.name} (new)")
    if ignored:
        print(f"  ignoring factor column(s) not in both files: {', '.join(ignored)}")
    print(f"  configs: {len(old_i)} old, {len(new_i)} new, {len(both)} matched "
          f"({len(old_i) - len(both)} only-old, {len(new_i) - len(both)} only-new)")

    deltas, status_changes = [], []
    for k in both:
        o, n = old_i[k], new_i[k]
        if okrow(o) and okrow(n):
            deltas.append((n["tg_tps"] / o["tg_tps"] - 1.0, k, o, n))
        elif o.get("status") != n.get("status"):
            status_changes.append((k, o.get("status"), n.get("status")))

    if status_changes:
        print(f"\n  status changes ({len(status_changes)}):")
        for k, so, sn in status_changes[:10]:
            print(f"    {so:>7} -> {sn:<7}  {cfgstr(k)}")
        if len(status_changes) > 10:
            print(f"    ... and {len(status_changes) - 10} more")

    summary = {"matched": len(both), "compared": len(deltas),
               "status_changes": len(status_changes), "median_pct": None,
               "old_best_tg": None, "new_best_tg": None,
               "old_winner_still_wins": None}

    if deltas:
        deltas.sort(key=lambda t: t[0])
        pcts = [d[0] for d in deltas]
        n = len(pcts)
        med = pcts[n // 2] if n % 2 else (pcts[n // 2 - 1] + pcts[n // 2]) / 2
        improved = sum(1 for p in pcts if p > 0.02)
        regressed = sum(1 for p in pcts if p < -0.02)
        summary.update(median_pct=med)
        print(f"\n  tg over the {n} config(s) OK in both: median {med:+.1%}, "
              f"{improved} improved / {regressed} regressed (beyond ±2%)")
        worst = [d for d in deltas[:5] if d[0] < -0.02]
        best = [d for d in reversed(deltas[-5:]) if d[0] > 0.02]
        for label, picks in (("largest regressions", worst),
                             ("largest improvements", best)):
            if picks:
                print(f"  {label}:")
                for p, k, o, nw in picks:
                    print(f"    {p:+7.1%}  tg {o['tg_tps']:6.1f} -> {nw['tg_tps']:6.1f}"
                          f"  (pp {o['pp_tps']:.0f} -> {nw['pp_tps']:.0f})  {cfgstr(k)}")

    old_ok = [r for r in old_i.values() if okrow(r)]
    new_ok = [r for r in new_i.values() if okrow(r)]
    if old_ok and new_ok:
        ob = max(old_ok, key=lambda r: r["tg_tps"])
        nb = max(new_ok, key=lambda r: r["tg_tps"])
        ko = tuple(str(ob.get(c, "")) for c in common)
        kn = tuple(str(nb.get(c, "")) for c in common)
        summary.update(old_best_tg=ob["tg_tps"], new_best_tg=nb["tg_tps"],
                       old_winner_still_wins=(ko == kn))
        print(f"\n  best OK config: old {ob['tg_tps']:.1f} tg t/s -> "
              f"new {nb['tg_tps']:.1f} tg t/s")
        if ko == kn:
            print(f"  old winner still wins: {cfgstr(kn)}")
        else:
            now = f"{new_i[ko]['tg_tps']:.1f} tg t/s" if ko in new_i \
                  else "not measured in new"
            print(f"  old winner ({cfgstr(ko)}): {now}")
            print(f"  NEW winner: {cfgstr(kn)}")
    return summary


def build_child_argv(args, cfg: Config, factors: dict, results_path: Path,
                     final: bool, prev_results: list[Path]) -> list[str]:
    """One pass's command line for the single-pass tool (explicit everything so
    the child reproduces the resolved config)."""
    argv = [str(args.model), "--run",
            "--driver", cfg.driver, "--profile", cfg.profile, "--array", "auto",
            "--reps", str(cfg.reps), "--n-prompt", str(cfg.n_prompt),
            "--n-gen", str(cfg.n_gen), "--ctx-floor", str(cfg.ctx_floor),
            "--parallel", str(cfg.parallel), "--score", cfg.score,
            "--timeout", str(args.timeout),
            "--cooldown", str(args.cooldown),
            "--spec-draft-n-max", str(cfg.spec_draft_n_max),
            "--llama-bench", str(cfg.llama_bench),
            "--llama-server", str(cfg.llama_server),
            # results_path already includes the dir; make the child's join a no-op
            "--results-dir", ".", "--results", str(results_path)]
    if args.no_mtp:
        argv.append("--no-mtp")
    if cfg.ngram:
        argv.append("--ngram")
    if cfg.ngram_type:                             # tuning stage: pin the variant
        argv += ["--ngram-type", cfg.ngram_type]
    if cfg.measure_vram:
        argv.append("--vram")
    if args.no_shuffle:
        argv.append("--no-shuffle")
    if args.no_thermal_wait:
        argv.append("--no-thermal-wait")
    if args.thermal_baseline is not None:      # children reuse the parent's idle
        argv += ["--thermal-baseline", str(args.thermal_baseline)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.max_depth is not None:
        argv += ["--max-depth", str(args.max_depth)]
    argv += ["--min-kv", "any"]   # parent already applied the floor; don't re-filter
    # Carry every earlier pass's measurements into this pass's report/picks: a
    # refinement pass can chase a noise-led region, and without the merge the
    # final answer would FORGET a better config pass 1 already measured.
    for rp in prev_results:
        argv += ["--merge-results", str(rp)]
    for name, levels in factors.items():
        flag = "--env" if name in cfg.env_factor_names else "--factor"
        argv += [flag, f"{name}={','.join(str(x) for x in levels)}"]
    if final:
        if args.confirm or args.full:
            argv.append("--confirm")
        if args.html:
            argv += ["--html", str(args.html)]
        if args.no_probe:
            argv.append("--no-probe")
        argv += ["--verify-picks", str(args.verify_picks)]
    else:
        argv.append("--no-probe")  # only probe on the final pass
        argv += ["--verify-picks", "0"]  # only verify final picks
    return argv


def run_iterations(args, cfg: Config):
    base = args.results
    suffix = base.suffix or ".csv"
    factors = dict(cfg.factors)
    prev: list[Path] = []
    for p in range(1, args.iterate + 1):
        final = p == args.iterate
        rp = base.with_name(f"{base.stem}.pass{p}{suffix}")
        print("\n" + "#" * 70)
        print(f"# PASS {p}/{args.iterate}  "
              + ", ".join(f"{k}={'/'.join(str(x) for x in v)}"
                          for k, v in factors.items()))
        print("#" * 70, flush=True)
        argv = build_child_argv(args, cfg, factors, rp, final, prev)
        env = {**os.environ, "LLAMA_OPTIMIZE_CHILD": "1"}
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), *argv], env=env)
        if rc != 0:
            print(f"\npass {p} exited with code {rc}; stopping iteration.")
            return
        if final:
            break
        rows = load_results_csv(rp, factors)
        if not rows:
            print("no results to refine from; stopping.")
            return
        prev.append(rp)
        cfg.factors = factors
        refined = refine_factors(cfg, rows)
        if refined == factors or all(len(v) == 1 for v in refined.values()):
            print("\nfactors converged — stopping refinement early.")
            return
        factors = refined
    print(f"\nAll passes complete. Final report + results (all passes merged): "
          f"{base.with_name(f'{base.stem}.pass{args.iterate}{suffix}')}")


def rank_gate_values(rows: list[dict], gate: str) -> list[tuple]:
    """(gate value, best measured objective) over OK rows, best first. Used to
    rank ngram variants after the screen stage."""
    best: dict[str, float] = {}
    for r in rows:
        if r.get("status") != "OK":
            continue
        v = str(r.get(gate, ""))
        if not v:
            continue
        s = score_of(r)
        if v not in best or s > best[v]:
            best[v] = s
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)


def keep_top_gate_values(ranked: list[tuple], k: int) -> list[str]:
    """The top-k gate values worth tuning: the best `k` real variants (never the
    'none'/off value unless nothing else was measured). At least one when any real
    variant exists."""
    real = [v for v, _ in ranked if v not in ("none", "")]
    return real[:max(1, k)] if real else [v for v, _ in ranked[:1]]


def screen_base_winners(cfg: Config, screen_factors: dict, rows: list[dict]) -> dict:
    """Hold the unconditional knobs at their screen-winning level for the tuning
    stage (so its runs are about the ngram knobs), keeping n_depth's full spread
    (the tradeoff axis, per DESIGN.md). The gate is pinned separately."""
    held: dict = {}
    for name, levels in screen_factors.items():
        if name == "ngram":
            continue
        if name == "n_depth":
            held[name] = levels                        # tradeoff axis: keep spread
            continue
        means = factor_level_means(rows, name)
        best = max(means, key=means.get) if means else str(levels[0])
        held[name] = [str(best)]                        # settle at the screen winner
    return held


def run_ngram_stages(args, cfg: Config):
    """Staged ngram search (docs/CONDITIONAL-FACTORS.md): a screen stage measures
    every variant at default knobs, then one tuning stage per surviving variant
    sweeps only that variant's knobs (gate pinned) with the unconditional knobs
    held at the screen winners. The final answer is the best MEASURED config across
    all stages (each stage merges the earlier ones). Each stage is a child of the
    single-pass tool, exactly like --iterate passes."""
    base = args.results
    suffix = base.suffix or ".csv"

    # Stage 0 — screen: sweep the parent's finalized factor set (base + gate, no
    # conditional children since cfg.ngram_type is unset), passed explicitly so the
    # child inherits the KV floor and any --factor overrides (as --iterate does).
    screen_factors = cfg.factors
    screen_rp = base.with_name(f"{base.stem}.ngram-screen{suffix}")
    print("\n" + "#" * 70)
    print("# NGRAM STAGE 0 — screen variants at default knobs")
    print("#" * 70, flush=True)
    argv = build_child_argv(args, cfg, screen_factors, screen_rp, False, [])
    env = {**os.environ, "LLAMA_OPTIMIZE_CHILD": "1"}
    rc = subprocess.call([sys.executable, os.path.abspath(__file__), *argv], env=env)
    if rc != 0:
        print(f"\nngram screen exited with code {rc}; stopping.")
        return
    rows = load_results_csv(screen_rp, screen_factors)
    if not rows:
        print("no screen results; stopping.")
        return

    ranked = rank_gate_values(rows, "ngram")
    kept = keep_top_gate_values(ranked, cfg.ngram_keep)
    print("\nngram screen ranking (best measured objective per variant):")
    for v, s in ranked:
        print(f"  {v:<14} {s:.2f}" + ("   → tuning" if v in kept else ""))
    held = screen_base_winners(cfg, screen_factors, rows)

    # Stage k — tune each surviving variant; the last one produces the final report.
    prev = [screen_rp]
    tunable = [v for v in kept if ngram_child_levels(v)]
    if not tunable:
        print("\nno surviving variant has tunable knobs; screen result stands.")
        return
    for i, v in enumerate(tunable):
        final = i == len(tunable) - 1
        rp = base.with_name(f"{base.stem}.ngram-{v}{suffix}")
        print("\n" + "#" * 70)
        print(f"# NGRAM STAGE {i + 1} — tune {v}")
        print("#" * 70, flush=True)
        cfg_v = replace(cfg, ngram_type=v)             # pin gate ⇒ child sweeps its knobs
        argv = build_child_argv(args, cfg_v, held, rp, final, prev)
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), *argv], env=env)
        if rc != 0:
            print(f"\nngram tuning of {v} exited with code {rc}; stopping.")
            return
        prev.append(rp)
    print(f"\nNgram staging complete. Final report + results (all stages merged): "
          f"{prev[-1]}")


# ---------------------------------------------------------------------------
# Morris screening (funnel stage 1): rank many knobs by importance (mu*) and flag
# interactions (sigma) at ~r*(k+1) runs, using the vendored `robust` morris tool
# for the design + analysis and our own driver (with crash journal) for the runs.
# ---------------------------------------------------------------------------
def find_robust_binary(name: str) -> Path:
    p = SUBMODULE_DIR / "build" / "bin" / name
    if p.exists():
        return p
    hits = list(SUBMODULE_DIR.glob(f"**/bin/{name}"))
    return hits[0] if hits else p


def parse_morris_analyze(text: str):
    """Parse the morris analyze table into [(factor, mu_star, sigma), ...]."""
    out, started = [], False
    for line in text.splitlines():
        if line.startswith("------"):
            started = True
            continue
        if started:
            if not line.strip():
                break
            parts = line.split()
            if len(parts) >= 3:
                try:
                    out.append((parts[0], float(parts[1]), float(parts[2])))
                except ValueError:
                    pass
    return out


def morris_screen(cfg: Config, args, ap, trajectories: int):
    """Run a Morris screen; report mu*/sigma; reduce cfg.factors to the ones that
    matter (drop negligible ones, fixed at their best-seen level)."""
    morris = find_robust_binary("morris")
    if not morris.exists():
        ap.error(f"morris binary not found at {morris}; build it: "
                 f"make -C {SUBMODULE_DIR}")
    base = args.results
    space_path = Path(str(base) + ".space")
    res_path = Path(str(base) + ".morris_results.csv")
    journal_path = Path(str(base) + ".journal")

    def is_cat(name):
        return FACTORS.get(name, {}).get("kind") == "cat" or name in cfg.env_factor_names

    # .space: numeric factors as [min,max]; categoricals as [0, n-1] index space
    lines = ["factors:"]
    for name, levels in cfg.factors.items():
        if is_cat(name):
            lines.append(f"  {name}: 0, {max(1, len(levels) - 1)}")
        else:
            nums = sorted(float(x) for x in levels)
            lo, hi = nums[0], (nums[-1] if nums[-1] != nums[0] else nums[0] + 1)
            lines.append(f"  {name}: {lo:g}, {hi:g}")
    seed = args.seed if args.seed is not None else 42
    lines += [f"trajectories: {trajectories}", "grid_levels: 4", f"seed: {seed}"]
    space_path.write_text("\n".join(lines) + "\n")

    out = subprocess.run([str(morris), "sample", str(space_path)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        ap.error(f"morris sample failed: {out.stderr.strip()}")
    design = list(csv.DictReader(out.stdout.splitlines()))
    print(f"\n{'#' * 70}\n# MORRIS SCREEN — {trajectories} trajectories, "
          f"{len(design)} runs (r*(k+1)={trajectories}*{len(cfg.factors) + 1})\n{'#' * 70}")
    if cfg.driver == "server":
        print("note: screening on the server driver reloads the model for every "
              "point (slow).\n      For base knobs, screen on the bench driver; "
              "reserve the server driver for MTP/concurrency knobs.")

    def map_point(row):
        f = {}
        for name, levels in cfg.factors.items():
            v = float(row[name])
            if is_cat(name):
                f[name] = levels[max(0, min(len(levels) - 1, int(round(v))))]
            else:
                f[name] = str(min(levels, key=lambda L: abs(float(L) - v)))
        return f

    cache, rows = {}, []
    journal = open(journal_path, "a")
    fh = open(res_path, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["run_id", "eff_tps"])
    fh.flush()
    try:
        for j, row in enumerate(design, 1):
            rid = row.get("run_id", str(j))
            f = map_point(row)
            ckey = tuple(sorted(f.items()))
            if ckey in cache:
                eff, status = cache[ckey], "OK(cached)"
            else:
                prefix = (f"[screen {j}/{len(design)}] "
                          + " ".join(f"{k}={f[k]}" for k in cfg.factors))
                journal_write(journal, "TRY", "run", f"screen-{rid}", json.dumps(f))
                res = with_ticker(prefix, args.timeout,
                                  lambda ff=f: drive_one(cfg, ff, args.timeout))
                eff = objective_tps(cfg, res["pp_tps"], res["tg_tps"])
                status = res["status"]
                cache[ckey] = eff
                print(f"{prefix} -> {status} {cfg.score}={eff:.1f} t/s", flush=True)
                # the screen decides which knobs get DROPPED — settle it like
                # the sweep so drift doesn't rank the factors
                if args.thermal_baseline is not None:
                    wait_until_cool(args.thermal_baseline)
                elif args.cooldown > 0:
                    time.sleep(args.cooldown)
            rows.append({"status": "OK" if eff > 0 else status, "eff_tps": eff, **f})
            w.writerow([rid, f"{eff:.4f}"])
            fh.flush()
    finally:
        fh.close()
        journal.close()

    out = subprocess.run([str(morris), "analyze", str(space_path), str(res_path),
                          "--metric", "eff_tps"], capture_output=True, text=True)
    print("\n" + out.stdout)
    rankings = parse_morris_analyze(out.stdout)
    if not rankings:
        print("(morris analyze returned no rankings — keeping all factors)")
        return

    max_mu = max(mu for _, mu, _ in rankings) or 1.0
    keep = [n for n, mu, _ in rankings if mu >= 0.1 * max_mu]
    if not keep:                       # never drop everything
        keep = [rankings[0][0]]
    # n_depth is the report's tradeoff axis (speed vs context), not a knob to
    # settle — dropping it here would collapse the Pareto/picks to one depth
    # (same guard as refine_factors).
    if "n_depth" in cfg.factors and "n_depth" not in keep:
        keep.append("n_depth")
    dropped = [n for n, _, _ in rankings if n not in keep]
    interacting = [n for n, mu, s in rankings if mu > 0 and s >= mu / 2]

    print(f"KEEP (matter): {', '.join(keep)}")
    if dropped:
        print(f"DROP (negligible, fixed at best): {', '.join(dropped)}")
    if interacting:
        print(f"INTERACTION/nonlinear (σ≥μ*/2): {', '.join(interacting)}")

    best = {}
    for name in cfg.factors:
        means = factor_level_means(rows, name)
        if means:
            best[name] = max(means, key=means.get)
    cfg.factors = {name: (levels if name in keep else [str(best.get(name, levels[0]))])
                   for name, levels in cfg.factors.items()}
    print("→ continuing to the Taguchi sweep on the factors that matter.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # flags alphabetized (keep it that way when adding one)
    ap.add_argument("model", type=Path, nargs="?", help="path to the GGUF model")
    ap.add_argument("--array", default="auto",
                    help="orthogonal array; default 'auto' picks the smallest that "
                         "fits your factors. Advanced: force L9/L18/L25/L27/L125/...")
    ap.add_argument("--confirm", action="store_true",
                    help="after the sweep, run the predicted-optimal config to "
                         "verify the model's prediction (implied by --full)")
    ap.add_argument("--cooldown", type=float, default=0,
                    help="fixed seconds to pause between runs so the GPU can cool "
                         "(fallback when no temp sensor; default 0)")
    ap.add_argument("--ctx-scan", action="store_true",
                    help="probe the physical context ceiling FIRST, then set the "
                         "n_depth axis to fractions of it (0, ¼, ½, ¾, 0.9×) so the "
                         "sweep/Pareto span your full usable context range")
    ap.add_argument("--ctx-size", "-c", type=int, default=None,
                    help="tune at a FIXED context size (like llama.cpp -c): "
                         "shorthand for --min-context N --max-context N")
    ap.add_argument("--diff", nargs=2, type=Path, metavar=("OLD.csv", "NEW.csv"),
                    help="compare two results CSVs (e.g. before/after a llama.cpp "
                         "upgrade): per-config tg deltas over the factor columns "
                         "both share, status changes, and whether the old winner "
                         "still wins. Needs no model or GPU; exits after the report")
    ap.add_argument("--driver", choices=["bench", "server"], default=None,
                    help="benchmark driver (default: from profile). 'server' "
                         "measures real generation incl. MTP and concurrency")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=v1,v2,...",
                    help="sweep an environment variable as a factor (repeatable), "
                         "e.g. --env GGML_CUDA_FORCE_MMQ=0,1")
    ap.add_argument("--factor", action="append", default=[], metavar="NAME=v1,v2,...",
                    help="override/add a sweepable factor (repeatable), e.g. "
                         "--factor ngl=56,60,64 --factor nkvo=0,1 --factor poll=0,50")
    ap.add_argument("--full", action="store_true",
                    help="thorough: 5 reps per config (steadier numbers, slower)")
    ap.add_argument("--html", type=Path, default=None,
                    help="also write a visual HTML report (Pareto + main effects)")
    ap.add_argument("--iterate", type=int, default=1, metavar="N",
                    help="run N refinement passes: each settles the low-impact "
                         "factors at their winner and refines the high-impact ones "
                         "onto a finer grid (screen -> refine -> ...). The final "
                         "report/picks merge ALL passes' results, so extra passes "
                         "can only add information, never lose pass 1's best. "
                         "default 1")
    ap.add_argument("--llama-bench", type=Path, default=None,
                    help="explicit path to the llama-bench binary")
    ap.add_argument("--llama-cpp", type=Path, default=None,
                    help="path to your llama.cpp (its root or build/bin dir). "
                         "Also read from $LLAMA_CPP or $PATH. Required if the "
                         "binaries aren't auto-found.")
    ap.add_argument("--llama-server", type=Path, default=None,
                    help="explicit path to the llama-server binary")
    ap.add_argument("--max-context", "--max-depth", type=int, default=None,
                    dest="max_depth",
                    help="cap the context axis and the ceiling probe at this "
                         "many tokens (don't explore above it)")
    ap.add_argument("--merge-results", action="append", type=Path, default=[],
                    metavar="CSV",
                    help="fold rows from an earlier results CSV into this run's "
                         "report/picks/Pareto without re-running them (repeatable; "
                         "--iterate uses this to carry every pass into the final "
                         "report). Main-effects stay on this run's own balanced "
                         "design.")
    ap.add_argument("--min-context", "--ctx-floor", type=int, default=None,
                    dest="ctx_floor",
                    help="minimum context you need — BALANCED targets it, FASTEST "
                         "only considers configs verified to hold it, and emitted "
                         "-c is floored at it where the sweep has evidence "
                         "(default: from profile)")
    ap.add_argument("--min-kv", default="q8_0", metavar="TYPE",
                    help="KV-cache quality floor: never consider a KV type lossier "
                         "than this (default q8_0, near-lossless). 'any' explores "
                         "all; e.g. --min-kv q4_0 to allow aggressive quantization")
    ap.add_argument("--n-gen", type=int, default=None,
                    help="generated tokens per measurement (default: from profile)")
    ap.add_argument("--n-prompt", type=int, default=None,
                    help="prompt tokens per measurement (default: from profile)")
    ap.add_argument("--no-mtp", action="store_true",
                    help="don't add draft-mtp flags to the emitted server command "
                         "even if the model has an MTP head")
    ap.add_argument("--ngram", action="store_true",
                    help="enable ngram self-speculative decoding (server only): a "
                         "screen stage measures every variant, then the top variants "
                         "get their parameters tuned (staged so the search stays "
                         "clean and affordable — see docs/CONDITIONAL-FACTORS.md)")
    ap.add_argument("--ngram-type", default=None,
                    choices=["ngram-simple", "ngram-mod", "ngram-map-k", "ngram-map-k4v"],
                    help="pin the ngram variant and tune only its parameters in one "
                         "pass (skips the variant screen)")
    ap.add_argument("--ngram-keep", type=int, default=2, metavar="K",
                    help="how many top variants from the screen to tune (default 2)")
    ap.add_argument("--ngram-fast", action="store_true",
                    help="greedy ngram search: tune only the single best-screened "
                         "variant (--ngram-keep 1)")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the max-context probe (which runs by default after "
                         "the sweep: binary-searches the physical context ceiling)")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="run configs in array order (default: randomized to "
                         "decorrelate thermal/background drift from factors)")
    ap.add_argument("--no-thermal-wait", action="store_true",
                    help="disable the default 'wait and watch' settle that pauses "
                         "between runs until GPU temp returns near its idle "
                         "baseline (keeps measurements thermally comparable)")
    ap.add_argument("--parallel", type=int, default=None,
                    help="concurrent request streams for the server driver "
                         "(default: from profile)")
    ap.add_argument("--probe-ctx", action="store_true",
                    help=argparse.SUPPRESS)  # deprecated: the probe is now default
    ap.add_argument("--profile", choices=sorted(PROFILES), default=None,
                    help="workload profile (request shape + objective): single "
                         "(interactive), agents (big-context tool use), multi "
                         "(concurrent serving). Usually set via --use-case; "
                         "default: single")
    ap.add_argument("--quick", action="store_true",
                    help="fast screen: 1 rep per config (noisier, ~1/3 the time)")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from an existing results CSV — no GPU, "
                         "no llama.cpp: reads --results (default: the model's CSV in "
                         "--results-dir), folds in --merge-results, honors --score "
                         "and --html. PROBED CEILING is included when the sweep "
                         "saved its probe result (<results>.probe.json).")
    ap.add_argument("--reps", type=int, default=None,
                    help="repetitions per config (default: 3, or --quick=1/--full=5)")
    ap.add_argument("--results", type=Path, default=None,
                    help="results CSV name, in --results-dir (default: the model's "
                         "name, e.g. <model>.csv; journal/HTML/pass files land beside)")
    ap.add_argument("--results-dir", type=Path, default=Path("results"),
                    help="directory for all output (default: results/)")
    ap.add_argument("--resume", action="store_true",
                    help="skip configs already present in --results (rows are "
                         "saved incrementally, so an interrupted sweep resumes)")
    ap.add_argument("--retry-crashed", action="store_true",
                    help="on resume, also retry configs that were started but never "
                         "finished (suspected machine crash/hang); default skips them")
    ap.add_argument("--run", action="store_true",
                    help="actually execute the benchmark sweep (uses the GPU)")
    ap.add_argument("--score", choices=["tg", "eff"], default="tg",
                    help="objective for stats/fits/picks: 'tg' (default) ranks by "
                         "generation speed alone (pp is measured and reported but "
                         "can't sway the pick); 'eff' ranks by blended effective "
                         "t/s for the profile's request (prefill + decode)")
    ap.add_argument("--screen", type=int, nargs="?", const=6, default=None, metavar="R",
                    help="Morris pre-screen with R trajectories (default 6) to rank "
                         "knobs by importance and drop the negligible ones before the "
                         "sweep — cheap (~R*(k+1) runs). Great with many --factor knobs.")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed for execution order (reproducibility)")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline logic checks and exit (no GPU, no model)")
    ap.add_argument("--server-start-timeout", type=int, default=180,
                    help="max seconds to wait for llama-server to load before "
                         "giving up on a config (default 180)")
    ap.add_argument("--spec-draft-n-max", type=int, default=2,
                    help="--spec-draft-n-max for MTP speculative decoding (default 2)")
    ap.add_argument("--thermal-baseline", type=float, default=None,
                    help=argparse.SUPPRESS)  # internal: parent hands the idle
    #                                          baseline to --iterate child passes
    ap.add_argument("--thinking", action="store_true",
                    help="tune for a reasoning/thinking workload — long generations "
                         "(decode-heavy). Sets n_gen to a reasoning length (~2048); "
                         "default (no flag) is non-thinking / short answers")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="per-run timeout in seconds")
    ap.add_argument("--use-case", choices=list(USE_CASES), default=None,
                    metavar="{app,single,agents,multi-user}",
                    help="high-level runbook that bundles driver+profile+concurrency: "
                         "app (general/embedded llama.cpp via llama-bench), single "
                         "(llama-server, one worker/user), agents (several concurrent "
                         "agents, long tool-use prompts), multi-user (many concurrent "
                         "chat users). --driver/--profile/--parallel override the "
                         "runbook.")
    ap.add_argument("--verify-picks", type=int, default=None, metavar="R",
                    help="re-measure each pick candidate R extra times and report "
                         "the MEDIAN of all its measurements (default: 2, or "
                         "--quick=0/--full=3; 0 disables) — guards the headline "
                         "numbers against thermal/run-to-run noise. Medians persist "
                         "to <results>.verify.json for --report-only; the CSV keeps "
                         "the raw sweep measurements")
    ap.add_argument("--vram", action="store_true",
                    help="measure actual peak VRAM used per run (polls "
                         "rocm-smi/nvidia-smi); records vram_mib and draws the "
                         "VRAM curve + physical ceiling on the Pareto chart")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.diff:
        sys.exit(0 if diff_results(*args.diff) is not None else 1)
    if not args.model:
        ap.error("model path is required (or use --selftest / --diff)")
    if not args.model.exists():
        ap.error(f"model not found: {args.model}")

    # default results name from the model; place relative names in --results-dir
    if args.results is None:
        args.results = Path(f"{args.model.stem}.csv")
    if not args.results.is_absolute():
        prefixed = args.results_dir / args.results
        # --report-only reads an existing file: a CWD-relative path that exists
        # wins over the --results-dir prefix (users paste the path they see)
        if not (args.report_only and args.results.exists()
                and not prefixed.exists()):
            args.results = prefixed
    if args.html is not None and not args.html.is_absolute():
        args.html = args.results_dir / args.html
    if args.run:  # ensure the output directory exists
        args.results.parent.mkdir(parents=True, exist_ok=True)
        if args.html:
            args.html.parent.mkdir(parents=True, exist_ok=True)

    meta = read_gguf_metadata(args.model)
    n_layers = model_block_count(meta)
    n_experts = model_expert_count(meta)
    n_ctx_train = model_context_length(meta)
    n_nextn = model_nextn_layers(meta)
    phys = detect_physical_cores()
    logical = detect_logical_cores()
    vram = detect_vram_mib()

    if args.quick and args.full:
        ap.error("--quick and --full are mutually exclusive")

    # --ctx-size N: fix context at N (min == max), like llama.cpp -c
    if args.ctx_size is not None:
        args.ctx_floor = args.ctx_size
        args.max_depth = args.ctx_size

    # Resolve the workload with precedence: built-in default < use-case runbook <
    # explicit flag. The use-case (if any) supplies defaults for profile/driver/
    # parallel; any flag the user set on the command line still wins.
    uc = USE_CASES.get(args.use_case) or {}
    profile = args.profile or uc.get("profile") or "single"
    prof = PROFILES[profile]
    n_prompt = args.n_prompt if args.n_prompt is not None else prof["n_prompt"]
    # thinking = long reasoning generation (decode-heavy); non-thinking = short
    n_gen = (args.n_gen if args.n_gen is not None
             else 2048 if args.thinking else prof["n_gen"])
    ctx_floor = args.ctx_floor if args.ctx_floor is not None else prof["ctx_floor"]
    driver = args.driver or uc.get("driver") or prof["driver"]
    parallel = (args.parallel if args.parallel is not None
                else uc.get("parallel", prof.get("parallel", 1)))
    reps = args.reps if args.reps is not None else (1 if args.quick else 5 if args.full else BENCH_REPS)
    if args.verify_picks is None:
        args.verify_picks = 0 if args.quick else 3 if args.full else 2

    llama_bench = resolve_binary("llama-bench", args.llama_bench, args.llama_cpp)
    llama_server = resolve_binary("llama-server", args.llama_server, args.llama_cpp)

    # A model with an MTP/NextN head defaults to the server driver: llama-bench
    # cannot do speculative decoding, so on bench the MTP speedup is neither
    # measured nor tunable (its knobs are server-only). An explicit --driver or
    # --use-case still wins, as does --no-mtp; needs llama-server built.
    # ngram self-speculation is also server-only.
    if args.ngram_fast:
        args.ngram_keep = 1
    args.ngram = args.ngram or args.ngram_type is not None   # --ngram-type implies ngram
    if (driver == "bench" and args.driver is None and uc.get("driver") is None
            and (n_nextn or 0) > 0 and not args.no_mtp and llama_server.exists()):
        driver = "server"
        print("note: model has an MTP head — driver auto-switched to server so "
              "the sweep measures and tunes MTP (--driver bench to override)")
    if (driver == "bench" and args.driver is None and uc.get("driver") is None
            and args.ngram and llama_server.exists()):
        driver = "server"
        print("note: ngram speculation requested — driver auto-switched to server "
              "(--driver bench to override)")

    cfg = Config(
        model=args.model.resolve(),
        llama_bench=llama_bench,
        llama_server=llama_server,
        array=args.array,
        ctx_floor=ctx_floor,
        reps=reps,
        n_prompt=n_prompt,
        n_gen=n_gen,
        max_depth=args.max_depth,
        emit_mtp=not args.no_mtp,
        ngram=args.ngram,
        ngram_type=args.ngram_type,
        ngram_keep=args.ngram_keep,
        spec_draft_n_max=args.spec_draft_n_max,
        profile=profile,
        driver=driver,
        parallel=parallel,
        score=args.score,
        measure_vram=args.vram,
        server_start_timeout=args.server_start_timeout,
        hw={"phys": phys, "logical": logical, "n_layers": n_layers, "vram": vram,
            "n_experts": n_experts, "n_ctx_train": n_ctx_train, "n_nextn": n_nextn,
            "numa_nodes": detect_numa_nodes()},
    )
    cfg.factors = build_factors(cfg)
    if args.ctx_size is not None:            # fixed context: don't sweep n_depth
        cfg.factors["n_depth"] = [str(args.ctx_size)]

    # apply --factor overrides / additions
    for spec in args.factor:
        name, _, vals = spec.partition("=")
        name = name.strip()
        levels = [v.strip() for v in vals.split(",") if v.strip()]
        if name not in FACTORS:
            ap.error(f"--factor: unknown factor '{name}' "
                     f"(sweepable: {', '.join(sorted(FACTORS))})")
        if is_server_only(name) and cfg.driver != "server":
            ap.error(f"--factor {name} requires the server driver "
                     "(--driver server or --profile multi)")
        if FACTORS[name].get("bench") is None and cfg.driver == "bench":
            ap.error(f"--factor {name} isn't supported by the bench driver; "
                     "use --driver server")
        if not levels:
            ap.error(f"--factor {name}: no levels given")
        cfg.factors[name] = levels

    # apply --env: each becomes an orthogonal factor that sets a process env var
    for spec in args.env:
        name, _, vals = spec.partition("=")
        name = name.strip()
        levels = [v.strip() for v in vals.split(",") if v.strip()]
        if not name or not levels:
            ap.error(f"--env expects NAME=v1,v2,... (got '{spec}')")
        cfg.factors[name] = levels
        cfg.env_factor_names.add(name)

    # apply the KV quality floor (quality is essentially only KV-type deep here;
    # MTP is lossless, other knobs don't affect quality)
    if "kv_type" in cfg.factors:
        kept = kv_at_or_above(cfg.factors["kv_type"], args.min_kv)
        dropped = [l for l in cfg.factors["kv_type"] if l not in kept]
        cfg.factors["kv_type"] = kept
        if dropped:
            print(f"KV quality floor --min-kv {args.min_kv}: dropping {dropped} "
                  f"(keeping {kept})")

    # --report-only: rebuild the report from the results CSV and exit — no GPU
    if args.report_only:
        if args.run:
            ap.error("--report-only doesn't run anything; drop --run")
        report_from_results(cfg, args, ap)
        return

    # Thermal baseline: capture the idle GPU temperature ONCE, before any GPU
    # work (--ctx-scan/--screen/pass 1 all heat the card), and thread it through
    # child passes via --thermal-baseline — a child capturing its own "idle" at
    # the start of pass 2 would bake a hot GPU into the target and neuter the
    # settle for that whole pass.
    if args.run and not args.no_thermal_wait and args.thermal_baseline is None:
        args.thermal_baseline = gpu_temp_c()

    # --ctx-scan: probe the physical ceiling first, then make the context axis
    # fractions of it, so the sweep spans the full usable range on THIS hardware.
    if args.ctx_scan and not (os.environ.get("LLAMA_OPTIMIZE_CHILD")):
        if not args.run:
            ap.error("--ctx-scan needs --run")
        needed = cfg.llama_server if cfg.driver == "server" else cfg.llama_bench
        if not needed.exists():
            ap.error(f"{needed.name} not found ({needed}); pass --llama-cpp")
        # base config = full offload + most context-efficient allowed KV (lossiest
        # allowed = smallest KV = furthest reach) + first level of the rest
        base = {k: v[0] for k, v in cfg.factors.items()}
        if "ngl" in cfg.factors:
            base["ngl"] = max(cfg.factors["ngl"], key=lambda x: int(x))
        if "kv_type" in cfg.factors:
            base["kv_type"] = max(cfg.factors["kv_type"],
                                  key=lambda k: KV_QUALITY.index(k) if k in KV_QUALITY else 0)
        cap = cfg.hw.get("n_ctx_train") or 131072
        if args.max_depth:                       # --max-context caps the scan
            cap = min(cap, args.max_depth)
        print(f"### Context scan — probing the physical ceiling first "
              f"(ngl={base.get('ngl')} kv={base.get('kv_type')}, cap={cap})...")
        res = probe_max_context(cfg, base, args.timeout, cap, args.thermal_baseline)
        if not res:
            ap.error("--ctx-scan: base config failed to load even at depth 0")
        ceiling = res[0]
        lo = args.ctx_floor or 0                  # --min-context sets the low end
        depths = sorted({max(0, (lo + int((ceiling - lo) * fr)) // 1024 * 1024)
                         for fr in (0.0, 0.25, 0.5, 0.75, 0.9)})
        cfg.factors["n_depth"] = [str(d) for d in depths]
        print(f"physical ceiling ~{ceiling} tokens → n_depth axis "
              f"[{lo}..{ceiling}]: {depths}\n")

    # funnel stage 1: Morris pre-screen (reduces cfg.factors to the ones that
    # matter) before the Taguchi sweep / iterate. Runs in the parent, not children.
    if args.screen and not (os.environ.get("LLAMA_OPTIMIZE_CHILD")):
        if not args.run:
            ap.error("--screen needs --run")
        morris_screen(cfg, args, ap, args.screen)

    # ngram staged search: a screen stage measures every variant, then the top-K
    # get their knobs tuned (gate pinned). Only the unpinned parent orchestrates —
    # a --ngram-type child (pinned gate) runs as a normal single-variant sweep.
    if (cfg.ngram and cfg.ngram_type is None and cfg.driver == "server"
            and not os.environ.get("LLAMA_OPTIMIZE_CHILD")):
        if not args.run:
            ap.error("--ngram needs --run")
        run_ngram_stages(args, cfg)
        return

    # iterative refinement: orchestrate N passes as subprocesses of this tool
    if args.iterate > 1 and not (os.environ.get("LLAMA_OPTIMIZE_CHILD")):
        if not args.run:
            ap.error("--iterate needs --run")
        run_iterations(args, cfg)
        return

    # resolve the array now that the factor set is final
    if str(cfg.array).lower() == "auto":
        cfg.array = choose_array(cfg.factors) or "auto"

    print("=" * 70)
    print("llama-optimize")
    print("=" * 70)
    print(f"model      : {cfg.model.name}")
    arch = meta.get("general.architecture", "?")
    moe = f"MoE ({n_experts} experts)" if n_experts else "dense"
    print(f"arch       : {arch}   layers: {n_layers if n_layers else '?'}   {moe}")
    print(f"CPU        : {phys} physical / {logical} logical cores")
    print(f"VRAM       : {vram} MiB" if vram else "VRAM       : (undetected)")
    if n_ctx_train:
        print(f"native ctx : {n_ctx_train}")
    if n_nextn:
        emit = ("swept as factors (mtp on/off, spec_n_max)" if "mtp" in cfg.factors
                else "will add --spec-type draft-mtp to server cmd" if cfg.emit_mtp
                else "disabled (--no-mtp)")
        print(f"MTP        : yes ({n_nextn} NextN layer(s)) — {emit}")
        if cfg.driver == "bench":
            print("             hint: add --driver server to MEASURE the MTP "
                  "speedup (bench can't); otherwise it's only emitted")
    if cfg.ngram:
        if cfg.ngram_type:
            emit = f"tuning {cfg.ngram_type} knobs (variant pinned)"
        elif "ngram" in cfg.factors:
            emit = "screening variants (tuning follows for the top ones)"
        else:
            emit = "will add --spec-type ngram-mod to server cmd"
        print(f"ngram      : yes — {emit}")
    print(f"profile    : {cfg.profile}  (request {cfg.n_prompt} prompt + "
          f"{cfg.n_gen} gen tokens; driver={cfg.driver})")
    print("objective  : " + ("eff (effective t/s: blends pp + tg)"
                             if cfg.score == "eff" else
                             "tg (generation t/s; pp reported, not scored)"))
    mode = "quick" if args.quick else "full" if args.full else "standard"
    print(f"mode       : {mode}  ({cfg.reps} rep{'s' if cfg.reps != 1 else ''}/config)")
    print(f"array      : {cfg.array}   ctx floor: {cfg.ctx_floor}")
    print("\nfactors:")
    for name, levels in cfg.factors.items():
        print(f"  {name:10s}: {', '.join(levels)}")
    fixed_bits = [f"mmap {'on' if FIXED_MMAP else 'off'}"]
    if "fa" not in cfg.factors:
        fixed_bits.insert(0, f"flash-attn {'on' if FIXED_FA else 'off'}")
    if "batch" not in cfg.factors:
        fixed_bits.append(f"batch {FIXED_BATCH}")
    print(f"fixed      : {', '.join(fixed_bits)}  "
          f"(p={cfg.n_prompt} n={cfg.n_gen} reps={cfg.reps})")

    try:
        exp, runs = generate_runs(cfg.factors, cfg.array)
    except Exception as e:
        ap.error(f"can't build the design for array '{cfg.array}' with "
                 f"{len(cfg.factors)} factors: {e}\n"
                 "  Try --array auto (default) to let it pick a fitting array.")
    print(f"\ngenerated {len(runs)} runs "
          + (f"(array={getattr(exp, 'array_type', cfg.array)})" if exp is not None
             else "(direct sweep — <=1 varying factor, no array needed)"))

    if not args.run:
        print("\n--- PLAN ONLY (no GPU used). Re-run with --run to execute. ---")
        print(f"\nSample command (run 1, {cfg.driver} driver):")
        f0 = runs[0]["factors"]
        if cfg.driver == "server":
            n_ctx = cfg.n_prompt + cfg.n_gen + 256
            print("  " + " ".join(build_server_args(cfg, f0, 8080, n_ctx)))
        else:
            print("  " + " ".join(bench_command(cfg, f0)))
        est = len(runs) * 90  # rough 90s/run guess
        print(f"\nAll {len(runs)} runs would execute sequentially "
              f"(~{est // 60} min at a rough 90s/run).")
        return

    needed = cfg.llama_server if cfg.driver == "server" else cfg.llama_bench
    if not needed.exists():
        ap.error(
            f"{needed.name} not found (looked at: {needed}).\n"
            "  Need the path to your llama.cpp build. Pass --llama-cpp "
            "/path/to/llama.cpp\n  (its build/bin dir), set $LLAMA_CPP, put it on "
            f"$PATH, or pass --{needed.name} directly.")

    # preflight: confirm the binary actually runs (catches missing GPU libs / a
    # wrong build) in a couple of seconds, instead of failing deep in the sweep.
    ok, why = preflight(needed)
    if not ok:
        ap.error(f"{needed.name} at {needed} won't run: {why}\n"
                 "  Is it built for your GPU? Check its ROCm/CUDA/Metal libraries "
                 "are on the loader path (e.g. LD_LIBRARY_PATH), or rebuild llama.cpp.")

    # Idle thermal baseline for the "wait and watch" settle between runs —
    # captured up front (before ctx-scan/screen) or inherited from the parent.
    thermal_wait = not args.no_thermal_wait
    thermal_baseline = args.thermal_baseline if thermal_wait else None
    if thermal_wait and thermal_baseline is not None:
        print(f"thermal    : idle baseline {thermal_baseline:.0f}°C — settle to "
              f"≤{thermal_baseline + THERMAL_BAND_C:.0f}°C between runs "
              f"(cap {THERMAL_CAP_S:.0f}s; --no-thermal-wait to disable)")
    elif thermal_wait:
        print("thermal    : no GPU temp sensor — "
              + (f"using fixed --cooldown {args.cooldown:.0f}s between runs"
                 if args.cooldown > 0 else "no settle between runs (see --cooldown)"))

    # --- execute sweep ---
    cols = (["run_id"] + list(cfg.factors.keys())
            + ["pp_tps", "tg_tps", "eff_tps", "status", "secs", "temp_c"]
            + (["vram_mib"] if cfg.measure_vram else []))

    # Resume keys on run_id (unique per array row), not config values, because
    # orthogonal arrays can repeat a config across rows (intentional replication).
    # Assumes resume uses the same factors/array so run_ids line up.
    rows, done = [], set()
    if args.resume and args.results.exists():
        with open(args.results, newline="") as fh:
            for r in csv.DictReader(fh):
                for c in ("pp_tps", "tg_tps", "eff_tps", "secs"):
                    try:
                        r[c] = float(r[c])
                    except (KeyError, ValueError, TypeError):
                        r[c] = 0.0
                # re-score under the CURRENT --score mode, not whatever mode
                # wrote the CSV — a resumed sweep must fit one objective
                r["eff_tps"] = objective_tps(cfg, r["pp_tps"], r["tg_tps"])
                if r.get("run_id"):
                    rows.append(r)
                    done.add(str(r["run_id"]))
        print(f"resuming: {len(done)} run(s) already in {args.results}, "
              "skipping them")

    fresh = not (args.resume and args.results.exists())
    fh = open(args.results, "w" if fresh else "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    if fresh:
        writer.writeheader()
        fh.flush()

    # Crash journal (see helpers): configs started but never finished on a prior
    # run likely hung/rebooted the machine — mark CRASH and skip, don't retry.
    journal_path = Path(str(args.results) + ".journal")
    if args.resume and not args.retry_crashed:
        tried_load, ok_load, tried_run = read_journal(journal_path)
        crashed_loads = set(tried_load) - ok_load
        # screen runs journal into the same file but are not sweep rows — a
        # completed screen must not resurface as phantom CRASH rows on resume
        crashed = {str(rid): fac for rid, fac in tried_run.items()
                   if str(rid) not in done and not str(rid).startswith("screen-")}
        for run in runs:                       # runs whose server load crashed
            rid = str(run.get("run_id"))
            if rid not in done and load_key_str(cfg, run["factors"]) in crashed_loads:
                crashed.setdefault(rid, run["factors"])
        for rid, fac in crashed.items():
            crow = {"run_id": rid, **{k: fac.get(k, "") for k in cfg.factors},
                    "pp_tps": 0.0, "tg_tps": 0.0, "eff_tps": 0.0,
                    "status": "CRASH", "secs": 0.0}
            rows.append(crow)
            writer.writerow(crow)
            done.add(rid)
        fh.flush()
        if crashed:
            ids = sorted(crashed, key=lambda x: (len(x), x))
            print(f"⚠  {len(crashed)} config(s) were started but never finished "
                  "on a prior run — suspected machine crash/hang.")
            print(f"   Marked CRASH and NOT retrying: runs {ids}")
            print("   (use --retry-crashed to attempt them again once addressed)")
    journal = open(journal_path, "w" if fresh else "a")

    # Execution plan. The server driver groups configs that share load-time
    # params (only the request — prompt length via n_depth — differs) so one
    # server serves the whole group. The bench driver runs each config solo.
    if cfg.driver == "server":
        groups: dict = {}
        order = []
        for run in runs:
            k = load_key(cfg, run["factors"])
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append(run)
        plan = [groups[k] for k in order]
        reused = len(runs) - len(plan)
        if reused > 0:
            print(f"server reuse: {len(plan)} server launch(es) for {len(runs)} "
                  f"runs ({reused} reload(s) saved)")
    else:
        plan = [[run] for run in runs]

    # Randomize execution order to decorrelate slow drift (GPU thermal throttling,
    # background load) from the factors — standard DOE practice. For the server
    # driver we shuffle whole groups so reuse still holds. --no-shuffle keeps
    # array order; --seed makes it reproducible.
    if not args.no_shuffle:
        seed = args.seed if args.seed is not None else random.randrange(1 << 30)
        random.Random(seed).shuffle(plan)
        print(f"execution order: randomized (seed={seed}) to decorrelate drift")

    sweep_start = time.time()
    i = 0
    try:
        for group in plan:
            session = None
            pending = [r for r in group if str(r.get("run_id", "")) not in done]
            if cfg.driver == "server" and pending:
                launch = pending[0]["factors"]
                par = int(launch.get("parallel", cfg.parallel))
                max_depth = max(int(r["factors"].get("n_depth", 0)) for r in pending)
                n_ctx = cfg.n_prompt + max_depth + cfg.n_gen + 256
                if par > 1:
                    n_ctx *= par
                lp = (f"server launch: ngl={launch['ngl']} kv={launch['kv_type']} "
                      f"ub={launch['ubatch']} ctx={n_ctx}")
                lk = load_key_str(cfg, launch)
                journal_write(journal, "TRY", "load", lk, json.dumps(launch))
                session = with_ticker(
                    lp, args.timeout,
                    lambda lf=launch, nc=n_ctx: ServerSession(cfg, lf, nc, args.timeout))
                if getattr(session, "ok", False):
                    journal_write(journal, "OK", "load", lk)  # load survived
            try:
                for run in group:
                    i += 1
                    f = run["factors"]
                    rid = run.get("run_id", i)
                    if str(rid) in done:
                        print(f"[{i}/{len(runs)}] run {rid}: already done, skipping")
                        continue
                    nl = cfg.hw.get("n_layers") or "?"
                    prefix = (f"[{i}/{len(runs)}] run {rid}: "
                              f"{f['ngl']}/{nl} layers on GPU, {f['threads']} threads, "
                              f"{f['kv_type']} KV cache, {f['n_depth']}-token context, "
                              f"ubatch {f['ubatch']}")
                    journal_write(journal, "TRY", "run", rid, json.dumps(f))  # durable
                    temp0 = gpu_temp_c()   # start temp: thermal comparability is
                    #                        checkable in the CSV, not assumed
                    if cfg.driver == "server":
                        res = with_ticker(
                            prefix, args.timeout,
                            lambda ff=f, ss=session: measure_in_session(
                                cfg, ff, ss, args.timeout))
                    else:
                        res = run_with_progress(cfg, f, args.timeout, prefix)
                    res["eff_tps"] = objective_tps(cfg, res["pp_tps"], res["tg_tps"])
                    row = {"run_id": rid, **f, **res,
                           "temp_c": f"{temp0:.0f}" if temp0 is not None else ""}
                    rows.append(row)
                    writer.writerow(row)   # incremental save: survive a crash/kill
                    fh.flush()
                    elapsed = time.time() - sweep_start
                    eta = (elapsed / i) * (len(runs) - i)
                    raw = (f"tg={res['tg_tps']:.1f} pp={res['pp_tps']:.1f}"
                           if cfg.score == "eff" else f"pp={res['pp_tps']:.1f}")
                    print(f"{prefix} -> {res['status']} "
                          f"{cfg.score}={res['eff_tps']:.1f} t/s ({raw}) ({res['secs']:.0f}s)  "
                          f"[{i}/{len(runs)} done, elapsed {fmt_dur(elapsed)}, "
                          f"ETA ~{fmt_dur(eta)}]", flush=True)
                    if i < len(runs):                  # settle before the next run
                        if thermal_wait and thermal_baseline is not None:
                            wait_until_cool(thermal_baseline)
                        elif args.cooldown > 0:
                            time.sleep(args.cooldown)  # fixed fallback, no sensor
            finally:
                if session:
                    session.close()
    finally:
        fh.close()
        journal.close()
    print(f"\nwrote {args.results}")

    all_rows = merge_result_rows(cfg, rows, args.merge_results)

    # Stage-3 verification: the pick candidates get re-measured and their
    # medians become the reported numbers (before the probe, so the probe's
    # base-config choice also sees the settled scores). Persisted to a sidecar
    # so --report-only re-applies it; the CSV keeps the raw measurements.
    if args.verify_picks > 0:
        verify = verify_picks(cfg, all_rows, args.verify_picks, args.timeout,
                              thermal_baseline)
        if verify:
            verify_sidecar(args.results).write_text(json.dumps(verify, indent=1))
            apply_verification(cfg, all_rows, verify)

    # Stage-2 probe runs BEFORE the report so its ceiling appears alongside
    # the picks (fixed context via --ctx-size: no ceiling search). The result
    # is persisted next to the results CSV so --report-only can re-show it.
    probe = None
    if not args.no_probe and args.ctx_size is None:
        probe = run_probe_stage(cfg, all_rows, args.timeout, thermal_baseline)
        if probe:
            probe_sidecar(args.results).write_text(json.dumps(probe, indent=1))

    report(cfg, all_rows, probe)
    opt, predicted = taguchi_effects(cfg, exp, rows)

    if args.html:
        write_html_report(cfg, all_rows, args.html, probe)

    if (args.confirm or args.full) and opt:
        # Run the predicted-optimal config directly to check the additive model
        # (Taguchi best practice). A big predicted-vs-actual gap => interactions.
        f = {k: cfg.factors[k][0] for k in cfg.factors}
        f.update({k: str(v) for k, v in opt.items() if k in cfg.factors})
        print("\n### Confirmation run (predicted-optimal config)")
        # the prediction came from settled runs — measure the check settled too,
        # or a hot GPU masquerades as "interactions"
        wait_until_cool(thermal_baseline)
        prefix = "confirm: " + " ".join(f"{k}={f[k]}" for k in cfg.factors)
        res = with_ticker(prefix, args.timeout,
                          lambda: drive_one(cfg, f, args.timeout))
        actual = objective_tps(cfg, res["pp_tps"], res["tg_tps"])
        raw = (f"tg={res['tg_tps']:.1f} pp={res['pp_tps']:.1f}"
               if cfg.score == "eff" else f"pp={res['pp_tps']:.1f}")
        print(f"  predicted {cfg.score}: "
              + (f"{predicted:.1f} t/s" if predicted else "(n/a)"))
        print(f"  measured  {cfg.score}: {actual:.1f} t/s  "
              f"({raw}, status={res['status']})")
        if predicted and actual > 0:
            err = abs(actual - predicted) / predicted * 100
            verdict = ("additive model holds — trust the prediction" if err <= 15
                       else "LARGE gap: interactions likely — trust the Pareto pick")
            print(f"  prediction error: {err:.0f}%  → {verdict}")


if __name__ == "__main__":
    main()
