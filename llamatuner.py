#!/usr/bin/env python3
"""
llamatuner - find good llama.cpp command-line parameters for a given GGUF model
on this machine, using a Taguchi orthogonal-array sweep over llama-bench.

Usage:
    llamatuner.py MODEL.gguf                 # plan only: print the matrix + commands
    llamatuner.py MODEL.gguf --run           # actually run the benchmark sweep
    llamatuner.py MODEL.gguf --run --array L125   # bigger sweep

The tool auto-detects CPU cores, VRAM, and the model's layer count to choose
sensible factor levels, runs the sweep (one llama-bench invocation per Taguchi
run), then reports the fastest / longest-context / balanced configurations as
ready-to-paste llama-server command lines.

Nothing touches the GPU unless --run is given.
"""

import argparse
import csv
import json
import os
import re
import shutil
import struct
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE = PROJECT_ROOT.parent
# The Taguchi/Morris/Sobol suite is vendored as the `robust` git submodule at
# ./taguchi. Its internal layout is nested, so locate the python binding by
# search rather than a fixed path.
SUBMODULE_DIR = PROJECT_ROOT / "taguchi"


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


def find_taguchi_binding() -> Path:
    """Return the dir to add to sys.path so `import taguchi` works."""
    hits = sorted(
        SUBMODULE_DIR.glob("**/bindings/python/taguchi/__init__.py"),
        key=lambda p: len(p.parts),
    )
    if not hits:
        raise SystemExit(
            f"taguchi python binding not found under {SUBMODULE_DIR}.\n"
            "Run:  git submodule update --init  &&  make -C taguchi/<...>  "
            "(build libtaguchi.so)."
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


def detect_vram_mib() -> int | None:
    """Best-effort VRAM total in MiB via rocm-smi; None if unavailable."""
    for cmd in (["rocm-smi", "--showmeminfo", "vram", "--json"],):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if out.returncode != 0:
                continue
            data = json.loads(out.stdout)
            for card in data.values():
                for k, v in card.items():
                    if "vram" in k.lower() and "total" in k.lower():
                        return int(v) // (1024 * 1024)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


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


def depth_levels(n_ctx_train: int | None, override_max: int | None = None) -> list[int]:
    """Five n_depth levels spanning 0..min(native ctx, cap). Adaptive so we
    never test beyond the model's native context."""
    top = min(n_ctx_train or DEFAULT_MAX_DEPTH, DEFAULT_MAX_DEPTH)
    if override_max is not None:
        top = min(top, override_max)
    return five_levels_span(0, top)


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
    spec_draft_n_max: int = 2     # --spec-draft-n-max for MTP
    profile: str = "single"       # workload profile (see PROFILES)
    driver: str = "bench"         # "bench" (llama-bench) or "server" (llama-server)
    parallel: int = 1             # concurrent request streams (server driver)
    factors: dict = field(default_factory=dict)
    hw: dict = field(default_factory=dict)
    env_factor_names: set = field(default_factory=set)  # factors that set env vars


def effective_tps(n_prompt: int, n_gen: int, pp: float, tg: float) -> float:
    """Throughput for a representative request of n_prompt prompt + n_gen gen
    tokens: total tokens / (prefill time + decode time)."""
    if pp <= 0 or tg <= 0:
        return 0.0
    return (n_prompt + n_gen) / (n_prompt / pp + n_gen / tg)


def build_factors(cfg: Config):
    phys = cfg.hw["phys"]
    logical = cfg.hw["logical"]
    n_layers = cfg.hw.get("n_layers")
    depths = depth_levels(cfg.hw.get("n_ctx_train"), cfg.max_depth)
    factors = {
        "ngl": [str(x) for x in ngl_levels(n_layers)],
        "n_depth": [str(x) for x in depths],
        "threads": [str(x) for x in thread_levels(phys, logical)],
        "kv_type": list(DEFAULT_KV_LEVELS),
        "ubatch": [str(x) for x in DEFAULT_UBATCH_LEVELS],
    }
    # For MoE models, expert CPU-offload (-ncmoe) is the biggest RAM/VRAM lever,
    # so promote it to a swept factor (uses L25's 6th column). For dense models
    # it would be inert, so we leave that column spare for error estimation.
    if cfg.hw.get("n_experts", 0) > 0:
        factors["ncmoe"] = [str(x) for x in ncmoe_levels(n_layers)]
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
    factors."""
    counts = [len(v) for v in factors.values()]
    if not counts:
        return None
    nf, mx, n2 = len(counts), max(counts), sum(1 for c in counts if c == 2)
    # one 2-level factor mixed with 3-level factors is the classic L18 case
    if mx == 3 and n2 >= 1 and nf <= 8:
        return "L18"
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
    sys.path.insert(0, str(find_taguchi_binding()))
    from taguchi import Experiment  # noqa: E402

    if array and array.lower() == "auto":
        array = None
    # The binding takes the array in the constructor; None => auto-select.
    exp = Experiment(array_type=array)
    for name, levels in factors.items():
        exp.add_factor(name, levels)
    runs = exp.generate()
    return exp, runs


# ---------------------------------------------------------------------------
# Command building + execution
# ---------------------------------------------------------------------------
# Factor name -> llama-bench flag(s). kv_type applies its value to both -ctk/-ctv.
# Extend this map to make a new llama-bench parameter sweepable as a factor.
BENCH_FLAG = {
    "ngl": ("-ngl",),
    "threads": ("-t",),
    "ubatch": ("-ub",),
    "n_depth": ("-d",),
    "ncmoe": ("-ncmoe",),
    "kv_type": ("-ctk", "-ctv"),
    "batch": ("-b",),
    "nkvo": ("-nkvo",),
    "poll": ("--poll",),
}


def bench_command(cfg: Config, f: dict) -> list[str]:
    cmd = [
        str(cfg.llama_bench),
        "-m", str(cfg.model),
        "-fa", str(FIXED_FA),
        "-mmp", str(FIXED_MMAP),
        "-p", str(cfg.n_prompt),
        "-n", str(cfg.n_gen),
        "-r", str(cfg.reps),
        "-o", "json",
    ]
    ub = int(f.get("ubatch", 512))
    if "batch" not in f:  # batch fixed unless swept; llama-bench needs -b >= -ub
        cmd += ["-b", str(max(FIXED_BATCH, ub))]
    for name, val in f.items():
        if name in cfg.env_factor_names:
            continue  # env vars are applied to the process, not the command line
        flags = BENCH_FLAG.get(name)
        if not flags:
            continue
        if name == "batch":
            val = str(max(int(val), ub))
        for fl in flags:
            cmd += [fl, val]
    return cmd


def run_env(cfg: Config, f: dict) -> dict:
    """Process environment for a run: base env plus any env-factor values."""
    env = dict(os.environ)
    for name in cfg.env_factor_names:
        if name in f:
            env[name] = f[name]
    return env


def server_command(cfg: Config, f: dict, ctx: int) -> str:
    batch = f.get("batch", str(FIXED_BATCH))
    parts = [
        "./llama-server",
        f"-m {cfg.model.name}",
        f"-ngl {f['ngl']}",
        f"-t {f['threads']}",
        f"-c {ctx}",
        f"-ctk {f['kv_type']} -ctv {f['kv_type']}",
        f"-ub {f['ubatch']} -b {batch}",
        "-fa 1",
    ]
    if "ncmoe" in f:
        parts.append(f"-ncmoe {f['ncmoe']}")
    if f.get("nkvo") == "1":
        parts.append("-nkvo")
    if "poll" in f:
        parts.append(f"--poll {f['poll']}")
    par = int(f.get("parallel", cfg.parallel))
    if par > 1:
        parts.append(f"--parallel {par}")
    # Multi-token prediction: if the model ships a NextN/MTP head, enable
    # draft-mtp speculative decoding for extra generation throughput. With the
    # server driver this speedup IS measured; with llama-bench it is NOT (bench
    # can't do speculative decoding) and stacks on top of the reported t/s.
    if cfg.emit_mtp and cfg.hw.get("n_nextn", 0) > 0:
        n_max = f.get("spec_n_max", str(cfg.spec_draft_n_max))
        parts.append(f"--spec-type draft-mtp --spec-draft-n-max {n_max}")
    cmd = " \\\n    ".join(parts)
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
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=run_env(cfg, f))
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "pp_tps": 0.0, "tg_tps": 0.0,
                "secs": time.time() - t0}
    secs = time.time() - t0
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0 or _OOM_PAT.search(combined):
        status = "OOM" if _OOM_PAT.search(combined) else "ERROR"
        return {"status": status, "pp_tps": 0.0, "tg_tps": 0.0, "secs": secs}
    pp, tg = parse_bench_json(proc.stdout)
    if tg is None:
        return {"status": "PARSE_FAIL", "pp_tps": 0.0, "tg_tps": 0.0, "secs": secs}
    return {"status": "OK", "pp_tps": pp or 0.0, "tg_tps": tg, "secs": secs}


# ---------------------------------------------------------------------------
# Server driver: launch llama-server and drive real generation (measures MTP /
# speculative decoding and real concurrency, which llama-bench cannot).
# ---------------------------------------------------------------------------
# Server-only factors (not llama-bench flags).
SERVER_ONLY = {"parallel", "spec_n_max"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_server_args(cfg: Config, f: dict, port: int, n_ctx: int) -> list[str]:
    ub = int(f.get("ubatch", 512))
    batch = int(f.get("batch", FIXED_BATCH))
    args = [
        str(cfg.llama_server), "-m", str(cfg.model),
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(n_ctx), "-fa", "1",
        "-ngl", f["ngl"], "-t", f["threads"],
        "-ctk", f["kv_type"], "-ctv", f["kv_type"],
        "-ub", str(ub), "-b", str(max(batch, ub)),
    ]
    if "ncmoe" in f:
        args += ["-ncmoe", f["ncmoe"]]
    if f.get("nkvo") == "1":
        args += ["-nkvo"]
    if "poll" in f:
        args += ["--poll", f["poll"]]
    par = int(f.get("parallel", cfg.parallel))
    if par > 1:
        args += ["--parallel", str(par)]
    if cfg.emit_mtp and cfg.hw.get("n_nextn", 0) > 0:
        n_max = f.get("spec_n_max", str(cfg.spec_draft_n_max))
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", n_max]
    return args


def _wait_health(port: int, deadline: float) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
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


def _completion(port: int, prompt, n_gen: int, timeout: int) -> dict:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_gen,
        "temperature": 0,
        "cache_prompt": False,
        "timings_per_token": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _measure_round(port: int, prompt, n_gen: int, par: int, timeout: int):
    """One round of `par` concurrent completions; returns (responses, wall_s)."""
    with ThreadPoolExecutor(max_workers=par) as ex:
        w0 = time.time()
        res = list(ex.map(lambda _: _completion(port, prompt, n_gen, timeout),
                          range(par)))
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
        if _wait_health(self.port, time.time() + timeout):
            self.ok = True
        else:
            self.proc.terminate()
            try:
                _, self.err = self.proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def measure(self, prompt_len, n_gen, par, reps, timeout):
        prompt = _realistic_prompt(prompt_len)
        _measure_round(self.port, prompt, n_gen, par, timeout)  # warmup, discard
        pps, tps = [], []
        for _ in range(max(1, reps)):
            res, wall = _measure_round(self.port, prompt, n_gen, par, timeout)
            if par == 1:
                tm = res[0].get("timings", {})
                pps.append(tm.get("prompt_per_second", 0.0) or 0.0)
                tps.append(tm.get("predicted_per_second", 0.0) or 0.0)
            else:
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


def measure_in_session(cfg: Config, f: dict, session, timeout: int) -> dict:
    """Measure one config against an (already-launched) server session."""
    t0 = time.time()
    if session is None or not session.ok:
        err = session.err if session else ""
        status = "OOM" if _OOM_PAT.search(err or "") else "ERROR"
        return {"status": status, "pp_tps": 0.0, "tg_tps": 0.0, "secs": 0.0}
    prompt_len = cfg.n_prompt + int(f.get("n_depth", 0))
    par = int(f.get("parallel", cfg.parallel))
    try:
        pp, tg = session.measure(prompt_len, cfg.n_gen, par, cfg.reps, timeout)
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return {"status": "ERROR", "pp_tps": 0.0, "tg_tps": 0.0,
                "secs": time.time() - t0}
    return {"status": "OK", "pp_tps": pp, "tg_tps": tg, "secs": time.time() - t0}


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


def report(cfg: Config, rows: list[dict]):
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

    print(f"objective  : effective t/s for a {cfg.profile} request "
          f"({cfg.n_prompt} prompt + {cfg.n_gen} gen tokens)")

    fastest = max(ok, key=score_of)
    usable = [r for r in ok if int(r["n_depth"]) >= cfg.ctx_floor]
    longest = max(ok, key=lambda r: (int(r["n_depth"]), score_of(r)))
    balanced = max(usable, key=score_of) if usable else None

    def show(title, r):
        if not r:
            print(f"\n### {title}: none met the constraint")
            return
        print(f"\n### {title}")
        print(f"  eff={score_of(r):.1f} t/s  (tg={r['tg_tps']:.1f}  "
              f"pp={r['pp_tps']:.1f})  depth={r['n_depth']}  ngl={r['ngl']}  "
              f"t={r['threads']}  kv={r['kv_type']}  ub={r['ubatch']}")
        # size context to cover the measured depth + a floor, rounded up to a
        # tidy power of two
        need = max(int(r["n_depth"]) + cfg.n_prompt + cfg.n_gen,
                   cfg.ctx_floor, 4096)
        ctx = 1 << (need - 1).bit_length()
        print("  suggested llama-server command:")
        print("    " + server_command(cfg, r, ctx))

    show("BEST (max effective t/s)", fastest)
    show(f"BALANCED (best with context >= {cfg.ctx_floor})", balanced)
    show("LONGEST CONTEXT", longest)

    print("\n### Pareto frontier (context vs effective t/s)")
    for r in pareto_frontier(rows):
        print(f"  depth={int(r['n_depth']):>6}  eff={score_of(r):6.1f} t/s  "
              f"(tg={r['tg_tps']:5.1f})  ngl={r['ngl']:>3}  "
              f"kv={r['kv_type']:>4}  ub={r['ubatch']:>4}")


def taguchi_effects(exp, rows: list[dict]):
    """Main-effects on tg_tps (higher is better)."""
    try:
        from taguchi import Analyzer
    except Exception:
        return
    # Feed EVERY run: failed configs (OOM/TIMEOUT/ERROR) carry tg_tps=0.0 as a
    # penalty. The analyzer requires a complete design, and scoring failures as 0
    # is the intended "failure is data" behaviour. (Caveat: a 0 from a timeout is
    # a censored value, so trust the Pareto for the pick and use main-effects only
    # to rank which factors matter — see README.)
    results = {int(r["run_id"]): score_of(r) for r in rows}
    n_failed = sum(1 for r in rows if r["status"] != "OK")
    if len(results) < 3:
        print("\n(not enough runs for main-effects analysis)")
        return
    try:
        with Analyzer(exp, metric_name="eff_tps") as an:
            an.add_results_from_dict(results)
            print("\n### Taguchi main effects (effective t/s, higher = better)")
            if n_failed:
                print(f"(note: {n_failed} failed run(s) scored as 0 t/s)")
            print(an.summary())
            opt = an.recommend_optimal(higher_is_better=True)
            print("Predicted-optimal levels:", opt)
    except Exception as e:
        print(f"\n(main-effects analysis skipped: {e})")


# ---------------------------------------------------------------------------
# Stage-2: max-context probe
# ---------------------------------------------------------------------------
def probe_max_context(cfg: Config, base_f: dict, timeout: int, cap: int):
    """Binary-search the largest n_depth that loads (no OOM) for base_f.

    Returns (max_depth, tg_tps_at_max) or None if even depth 0 fails.
    """
    def try_depth(d):
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

        # factor-level generation
        assert five_levels_span(0, 64) == [0, 16, 32, 48, 64]
        assert ngl_levels(64)[0] == 0 and ngl_levels(64)[-1] == 64
        assert len(thread_levels(8, 16)) >= 3

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

        # realistic prompt: varied text of ~n*4 chars, not a repeated token
        assert len(_realistic_prompt(100)) == 400
        assert len(set(_realistic_prompt(200))) > 20  # genuinely varied

        # array auto-selection by factor levels
        assert choose_array({f"f{i}": ["a", "b", "c", "d", "e"] for i in range(5)}) == "L25"
        assert choose_array({f"f{i}": ["a", "b", "c"] for i in range(6)}) == "L27"
        assert choose_array({f"f{i}": ["0", "1"] for i in range(7)}) == "L8"
        assert choose_array({"a": ["0", "1"], "b": ["x", "y", "z"],
                             "c": ["p", "q", "r"]}) == "L18"
        # a fixed (1-level) factor among 3-level ones still picks a 3-level array
        assert choose_array({"t": ["8"], "a": ["1", "2", "3"],
                             "b": ["1", "2", "3"]}) == "L9"
    except AssertionError as e:
        print(f"selftest FAILED: {e}")
        return False
    print("selftest: all checks passed")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, nargs="?", help="path to the GGUF model")
    ap.add_argument("--run", action="store_true",
                    help="actually execute the benchmark sweep (uses the GPU)")
    ap.add_argument("--quick", action="store_true",
                    help="fast screen: 1 rep per config (noisier, ~1/3 the time)")
    ap.add_argument("--full", action="store_true",
                    help="thorough: 5 reps per config (steadier numbers, slower)")
    ap.add_argument("--array", default="auto",
                    help="orthogonal array; default 'auto' picks the smallest that "
                         "fits your factors. Advanced: force L9/L18/L25/L27/L125/...")
    ap.add_argument("--factor", action="append", default=[], metavar="NAME=v1,v2,...",
                    help="override/add a sweepable factor (repeatable), e.g. "
                         "--factor ngl=56,60,64 --factor nkvo=0,1 --factor poll=0,50")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=v1,v2,...",
                    help="sweep an environment variable as a factor (repeatable), "
                         "e.g. --env GGML_CUDA_FORCE_MMQ=0,1")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="single",
                    help="workload profile: single (interactive), agents "
                         "(big-context tool use), multi (concurrent serving); "
                         "sets request shape + objective. default: single")
    ap.add_argument("--driver", choices=["bench", "server"], default=None,
                    help="benchmark driver (default: from profile). 'server' "
                         "measures real generation incl. MTP and concurrency")
    ap.add_argument("--parallel", type=int, default=None,
                    help="concurrent request streams for the server driver "
                         "(default: from profile)")
    ap.add_argument("--llama-cpp", type=Path, default=None,
                    help="path to your llama.cpp (its root or build/bin dir). "
                         "Also read from $LLAMA_CPP or $PATH. Required if the "
                         "binaries aren't auto-found.")
    ap.add_argument("--llama-server", type=Path, default=None,
                    help="explicit path to the llama-server binary")
    ap.add_argument("--ctx-floor", type=int, default=None,
                    help="minimum usable context for BALANCED (default: from profile)")
    ap.add_argument("--probe-ctx", action="store_true",
                    help="after the sweep, binary-search the max context that "
                         "loads for the fastest config (needs --run)")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline logic checks and exit (no GPU, no model)")
    ap.add_argument("--reps", type=int, default=None,
                    help="repetitions per config (default: 3, or --quick=1/--full=5)")
    ap.add_argument("--n-prompt", type=int, default=None,
                    help="prompt tokens per measurement (default: from profile)")
    ap.add_argument("--n-gen", type=int, default=None,
                    help="generated tokens per measurement (default: from profile)")
    ap.add_argument("--max-depth", type=int, default=None,
                    help="cap n_depth factor levels (memory/time budget)")
    ap.add_argument("--no-mtp", action="store_true",
                    help="don't add draft-mtp flags to the emitted server command "
                         "even if the model has an MTP head")
    ap.add_argument("--spec-draft-n-max", type=int, default=2,
                    help="--spec-draft-n-max for MTP speculative decoding (default 2)")
    ap.add_argument("--llama-bench", type=Path, default=None,
                    help="explicit path to the llama-bench binary")
    ap.add_argument("--timeout", type=int, default=1200,
                    help="per-run timeout in seconds")
    ap.add_argument("--results", type=Path, default=Path("results.csv"))
    ap.add_argument("--resume", action="store_true",
                    help="skip configs already present in --results (rows are "
                         "saved incrementally, so an interrupted sweep resumes)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.model:
        ap.error("model path is required (or use --selftest)")
    if not args.model.exists():
        ap.error(f"model not found: {args.model}")

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

    # resolve request shape from the profile, allowing explicit overrides
    prof = PROFILES[args.profile]
    n_prompt = args.n_prompt if args.n_prompt is not None else prof["n_prompt"]
    n_gen = args.n_gen if args.n_gen is not None else prof["n_gen"]
    ctx_floor = args.ctx_floor if args.ctx_floor is not None else prof["ctx_floor"]
    driver = args.driver if args.driver is not None else prof["driver"]
    parallel = args.parallel if args.parallel is not None else prof.get("parallel", 1)
    reps = args.reps if args.reps is not None else (1 if args.quick else 5 if args.full else BENCH_REPS)

    llama_bench = resolve_binary("llama-bench", args.llama_bench, args.llama_cpp)
    llama_server = resolve_binary("llama-server", args.llama_server, args.llama_cpp)

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
        spec_draft_n_max=args.spec_draft_n_max,
        profile=args.profile,
        driver=driver,
        parallel=parallel,
        hw={"phys": phys, "logical": logical, "n_layers": n_layers, "vram": vram,
            "n_experts": n_experts, "n_ctx_train": n_ctx_train, "n_nextn": n_nextn},
    )
    cfg.factors = build_factors(cfg)

    # apply --factor overrides / additions
    for spec in args.factor:
        name, _, vals = spec.partition("=")
        name = name.strip()
        levels = [v.strip() for v in vals.split(",") if v.strip()]
        if name not in BENCH_FLAG and name not in SERVER_ONLY:
            ap.error(f"--factor: unknown factor '{name}' "
                     f"(sweepable: {sorted(set(BENCH_FLAG) | SERVER_ONLY)})")
        if name in SERVER_ONLY and cfg.driver != "server":
            ap.error(f"--factor {name} requires the server driver "
                     "(--driver server or --profile multi)")
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

    # resolve the array now that the factor set is final
    if str(cfg.array).lower() == "auto":
        cfg.array = choose_array(cfg.factors) or "auto"

    print("=" * 70)
    print("llamatuner")
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
        emit = "will add --spec-type draft-mtp to server cmd" if cfg.emit_mtp \
               else "disabled (--no-mtp)"
        print(f"MTP        : yes ({n_nextn} NextN layer(s)) — {emit}")
        if cfg.driver == "bench":
            print("             hint: add --driver server to MEASURE the MTP "
                  "speedup (bench can't); otherwise it's only emitted")
    print(f"profile    : {cfg.profile}  (request {cfg.n_prompt} prompt + "
          f"{cfg.n_gen} gen tokens; driver={cfg.driver})")
    mode = "quick" if args.quick else "full" if args.full else "standard"
    print(f"mode       : {mode}  ({cfg.reps} rep{'s' if cfg.reps != 1 else ''}/config)")
    print(f"array      : {cfg.array}   ctx floor: {cfg.ctx_floor}")
    print("\nfactors:")
    for name, levels in cfg.factors.items():
        print(f"  {name:10s}: {', '.join(levels)}")
    print(f"fixed      : -fa {FIXED_FA}  -mmp {FIXED_MMAP}  -b {FIXED_BATCH}  "
          f"(-p {cfg.n_prompt} -n {cfg.n_gen} -r {cfg.reps})")

    try:
        exp, runs = generate_runs(cfg.factors, cfg.array)
    except Exception as e:
        ap.error(f"can't build the design for array '{cfg.array}' with "
                 f"{len(cfg.factors)} factors: {e}\n"
                 "  Try --array auto (default) to let it pick a fitting array.")
    print(f"\ngenerated {len(runs)} runs "
          f"(array={getattr(exp, 'array_type', cfg.array)})")

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

    # --- execute sweep ---
    cols = (["run_id"] + list(cfg.factors.keys())
            + ["pp_tps", "tg_tps", "eff_tps", "status", "secs"])

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
                session = with_ticker(
                    lp, args.timeout,
                    lambda lf=launch, nc=n_ctx: ServerSession(cfg, lf, nc, args.timeout))
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
                    if cfg.driver == "server":
                        res = with_ticker(
                            prefix, args.timeout,
                            lambda ff=f, ss=session: measure_in_session(
                                cfg, ff, ss, args.timeout))
                    else:
                        res = run_with_progress(cfg, f, args.timeout, prefix)
                    res["eff_tps"] = effective_tps(cfg.n_prompt, cfg.n_gen,
                                                   res["pp_tps"], res["tg_tps"])
                    row = {"run_id": rid, **f, **res}
                    rows.append(row)
                    writer.writerow(row)   # incremental save: survive a crash/kill
                    fh.flush()
                    elapsed = time.time() - sweep_start
                    eta = (elapsed / i) * (len(runs) - i)
                    print(f"{prefix} -> {res['status']} "
                          f"eff={res['eff_tps']:.1f} t/s (tg={res['tg_tps']:.1f} "
                          f"pp={res['pp_tps']:.1f}) ({res['secs']:.0f}s)  "
                          f"[{i}/{len(runs)} done, elapsed {fmt_dur(elapsed)}, "
                          f"ETA ~{fmt_dur(eta)}]", flush=True)
            finally:
                if session:
                    session.close()
    finally:
        fh.close()
    print(f"\nwrote {args.results}")

    report(cfg, rows)
    taguchi_effects(exp, rows)

    if args.probe_ctx:
        ok = [r for r in rows if r["status"] == "OK"]
        if ok:
            fastest = max(ok, key=score_of)
            cap = cfg.hw.get("n_ctx_train") or 131072
            print(f"\n### Max-context probe (fastest config, cap={cap})")
            base = {k: fastest[k] for k in cfg.factors}
            res = probe_max_context(cfg, base, args.timeout, cap)
            if res:
                depth, tps = res
                print(f"  largest context that loads: ~{depth} tokens"
                      + (f"  (tg={tps:.1f} t/s there)" if tps else ""))
            else:
                print("  even depth 0 failed to load — check the config")


if __name__ == "__main__":
    main()
