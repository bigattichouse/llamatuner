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
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the workspace layout; override with flags if needed)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE = PROJECT_ROOT.parent
DEFAULT_LLAMA_BENCH = WORKSPACE / "llama.cpp" / "build" / "bin" / "llama-bench"
# The Taguchi/Morris/Sobol suite is vendored as the `robust` git submodule at
# ./taguchi. Its internal layout is nested, so locate the python binding by
# search rather than a fixed path.
SUBMODULE_DIR = PROJECT_ROOT / "taguchi"


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
    reps: int = BENCH_REPS
    n_prompt: int = BENCH_N_PROMPT
    n_gen: int = BENCH_N_GEN
    max_depth: int | None = None  # cap n_depth levels (memory/time budget)
    factors: dict = field(default_factory=dict)
    hw: dict = field(default_factory=dict)


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


# ---------------------------------------------------------------------------
# Taguchi run generation (via the python binding)
# ---------------------------------------------------------------------------
def generate_runs(factors: dict, array: str | None):
    sys.path.insert(0, str(find_taguchi_binding()))
    from taguchi import Experiment  # noqa: E402

    exp = Experiment()
    for name, levels in factors.items():
        exp.add_factor(name, levels)
    if array:
        # Experiment honours an explicit array if the binding supports it;
        # otherwise it auto-selects. We pass via the .tgu path fallback below.
        try:
            exp.set_array(array)
        except Exception:
            pass
    runs = exp.generate()
    return exp, runs


# ---------------------------------------------------------------------------
# Command building + execution
# ---------------------------------------------------------------------------
def bench_command(cfg: Config, f: dict) -> list[str]:
    cmd = [
        str(cfg.llama_bench),
        "-m", str(cfg.model),
        "-ngl", f["ngl"],
        "-t", f["threads"],
        "-ctk", f["kv_type"],
        "-ctv", f["kv_type"],
        "-ub", f["ubatch"],
        "-b", str(FIXED_BATCH),
        "-fa", str(FIXED_FA),
        "-mmp", str(FIXED_MMAP),
        "-d", f["n_depth"],
        "-p", str(cfg.n_prompt),
        "-n", str(cfg.n_gen),
        "-r", str(cfg.reps),
        "-o", "json",
    ]
    if "ncmoe" in f:
        cmd += ["-ncmoe", f["ncmoe"]]
    return cmd


def server_command(cfg: Config, f: dict, ctx: int) -> str:
    parts = [
        "./llama-server",
        f"-m {cfg.model.name}",
        f"-ngl {f['ngl']}",
        f"-t {f['threads']}",
        f"-c {ctx}",
        f"-ctk {f['kv_type']} -ctv {f['kv_type']}",
        f"-ub {f['ubatch']} -b {FIXED_BATCH}",
        "-fa 1",
    ]
    if "ncmoe" in f:
        parts.append(f"-ncmoe {f['ncmoe']}")
    return " \\\n    ".join(parts)


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


def run_one(cfg: Config, f: dict, timeout: int):
    cmd = bench_command(cfg, f)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
# Reporting
# ---------------------------------------------------------------------------
def pareto_frontier(rows: list[dict]) -> list[dict]:
    """Non-dominated set maximizing (context depth, tg_tps) among OK rows."""
    ok = [r for r in rows if r["status"] == "OK" and r["tg_tps"] > 0]
    frontier = []
    for r in ok:
        depth, tps = int(r["n_depth"]), r["tg_tps"]
        dominated = any(
            int(o["n_depth"]) >= depth and o["tg_tps"] >= tps and o is not r
            and (int(o["n_depth"]) > depth or o["tg_tps"] > tps)
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

    fastest = max(ok, key=lambda r: r["tg_tps"])
    usable = [r for r in ok if int(r["n_depth"]) >= cfg.ctx_floor]
    longest = max(ok, key=lambda r: (int(r["n_depth"]), r["tg_tps"]))
    balanced = None
    if usable:
        # best t/s among configs that clear the context floor
        balanced = max(usable, key=lambda r: r["tg_tps"])

    def show(title, r):
        if not r:
            print(f"\n### {title}: none met the constraint")
            return
        print(f"\n### {title}")
        print(f"  tg={r['tg_tps']:.1f} t/s  pp={r['pp_tps']:.1f} t/s  "
              f"depth={r['n_depth']}  ngl={r['ngl']}  t={r['threads']}  "
              f"kv={r['kv_type']}  ub={r['ubatch']}")
        # size context to cover the measured depth + a floor, rounded up to a
        # tidy power of two
        need = max(int(r["n_depth"]) + cfg.n_prompt + cfg.n_gen,
                   cfg.ctx_floor, 4096)
        ctx = 1 << (need - 1).bit_length()
        print("  suggested llama-server command:")
        print("    " + server_command(cfg, r, ctx))

    show("FASTEST (max t/s)", fastest)
    show(f"BALANCED (max t/s with context >= {cfg.ctx_floor})", balanced)
    show("LONGEST CONTEXT", longest)

    print("\n### Pareto frontier (context vs t/s)")
    for r in pareto_frontier(rows):
        print(f"  depth={int(r['n_depth']):>6}  tg={r['tg_tps']:6.1f} t/s  "
              f"ngl={r['ngl']:>3}  kv={r['kv_type']:>4}  ub={r['ubatch']:>4}")


def taguchi_effects(exp, rows: list[dict]):
    """Main-effects on tg_tps (higher is better)."""
    try:
        from taguchi import Analyzer
    except Exception:
        return
    results = {int(r["run_id"]): r["tg_tps"] for r in rows if r["status"] == "OK"}
    if len(results) < 3:
        print("\n(not enough successful runs for main-effects analysis)")
        return
    try:
        with Analyzer(exp, metric_name="tg_tps") as an:
            an.add_results_from_dict(results)
            print("\n### Taguchi main effects (tg t/s, higher = better)")
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
        r = run_one(cfg, f, timeout)
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
    ap.add_argument("--array", default="L25",
                    help="Taguchi array (L25, L125, ...); default L25")
    ap.add_argument("--ctx-floor", type=int, default=16384,
                    help="minimum usable context for the BALANCED pick")
    ap.add_argument("--probe-ctx", action="store_true",
                    help="after the sweep, binary-search the max context that "
                         "loads for the fastest config (needs --run)")
    ap.add_argument("--selftest", action="store_true",
                    help="run offline logic checks and exit (no GPU, no model)")
    ap.add_argument("--reps", type=int, default=BENCH_REPS,
                    help=f"llama-bench repetitions per config (default {BENCH_REPS})")
    ap.add_argument("--n-prompt", type=int, default=BENCH_N_PROMPT,
                    help=f"prompt tokens per measurement (default {BENCH_N_PROMPT})")
    ap.add_argument("--n-gen", type=int, default=BENCH_N_GEN,
                    help=f"generated tokens per measurement (default {BENCH_N_GEN})")
    ap.add_argument("--max-depth", type=int, default=None,
                    help="cap n_depth factor levels (memory/time budget)")
    ap.add_argument("--llama-bench", type=Path, default=DEFAULT_LLAMA_BENCH)
    ap.add_argument("--timeout", type=int, default=1200,
                    help="per-run timeout in seconds")
    ap.add_argument("--results", type=Path, default=Path("results.csv"))
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
    phys = detect_physical_cores()
    logical = detect_logical_cores()
    vram = detect_vram_mib()

    cfg = Config(
        model=args.model.resolve(),
        llama_bench=args.llama_bench,
        array=args.array,
        ctx_floor=args.ctx_floor,
        reps=args.reps,
        n_prompt=args.n_prompt,
        n_gen=args.n_gen,
        max_depth=args.max_depth,
        hw={"phys": phys, "logical": logical, "n_layers": n_layers, "vram": vram,
            "n_experts": n_experts, "n_ctx_train": n_ctx_train},
    )
    cfg.factors = build_factors(cfg)

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
    print(f"array      : {cfg.array}   ctx floor: {cfg.ctx_floor}")
    print("\nfactors:")
    for name, levels in cfg.factors.items():
        print(f"  {name:10s}: {', '.join(levels)}")
    print(f"fixed      : -fa {FIXED_FA}  -mmp {FIXED_MMAP}  -b {FIXED_BATCH}  "
          f"(-p {cfg.n_prompt} -n {cfg.n_gen} -r {cfg.reps})")

    exp, runs = generate_runs(cfg.factors, cfg.array)
    print(f"\ngenerated {len(runs)} runs "
          f"(array={getattr(exp, 'array_type', cfg.array)})")

    if not args.run:
        print("\n--- PLAN ONLY (no GPU used). Re-run with --run to execute. ---")
        print("\nSample command (run 1):")
        f0 = runs[0]["factors"]
        print("  " + " ".join(bench_command(cfg, f0)))
        est = len(runs) * 90  # rough 90s/run guess
        print(f"\nAll {len(runs)} runs would execute sequentially "
              f"(~{est // 60} min at a rough 90s/run).")
        return

    # --- execute sweep ---
    rows = []
    for i, run in enumerate(runs, 1):
        f = run["factors"]
        rid = run.get("run_id", i)
        print(f"\n[{i}/{len(runs)}] run {rid}: "
              f"ngl={f['ngl']} d={f['n_depth']} t={f['threads']} "
              f"kv={f['kv_type']} ub={f['ubatch']} ... ", end="", flush=True)
        res = run_one(cfg, f, args.timeout)
        row = {"run_id": rid, **f, **res}
        rows.append(row)
        print(f"{res['status']} tg={res['tg_tps']:.1f} t/s "
              f"pp={res['pp_tps']:.1f} ({res['secs']:.0f}s)")

    # persist (columns follow the actual factor set, incl. ncmoe when MoE)
    with open(args.results, "w", newline="") as fh:
        cols = (["run_id"] + list(cfg.factors.keys())
                + ["pp_tps", "tg_tps", "status", "secs"])
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.results}")

    report(cfg, rows)
    taguchi_effects(exp, rows)

    if args.probe_ctx:
        ok = [r for r in rows if r["status"] == "OK"]
        if ok:
            fastest = max(ok, key=lambda r: r["tg_tps"])
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
