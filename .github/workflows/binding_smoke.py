#!/usr/bin/env python3
"""CI smoke test for the taguchi-binding paths the selftest deliberately skips:
orthogonal-array generation (incl. L125 and mixed level counts riding 5-level
columns) and the main-effects analyzer. Needs the submodule built
(`make -C robust`); no GPU, no model."""
import importlib.util
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("lo", ROOT / "llama-optimize.py")
lo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lo)


def dense_server_cfg(**hw_over):
    hw = {"phys": 8, "logical": 16, "n_layers": 32, "vram": 24000,
          "n_ctx_train": 32768, "n_experts": 0, "n_nextn": 0, "numa_nodes": 1}
    hw.update(hw_over)
    return lo.Config(model=Path("m"), llama_bench=Path("b"), array="auto",
                     ctx_floor=8192, driver="server", hw=hw)


def check(cfg, expect_array):
    f = lo.build_factors(cfg)
    f["n_depth"] = ["0", "4096", "16384", "32768", "65536"]
    cfg.factors = f
    arr = lo.choose_array(f)
    assert arr == expect_array, f"expected {expect_array}, chose {arr}"
    exp, runs = lo.generate_runs(f, arr)
    assert len(runs) == int(arr[1:]), (arr, len(runs))
    for k, levels in f.items():           # every factor exercised at every level
        seen = {r["factors"][k] for r in runs}
        assert seen == set(levels), (k, seen, levels)

    random.seed(1)
    rows = [{"run_id": r["run_id"], **r["factors"], "status": "OK",
             "pp_tps": 500.0, "tg_tps": 20 + random.random() * 30,
             "eff_tps": 0.0, "secs": 10.0} for r in runs]
    for r in rows:
        r["eff_tps"] = r["tg_tps"]
    opt, pred = lo.taguchi_effects(cfg, exp, rows)
    assert opt is not None and pred and pred > 0, "analyzer produced no optimum"
    print(f"  {expect_array}: {len(runs)} runs, {len(f)} factors, "
          f"analyzer predicted {pred:.1f} t/s — ok")


# default dense server sweep overflows L25 -> L125, with 2/3-level factors
# riding on 5-level columns
check(dense_server_cfg(), "L125")
# the full MTP surface still fits L125's 31 columns
check(dense_server_cfg(n_nextn=1), "L125")
# a pruned 3-level set draws a 3-level array through the binding
cfg = dense_server_cfg()
cfg.factors = {"a": ["1", "2", "3"], "b": ["x", "y", "z"], "c": ["0", "1"]}
exp, runs = lo.generate_runs(cfg.factors, lo.choose_array(cfg.factors))
assert len(runs) == 9, len(runs)
print("  L9: 9 runs — ok")
print("binding smoke test: all checks passed")
