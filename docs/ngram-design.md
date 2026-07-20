# ngram support — implementation design & work log

Isolates the concrete ngram self-speculative-decoding work (`--ngram`). The
*principles* — why conditional parameters need special handling and the general
mechanism — live in [`CONDITIONAL-FACTORS.md`](CONDITIONAL-FACTORS.md); ngram is
that document's first real consumer. This file tracks the ngram-specific factor
model, the staged plan applied to it, the `robust` submodule bump that rides
along, and the task checklist.

## Origin

Adapted from PR #1 (`giveen:main`), which added `--ngram` and made `--ctx-floor`
respected by the depth sweep. The PR works mechanically but sweeps every ngram
variant's parameters as independent orthogonal-array factors — the conditional
factor problem `CONDITIONAL-FACTORS.md` exists to solve. We pulled it into a fix
branch, corrected the review issues, and are re-basing the ngram design on the
staged mechanism before merging.

## The author's intent (what we are preserving)

The PR expresses a clear and reasonable goal: **explore every ngram variant and
its tuning parameters and return the configuration that runs fastest on this
machine and workload.** It reached for "add every parameter as a factor" because
that is the obvious way to say "search all of it" — it does not assume the author
knew how a Taguchi orthogonal array or the Morris screen consumes those factors
(that main effects are only clean when every factor is live in every row). Our
job is to keep that intent — the *full* variant-and-parameter search — and change
only the *method* so the search is statistically valid and affordable. The
redesign must not quietly narrow the search to one variant; it must still measure
every variant and actually tune the promising ones. Where this doc says "the PR
is wrong," it means the mechanism, never the goal.

## Status

**Landed on `pr-1-fixes` (commit 426d289):**

- Emission gating — a first-cut `_ngram_param_inactive(name, f)` prefix map keeps
  llama-server from ever receiving flags for a variant it isn't running
  (invariant I2). *Will be replaced by the general `is_active` — see below.*
- Removed a duplicated comment block in `server_command` and a stray fragment.
- Restored the `ggml_backend_alloc` OOM-detection assertion that the PR
  accidentally overwrote, and re-grouped the thread/depth asserts.
- Restored the `selftest: all checks passed` success print.
- `--ngram` now respects an explicit use-case driver in the auto-switch,
  matching the MTP block.
- Selftest coverage for the gating and the ngram=none case; full selftest green.

**Not yet done (this design):** array inflation (F2), biased main effects (F3),
and orthogonality (F4) are still present because the per-variant knobs still
enter one flat array. The staged plan below closes them.

## The ngram factor model

`--spec-type` is the **gate** `G`. Its children:

| Gate value | children | flags |
|------------|----------|-------|
| `none` | — | *(omit `--spec-type`)* |
| `ngram-simple` | size-n, size-m, min-hits | `--spec-ngram-simple-{size-n,size-m,min-hits}` |
| `ngram-map-k` | size-n, size-m, min-hits | `--spec-ngram-map-k-{size-n,size-m,min-hits}` |
| `ngram-map-k4v` | size-n, size-m, min-hits | `--spec-ngram-map-k4v-{size-n,size-m,min-hits}` |
| `ngram-mod` | n-match, n-min, n-max | `--spec-ngram-mod-{n-match,n-min,n-max}` |

Three of the five variants share the same `{size-n, size-m, min-hits}` structure,
differing only in the flag's middle token — which is exactly the gate value.
Collapse those nine registry entries to three logical factors via `flag_for`:

```python
"ngram_size_n": {
    "kind": "num", "server_only": True, "bench": None,
    "active_when": ("ngram", {"ngram-simple", "ngram-map-k", "ngram-map-k4v"}),
    "flag_for": lambda v: (f"--spec-{v}-size-n",),   # v ∈ the live set above
},
# ...size_m, min_hits likewise
"ngram_mod_n_match": {
    "kind": "num", "server_only": True, "bench": None,
    "server": ("--spec-ngram-mod-n-match",),
    "active_when": ("ngram", {"ngram-mod"}),
},
# ...ngram_mod_n_min, ngram_mod_n_max likewise
"ngram": {
    "kind": "cat", "server_only": True, "bench": None,
    "server": ("--spec-type",),
    "translate": {"none": "", "ngram-simple": "ngram-simple", ...},
},
```

Registry count drops from 14 ngram entries (gate + spec_n_max + 12 knobs) to 8
(gate + spec_n_max + 3 shared + 3 mod), and every one carries an `active_when`
except the gate and `spec_n_max`.

The existing merge into a single comma-separated `--spec-type` when MTP + ngram
co-activate is kept — and **verified correct**: `--spec-type` parses a
comma-separated list into `params.speculative.types` (`arg.cpp`), so
`draft-mtp,ngram-mod` legitimately runs both.

## Verified against llama.cpp

Checked against the local `llama.cpp` tree (`common/arg.cpp`, `common/common.h`,
`common/speculative.cpp`). Three things the PR got subtly wrong or that pin the
registry down:

**1. `spec_n_max` (`--spec-draft-n-max`) is NOT an ngram knob — drop it from the
ngram sweep.** `common.h` is explicit: the `draft` sub-struct is *"used by Simple,
MTP, Eagle3, etc. — all methods that require some kind of draft model."* The
ngram implementations read their own `ngram_mod` / `ngram_map` structs and never
touch `draft.n_max/n_min/p_min/p_split`. So the PR is wrong on two counts:
- It adds `spec_n_max` to the ngram factor set (when ngram is on without MTP) —
  a **globally inert factor**: it has no effect on *any* ngram variant, so it is
  pure dilution and is **not** caught by variant-gating (it isn't variant-specific,
  it's applicable to *no* ngram variant). Fix: only add `spec_n_max` under MTP;
  never for ngram.
- It emits `--spec-draft-n-max 16` as an ngram default — a harmless no-op that is
  also misleading (ngram-mod's own default draft length is 64). Drop it.

The correct ngram "how many tokens to draft" knobs are already per-variant and
already covered: `ngram-mod` → `n-max`; `ngram-simple/map-k/map-k4v` → `size-m`.

**2. Real defaults + bounds (fill the registry from source, don't guess):**

| factor | default | valid range |
|--------|---------|-------------|
| `ngram_mod_n_match` | 24 | 1–1024 |
| `ngram_mod_n_max` | 64 | 0–1024 |
| `ngram_mod_n_min` | 48 | 0–1024 |
| `ngram_size_n` (simple/map-k/map-k4v) | 12 | 1–1024 |
| `ngram_size_m` | 48 | 1–1024 |
| `ngram_min_hits` | 1 | ≥1 |

`size_*`/`min_hits` are `uint16`. The PR's sweep levels sit inside these bounds.

**3. Coupled-constraint — decided: no clamp.** `ngram_mod_n_min ≤ n_max` is the
implied relation (defaults 48 ≤ 64), and the sweep can produce inverted rows
(`n_min=96, n_max=32`). Verified against llama.cpp: inverted bounds are **legal**
(`arg.cpp` only enforces 0–1024 per value, independently) and **do not crash** —
`ngram-mod` drafts at most `n_max` tokens (`speculative.cpp:986`), and a draft
shorter than `n_min` is simply rejected (`speculative.cpp:748`), so an inverted
row accepts no draft tokens and runs at ~baseline. It therefore **self-
deprioritizes**: it scores at/below the non-speculative baseline and the Pareto /
main-effects naturally rank it last. We deliberately do **not** clamp — clamping
would desync the CSV factor level from the config actually run and hide that real
signal. (The general "sibling ordering constraint" pattern is noted in
`CONDITIONAL-FACTORS.md`; the chosen policy here is "accept as expected-poor".)

**Deliberate exclusion:** the `ngram-cache` spec type exists but is omitted — it
needs external static/dynamic cache files (`--lookup-cache-static/-dynamic`), not
a numeric sweep. The removed `--spec-ngram-size-{n,m}` / `--spec-ngram-min-hits`
aliases (they `arg_removed`-error) are correctly absent.

## Staged plan applied to ngram

Per the planner in `CONDITIONAL-FACTORS.md`:

- **Stage 0 — screen the variants.** Factors: base sweep + `ngram` (5 levels).
  Each variant runs with its default knobs. ~6 factors → **L25**. This *measures*
  all five variants; it does not yet pick one. (No `spec_n_max` here — it doesn't
  affect ngram; see "Verified against llama.cpp".)
- **Keep the contenders.** Carry forward every variant within a margin of the best
  (default keep top-`K=2`; `--ngram-fast` ⇒ `K=1` greedy; a "tune all five" mode
  for exhaustive users). Only clearly-dominated variants are dropped — same
  caution the Morris screen applies to knobs. This is what preserves the author's
  breadth: a variant that is weak at defaults but strong once tuned still gets its
  Stage-k shot.
- **Stage k — tune each kept variant.** One clean OA per surviving variant: gate
  pinned to it, only that variant's children swept (3 knobs → **L25** or smaller),
  unconditional knobs held at their Stage-0 winners.
- **Pick off measured evidence.** Best *measured* config across Stage 0 + all
  Stage-k branches wins — the tool already selects from the measured Pareto, so a
  variant that only leads once tuned wins on its merits.

**~75 runs at `K=2` (25 + 2·25), all fully live**; ~150 to tune all five — versus
the current flat **L125 (125 runs)** that dilutes every variant at once and tunes
none of them cleanly. Same order of budget, but every run informs a real choice.

Surface:
- `--ngram` + `--iterate ≥ 2` → Stage 0 then Stage k automatically (children open
  for each kept variant when the gate narrows — the `refine_factors` transition).
- `--ngram` + `--iterate 1` (default) → Stage 0 only: all variants measured at
  default knobs, best-measured reported; report states that per-variant *tuning*
  needs `--iterate ≥ 2` (so a single-pass user is not misled).
- `--ngram-type <variant>` → pin the gate up front; single pass runs Stage k
  directly on that variant's knobs.

## `robust` submodule (Taguchi/Morris/Sobol binding)

The staged plan leans harder on the binding: more small arrays generated
per run, and the analyzer conditioned on live rows. The vendored suite needs to
be current.

**State (as of this design):**

- Submodule path `taguchi`, url `github.com/bigattichouse/robust`.
- `main` records **`4d7fd61`**; **PR #1 records the stale `2fcb231`** (7 commits
  behind). The gap includes upstream hardening/bug fixes:
  `cf2ba54` (non-finite & control-char input hardening), `6b7943f` (fuzz target,
  sanitizer suite, CI, round-trip). Upstream `origin/main` has one further
  docs-only commit, `34ff2d0`.
- **Action:** the merge must bump the submodule to **at least `main`'s `4d7fd61`**
  — never let the PR drag the pointer back to `2fcb231` and re-introduce the
  fixed bugs. Bumping to upstream `34ff2d0` is optional (docs only). Re-run the
  binding smoke test (`ROADMAP §6`) after the bump: exercise L25 **and** L125
  generation plus the analyzer, the paths the pure-Python selftest skips.

**Rename `taguchi` → `robust` (align with the upstream project name).** The
directory predates the project's rename to `robust`; the code comment already
notes the mismatch ("vendored as the `robust` git submodule at ./taguchi"). This
is a **directory/submodule-path** rename only — the Python package inside is still
`taguchi` (`bindings/python/taguchi/__init__.py`), so `import taguchi` and
`from taguchi import Experiment/Analyzer` are unchanged. Touch-points:

- `.gitmodules`: `[submodule "taguchi"]` name + `path = taguchi` → `robust`.
- `.git/modules/taguchi` → `.git/modules/robust` (via `git mv taguchi robust`,
  which git handles for submodules ≥ 1.8).
- `llama-optimize.py`: `SUBMODULE_DIR = PROJECT_ROOT / "taguchi"` → `"robust"`;
  the surrounding comments; `find_taguchi_binding` / `find_robust_binary` glob
  under `SUBMODULE_DIR` so they need no logic change (optionally rename
  `find_taguchi_binding` for consistency).
- README setup/build instructions that reference `./taguchi`.

Do the rename as its **own commit**, separate from the ngram feature, so a
bisect can tell a path rename from a behavior change. Confirm upstream has **not**
also renamed the Python package before assuming `import taguchi` stays valid.

## Task checklist

Declarative core (no behavior change for existing factors):

- [x] Add `active_when` + `flag_for` support to the `FACTORS` schema.
- [x] Implement `is_active(name, assignment)` and `active_factors(...)`.
- [x] Registry validator (gate exists, values are real levels, `flag_for` total,
      gate graph acyclic); run in `selftest` and assert at `build_factors` time.

ngram conversion:

- [x] Move ngram entries to `active_when`; collapse the 9 shared knobs to 3 via
      `flag_for`; set documented defaults (from the verified table above).
- [x] **Drop `spec_n_max` from the ngram sweep** and remove the
      `--spec-draft-n-max 16` ngram default (it's a draft-model/MTP knob, inert
      for ngram); keep it swept only under MTP.
- [x] Replace `_ngram_param_inactive` with `is_active` in `factor_flags`.
- [x] Condition `factor_level_means` on `is_active` (invariant I3).
- [x] Inverted `ngram_mod_n_min > n_max` rows: decided **no clamp** — legal,
      no crash, self-deprioritizing (see "Verified against llama.cpp" #3).

Staging:

- [x] Stage planner (`plan_stages`) with the liveness property test.
- [x] Keep-contenders policy: `--ngram-keep` (top-`K`, default 2) + `--ngram-fast`
      (`K=1`); `keep_top_gate_values` never tunes the "none"/off variant.
- [x] Executor `run_ngram_stages`: `build_factors` screen-exclusion (gate only)
      and a `cfg.ngram_type` pin; screen child → rank → keep top-`K` → one
      `--ngram-type` tuning child per survivor (base held at screen winners,
      n_depth spread) → best MEASURED config across merged stages.
      (Realised via the orchestrator + `--ngram-type` children rather than a
      `refine_factors` gate transition — same intent, cleaner separation.)
- [x] `--ngram-type` flag (single-pass Stage k).
- [x] Banner reflects screen vs tune; report note that per-variant tuning is the
      staged flow. *(Not validated end-to-end on a GPU yet.)*
- [ ] Optional: let tuning stages themselves refine over `--iterate` passes
      (currently one tuning pass per variant).

Submodule:

- [ ] Bump submodule to ≥ `4d7fd61` on merge; re-run binding smoke test.
- [ ] Separate commit: rename `taguchi` → `robust` (dir + `.gitmodules` + refs),
      after confirming the Python package name.

Docs:

- [ ] README ngram section rewritten around the staged flow.
- [ ] Cross-link this + `CONDITIONAL-FACTORS.md` from `DESIGN.md`; tick ROADMAP.

## Tests

Full plan in `CONDITIONAL-FACTORS.md` ("Test plan"). ngram-specific cases: the
gate × child emission table (including `flag_for` spellings and `ngram=none`),
the MTP + ngram `--spec-type` merge (retained), the `refine_factors` stage-hop on
an `ngram-mod`-wins result set, and the L25-not-L125 sizing witness. All
pure-Python and GPU-free — they run offline while the card is in use.
