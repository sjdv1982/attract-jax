## Nonbonded Roadmap (Milestones 1-7)

### Brief Summary
This roadmap delivers two immediate milestones already agreed (cdie removal and Section 2 MVP rewrite), then adds five follow-up milestones in this order:

1. `cdie` removal + single-forcefield hardcoding.
2. Major Section 2 MVP architectural rewrite (manual codegen-equivalent boilerplate), with minimal Python forcefield module wiring.
3. Implement `codegen` flow end-to-end using a temporary `dummy` forcefield and validate build/discovery.
4. Validate capability-based codegen emission for grad+energy and energy-only wrappers.
4a. Add plateau mode support to `pose_loop.h` (clamp vs. correction), validate existing scores are reproduced.
5. Move to in-house grid generation (replace dependency on external `.grid`) and validate via concat scoring.
6. Validate in-house grid generation on pure JAX (no C kernel) via concat scoring.
7. Add `nonbon12 + cdie` implementation (no test requirement yet, per decision).

---

### Public API / Interface / Type Changes

1. Milestone 1:
- Remove `--cdie` from maintained CLIs.
- Remove runtime `cdie` branches from maintained code.
- Remove runtime `potshape` branches from maintained code and hardcode nonbon8 behavior.

2. Milestone 2:
- Introduce Section 2 policy/template kernel structure (MVP subset).
- Expose manual wrapper symbols equivalent to codegen output for MVP combinations.
- Keep external `.grid` input path intact.
- Add minimal Section 5.1 Python forcefield module contract for nonbon8 (`lj_energy`, `elec_energy`, `load_params`).

3. Milestone 3:
- Add `codegen_ff.py init` and `codegen_ff.py codegen` sufficient to reproduce nonbon8-generated wrapper output shape.
- Makefile auto-build path must work for generated forcefield outputs.
- Python forcefield discovery must load generated forcefield (`dummy`) correctly.

4. Milestone 4:
- Validate that codegen emits wrapper variants from FF capabilities:
  - `lj_grad.h` + `elec_grad.h` present -> grad+energy wrappers.
  - either gradient header missing -> energy-only wrappers.
- Confirm Python symbol probing/dispatch correctly selects available wrapper families.

4a. Milestone 4a:
- Add plateau mode as a template parameter to `pose_loop.h`: `correction` (current behavior: `E(d) - E(plateaudis)`) and `clamp` (grid precomputation: `E(max(d, plateaudis))`).
- Same split for gradients.
- Codegen emits wrappers for both modes.
- Validate that existing concat scoring (first1000 + first10k) is reproduced exactly with `correction` mode — no regression.

5. Milestone 5:
- Add in-house grid generation interface and remove runtime requirement for external `.grid` files in that path.
- Add Python NB dispatch abstraction: a `score_pairs()` function that routes to either JAX (`vmap(ff_module.lj_energy)` etc.) or C kernel (voxel-as-pose via `pose_loop.h`), selected by backend flag.
- Grid precomputation and NB correction both call through this dispatch — the only difference is input preparation.

6. Milestone 6:
- Ensure pure JAX path supports in-house grids without C kernel dependency.
- Validate that the NB dispatch with `backend="jax"` produces identical results to `backend="kernel"` on the same inputs.

7. Milestone 7:
- Add second FF implementation: nonbon12 + cdie (feature present, no validation gate yet).

---

### Milestone-by-Milestone Plan

1. Milestone 1: Remove `cdie`, hardcode single FF (`rdie`, nonbon8)
- Update production kernel path and maintained harness kernels (`first1000`, `first10k`).
- Remove `cdie` and `potshape` runtime branching in maintained paths.
- Remove `--cdie` from maintained CLI surfaces.
- Keep current `.grid` reading behavior unchanged.
- Validate with current first1000/first10k scoring and minimization gates.

2. Milestone 2: Section 2 MVP rewrite (manual boilerplate), minimal Python FF wiring
- Refactor C kernel to policy/template architecture:
  - RotPolicy: Euler only.
  - FFPolicy: nonbon8 only.
  - ComputeGrad: gradient path required.
- Manually create codegen-equivalent boilerplate for this MVP.
- Keep external `.grid` loading for all runtime paths.
- Add minimal Section 3.2/5.1-aligned Python forcefield package wiring for nonbon8.
- Validate:
  - Scoring: concat first1000 and first10k, strict parity.
  - Minimization: distribution/runtime similarity (no pose-by-pose correspondence).

3. Milestone 3: Codegen implementation via temporary `dummy` forcefield
- Implement `codegen_ff.py init` to create a dummy FF scaffold.
- Copy nonbon8 physics `.h` files into dummy FF.
- Implement `codegen_ff.py codegen` so generated C++ output for `dummy` matches nonbon8 content/shape expectations.
- Build and run using `dummy`; verify Makefile discovery and Python FF discovery operate end-to-end.
- Delete dummy FF after proving workflow.
- Deliver codegen tests that assert generated output equivalence properties (content and symbols).

4. Milestone 4: Validate capability-based wrapper emission and dispatch
- Confirm codegen capability detection works exactly as specified:
  - `lj_grad.h` + `elec_grad.h` present -> grad+energy wrapper set.
  - either gradient header missing -> energy-only wrapper set.
- Remove any remaining manual wrapper fallback in maintained paths.
- Validate generated symbols are callable and selected correctly by Python.
- Regression check: where both variants are available, energy-only scoring equals the energy component of grad+energy within strict tolerance.

4a. Milestone 4a: Plateau mode support in `pose_loop.h`
- Currently `pose_loop.h` hardcodes the NB correction plateau behavior: `E(d) - E(plateaudis)`.
  Grid precomputation needs the opposite: `E(max(d, plateaudis))` (clamp mode).
- Add a template parameter (or compile-time flag) to `pose_loop.h` selecting between:
  - `correction` mode (existing behavior): compute `E(d_actual) - E(plateaudis)` and
    `G(d_actual) - G(plateaudis)`. Used at runtime for NB correction.
  - `clamp` mode (new): compute `E(max(d_actual, plateaudis))` and
    `G(max(d_actual, plateaudis))`. Used for grid precomputation.
- Codegen (`codegen_ff.py`) must emit wrapper symbols for both modes.
- The same plateau mode split applies to both LJ and electrostatic terms, and to both
  energy and gradient paths.
- **Validation gate:** Run existing concat scoring (first1000 + first10k) with `correction`
  mode. Scores must be identical to current references — zero regression.
- The `clamp` mode is not yet validated end-to-end (that happens in M5 when grids are
  generated in-house). M4a only proves that the mode parameterization compiles, links,
  and does not break existing behavior.

5. Milestone 5: In-house grid generation (replace external `.grid` dependency)
- Implement internal grid generation pipeline.
- Keep existing external-grid path optionally for transition, but target path uses in-house grid artifacts.
- Required validation: concat scoring on first1000 + first10k must be **harness-strict** close to current references.

#### Milestone 5 implementation detail

**What to build:** Two things:

**(A) In-house grid generator.** A Python module that generates grid data (potential grids +
neighbour grids) from receptor ensemble coordinates + force field parameters, replacing the
legacy ATTRACT `make-grid` C binary. The output must be a data structure compatible with what
`read_grid_with_electro()` in `util/reproduce_grid_score.py` currently returns (the `Grid`
named tuple), so that everything downstream — `JaxScoreOracle`, `build_kernel`, the test
harnesses — works without modification.

**(B) Python NB dispatch abstraction.** A function that presents a uniform interface for
pairwise NB evaluation and routes to either JAX or the C kernel:

```python
def score_pairs(coords_i, neighbor_data, ff_params, ff_module, backend="jax"):
    """Evaluate pairwise NB energy (and optionally gradients) over neighbor lists.

    backend="jax":    vmap(ff_module.lj_energy) + vmap(ff_module.elec_energy)
    backend="kernel": wrap inputs as pseudo-poses, call pose_loop via ctypes
    """
```

Both the grid generator (computing potential grid values at voxels) and the runtime NB
correction (computing NB energy for ligand atoms against their neighbors) call through this
dispatch. The differences are:
- Grid precomputation: `coords_i` = voxel centers, identity rotation, one "pose" per voxel.
- NB correction: `coords_i` = transformed ligand atom positions, from the pose DOFs.
- **Plateau distance handling differs between the two uses** (see below).

**Plateau distance correction — grid vs. NB correction:**
The grid and the NB correction together must produce the correct total energy `E(d_actual)`.
They split the work using the plateau distance `plateaudis`:
- **Grid precomputation** stores the energy (and gradients) evaluated with distances clamped
  *at* plateau distance: `d = max(d_actual, plateaudis)`. This gives a smooth potential that
  avoids singularities at short range.
- **Runtime NB correction** computes the difference `E(d_actual) - E(plateaudis)` for each
  nearby atom pair, adding the missing short-range contribution back. For pairs where
  `d_actual >= plateaudis`, this correction is zero.
- The same split applies to gradients: the grid stores gradients at `d = max(d_actual,
  plateaudis)`, and the NB correction adds the gradient difference.
- This means `score_pairs()` needs a mode parameter (e.g. `plateau_mode="clamp"|"correction"`)
  or the grid and NB correction paths must call different energy functions. The dispatch
  cannot treat them as identical computations with different inputs — the plateau handling
  is fundamentally different.

**C kernel strategy — voxel-as-pose (no kernel restructuring beyond M4a):**
The C kernel's only entry point remains `pose_loop.h`. For grid precomputation, each voxel is
wrapped as a pseudo-pose: rotation = identity, translation = voxel center. The pose loop's
existing neighbor iteration, parameter loading, and OpenMP parallelism apply unchanged.
There are only two consumers of the NB kernel (NB correction and grid precomputation), both
served by pose_loop. The plateau mode template parameter added in M4a selects between
`correction` (runtime NB) and `clamp` (grid precomputation).

The dispatch abstraction lives entirely on the Python side. No new C entry points are needed
beyond the two wrapper variants (correction + clamp) emitted by codegen in M4a.

**What already exists — DO NOT reimplement:**
- `util/reproduce_grid_score.py`: contains `read_grid_with_electro()` (grid binary parser,
  returns `Grid` named tuple), `build_kernel()` (returns JAX scoring callable with `.ad` and
  `.pot_ad` variants), `nonbon()` / `elec_dsq_*()` (JAX pairwise energy functions),
  `potential_atom_energies()` (trilinear grid interpolation with custom_jvp).
  **Do not rewrite any of these.** The new grid generator produces a `Grid`-compatible object;
  existing code consumes it unchanged.
- `util/jax_scorer.py`: `JaxScoreOracle` — the scoring oracle used by `minfor.py`. Already
  supports `nb_kernel="jax"` (pure JAX) and `nb_kernel="nonbon8"` (C kernel). Already accepts
  a `grid_file` path. The change for M5 is to also accept a pre-built `Grid` object (or a
  new path type pointing to in-house grid output), not to rewrite the oracle.
- `util/minfor.py`: the CLI entry point. Already has `--grid`, `--oracle jax`,
  `--nb-kernel nonbon8`, `--attract-par-npz`, `--autodiff-potentials`. **Do not rewrite the
  CLI or the scoring pipeline.** The only CLI change is to support an alternative to `--grid`
  that invokes in-house grid generation (e.g. `--generate-grid` with receptor PDB inputs, or
  auto-generation when `--grid` is not passed).
- `native/nb_kernel/forcefields/nonbon8/`: Python reference functions (`lj.py`, `elec.py`,
  `params.py`) and C headers. The in-house grid generator should use the Python energy
  functions from the force field module (Section 5 of the high-level plan), not hardcode
  the physics.
- Test shell scripts: `test/first1000/test_first1000_concat_score.sh` and
  `test/first10k/test_first10k_concat_score.sh` — these run `minfor.py --score --oracle jax
  --nb-kernel nonbon8` and compare output against reference `.score` files. **Do not rewrite
  these scripts.** Validation means: run the existing test scripts (possibly with a modified
  `--grid` argument pointing to in-house output) and confirm the scores match references.

**Early prototype for reference (not production-ready):**
- `test-calc-grid-energy.py` (lines 94-103): builds a `cKDTree` from receptor coordinates,
  queries grid voxel centers, and computes LJ potentials. Single atom type only, no
  electrostatics, no ensemble support. Use as conceptual reference for the KD-tree approach,
  but do not try to extend this script — write a clean module.

**Grid culling scripts (optional, not part of M5 core):**
- `playground/precompute_interior_voxels.py`: patches an existing `.grid` binary to zero out
  interior voxels using KD-tree pre-filter + energy lower-bound check. This is a post-
  processing optimization, not grid generation. It may be integrated later but is not required
  for M5.

**Critical: voxel corner vs. center conventions (from `make-grid.cpp` / `grid_calculate.cpp`):**

The potential grid and the neighbour grid use **different spatial conventions**. Getting this
wrong will produce silently incorrect results. The reference implementation is in
`attract/bin/make-grid.cpp` → `grid_calculate.cpp`.

*Potential grid — values at voxel CORNERS:*
- Voxel `(i, j, k)` stores the potential at position `ori + (i, j, k) * gridspacing`.
  There is no half-voxel offset.
- At scoring time, a ligand atom at continuous position `p` is mapped to fractional voxel
  coordinates `v = (p - ori) / gridspacing`, and the potential is trilinearly interpolated
  between the 8 surrounding corner values using `floor(v)` and `ceil(v)` with fractional
  weights `w = v - floor(v)`.
- The same convention applies to the gradient channels (gx, gy, gz) stored alongside energy.
- Reference: `grid_calculate.cpp` uses `diszyx` (no 0.5 offset) for `_calc_potential()`.
  Scoring: `grid.cpp` `trilin()` uses `floor/ceil` with fractional weights.
  Python: `reproduce_grid_score.py` `interpolate_grid()` — identical trilinear interpolation.

*Neighbour grid — lists at voxel CENTERS (half-voxel shifted):*
- The neighbour list for voxel `(i, j, k)` contains receptor atoms within `neighbourdis` of
  position `ori + (i+0.5, j+0.5, k+0.5) * gridspacing` — the **center** of the voxel, not
  the corner.
- At scoring time, the neighbour list is looked up by **nearest-integer rounding**:
  `floor(v + 0.5)`, which selects the voxel whose center is closest to the ligand atom.
  There is no interpolation — a single voxel's neighbour list is used.
- Reference: `grid_calculate.cpp` uses `disjzyx` (shifted by `+0.5*gridspacing` in all axes)
  for `_calc_neighbours()`. Scoring: `grid.cpp` uses `floor(ax + 0.5)`.
  Python: `reproduce_grid_score.py` `generate_ind_innergrid()` uses `floor(vox + 0.5)`.

*Outer (big) grid:*
- Same corner convention for potentials, but at double the spacing.
- Voxel coordinates: `vox_outer = (vox_inner + gridextension) / 2.0`.

**The in-house grid generator MUST reproduce these exact conventions.** If potential values
are accidentally computed at voxel centers, or neighbours are built from voxel corners,
the resulting scores will be wrong by O(gridspacing) in position — enough to fail validation.

**What the in-house grid generator must produce:**
The `Grid` named tuple fields (see `read_grid_with_electro` return type):
- `inner_potential_grid`: shape `(nr_vdw_channels, dx, dy, dz, 4)` — energy + 3 gradient
  components per atom type per voxel.
- `outer_potential_grid`: same shape, for the outer (coarser) grid region.
- `inner_elec_grid` / `outer_elec_grid`: electrostatic potential grids.
- `neighbour_grid`: shape `(dx, dy, dz, max_nr_neighbours)` — receptor atom indices (uint16).
- `nr_neighbours`: shape `(dx, dy, dz)` — count of neighbours per voxel.
- `max_nr_neighbours`, `alphabet_atomtypes`, `plateaudis`, `gridspacing`, `dim`, `dim2`,
  `origin`: scalar/array metadata.

The downstream code indexes into these arrays by position. As long as the shapes and semantics
match, the source (legacy binary vs. in-house Python) is irrelevant.

6. Milestone 6: In-house grid generation without C kernel (pure JAX)
- Run same in-house grid pipeline but force NB backend to JAX-only.
- Required validation: concat scoring only (first1000 + first10k), harness-strict thresholds.

#### Milestone 6 implementation detail

**What M6 actually means:** M5 builds two things: the grid generator and the NB dispatch
abstraction (`score_pairs()` with `backend="jax"|"kernel"`). M6 confirms that the entire
pipeline works when `backend="jax"` everywhere — both grid precomputation and runtime scoring.
This is mostly a validation milestone, not a major implementation milestone.

**What already exists — DO NOT reimplement:**
- `JaxScoreOracle` with `nb_kernel="jax"`: this path already works. It uses `main_ad` from
  `build_kernel()` which evaluates both potential grid lookup and NB correction in pure JAX,
  with gradients via `jax.value_and_grad` + `jax.vmap`. No C code is involved.
- `util/minfor_nb.py`: pure JAX NB evaluation functions (`build_nb_grad_fn`,
  `_single_nb_energy`, `nb_energy_vectorized`). These are already used by the `nb_kernel="jax"`
  path. **Do not rewrite them.**
- The existing test scripts already support `--nb-kernel jax` (just change the CLI flag from
  `nonbon8` to `jax` in the invocation — or add a parallel test invocation).
- The `score_pairs()` dispatch built in M5 — M6 just calls it with `backend="jax"`.

**What to validate for M6:**

1. **Grid precomputation with `backend="jax"`:** The M5 grid generator calls `score_pairs()`
   to evaluate pairwise energies at each voxel. When `backend="jax"`, this uses
   `vmap(ff_module.lj_energy)` etc. Confirm the resulting grid is numerically identical
   (within tolerance) to the grid produced with `backend="kernel"`.
2. **Runtime scoring with `--nb-kernel jax`:** Run the concat scoring test scripts with
   `--nb-kernel jax` and the in-house grid from step 1. Confirm scores match references
   within harness-strict tolerance.
3. **Cross-backend consistency:** `score_pairs(..., backend="jax")` and
   `score_pairs(..., backend="kernel")` must agree within strict tolerance on the same inputs.
   This is a unit-level check on the dispatch itself, independent of the full scoring pipeline.

**What to build for M6:**
- A variant of the concat scoring test scripts (or a flag/mode in the existing ones) that
  runs with `--nb-kernel jax` instead of `--nb-kernel nonbon8`.
- A unit test for `score_pairs()` cross-backend consistency (same inputs, both backends,
  compare outputs).

**What M6 is NOT:**
- It is NOT a rewrite of the JAX scoring pipeline.
- It is NOT a new grid format or new grid generator.
- It is NOT a new test harness. The existing `test_first1000_concat_score.sh` /
  `test_first10k_concat_score.sh` scripts are the validation mechanism — either modified to
  also run the `--nb-kernel jax` variant, or accompanied by a thin wrapper that does so.

**Key risk for agents:** An agent reading "In-house grid generation without C kernel (pure JAX)"
may interpret this as "implement a pure-JAX grid generation + scoring system from scratch."
This is wrong. The pure JAX scoring path already exists (`nb_kernel="jax"`). The in-house grid
generator and NB dispatch are built in M5. M6 is the *intersection*: confirm everything works
together with `backend="jax"`, and that the two backends agree.

7. Milestone 7: nonbon12 + cdie implementation
- Add forcefield implementation and integration points.
- No validation/test gate required in this milestone (explicit decision).
- Mark feature as provisional and testing-deferred in docs/notes.

---

### Test Cases and Scenarios

1. Canonical rerun scripts (one script per case, each takes `OUT_DIR`):
- `/home/sjoerd/attract-namespace/test/first1000/test_first1000_concat_score.sh`
- `/home/sjoerd/attract-namespace/test/first10k/test_first10k_concat_score.sh`
- `/home/sjoerd/attract-namespace/test/first1000/test_first1000_minimization.sh`
- `/home/sjoerd/attract-namespace/test/first10k/test_first10k_minimization.sh`

2. Inputs used by the four scripts:
- `/home/sjoerd/attract-namespace/test/first1000/tmp_active_concat.dat`
- `/home/sjoerd/attract-namespace/test/first10k/tmp_active_concat.dat`
- `/home/sjoerd/attract-namespace/test/systsearch-ens1-first1000.dat`
- `/home/sjoerd/attract-namespace/test/systsearch-ens1-first10000.dat`
- `/home/sjoerd/attract-namespace/test/receptorgrid.grid`
- `/home/sjoerd/attract-namespace/test/partner1-ensemble.list`
- `/home/sjoerd/attract-namespace/test/ligandr.pdb`
- `/home/sjoerd/data/work/attract-jax/attract-par.npz`

3. Score references for energy/gradient checks:
- `/home/sjoerd/attract-namespace/test/first1000/score_legacy_first1000_concat_after_gridexcise.score`
- `/home/sjoerd/attract-namespace/test/first1000/score_jax_fused_first1000_concat_pregridexcise_style.score`
- `/home/sjoerd/attract-namespace/test/first1000/first1000_target_nb.*.energy.npy`
- `/home/sjoerd/attract-namespace/test/first1000/first1000_target_nb.*.grad.npy`
- `/home/sjoerd/attract-namespace/test/first10k/first1000_target_nb.*.energy.npy`
- `/home/sjoerd/attract-namespace/test/first10k/first1000_target_nb.*.grad.npy`

4. Minimization references on disk (JAX/legacy):
- `/home/sjoerd/attract-namespace/test/first1000/minfor_jax_fused_first1000.*`
- `/home/sjoerd/attract-namespace/test/first1000/minfor_legacy_first1000.*`
- `/home/sjoerd/attract-namespace/test/first10k/minfor_jax_fused_first10k.*`
- `/home/sjoerd/attract-namespace/test/first10k/minfor_legacy_first10k.*`
- `/home/sjoerd/attract-namespace/test/minfor_jax_fused_first10k_after_gridexcise.*`

5. LRMSD reference inputs:
- `/home/sjoerd/attract-namespace/test/ligand-heavy.pdb`
- `/home/sjoerd/attract-namespace/test/refe-rmsd-2.pdb`
- `/home/sjoerd/attract-namespace/test/partner1-ensemble-aa-rmsd.list`
- `/home/sjoerd/attract-namespace/test/partner1-ensemble/model-1-heavy.pdb`

6. Latest full four-case rerun artifacts (2026-03-02):
- `/tmp/status_case_runs_retry_20260302_163727/1_first1000_concat/`
- `/tmp/status_case_runs_retry_20260302_163727/2_first10k_concat/`
- `/tmp/status_case_runs_retry_20260302_163727/3_first1000_min_rerun/`
- `/tmp/status_case_runs_retry_20260302_163727/4_first10k_min_rerun/`
- `/tmp/status_case_runs_retry_20260302_163727/validation_summary.json`

7. Milestone 1-2 scoring gate:
- Use strict existing harness/score parity behavior (no relaxed threshold).

8. Milestone 1-2 minimization gate:
- No pose-by-pose correspondence requirement.
- Require similar runtime and similar energy/LRMSD distributions (existing decision).

9. Milestone 5-6 gate:
- Concat scoring only, first1000 + first10k, harness-strict closeness.

10. Milestone 7 gate:
- No tests required yet.

---

### Explicit Assumptions and Defaults

1. Single-forcefield focus remains through Milestone 6.
2. `cdie` and runtime `potshape` removed early (Milestone 1).
3. External `.grid` files are kept through Milestone 4.
4. Grid migration is intentionally late (Milestone 5).
5. Rotvec is deferred; not part of this roadmap segment.
6. Follow-up after Milestone 2 starts with codegen implementation.
7. nonbon12 milestone lands without test obligations for now.
8. Plan file target remains `/home/sjoerd/attract-namespace/attract-jax/plan-nonbonded-m1-m2.md` unless renamed later.
