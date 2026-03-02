## Nonbonded Roadmap (Milestones 1-7)

### Brief Summary
This roadmap delivers two immediate milestones already agreed (cdie removal and Section 2 MVP rewrite), then adds five follow-up milestones in this order:

1. `cdie` removal + single-forcefield hardcoding.
2. Major Section 2 MVP architectural rewrite (manual codegen-equivalent boilerplate), with minimal Python forcefield module wiring.
3. Implement `codegen` flow end-to-end using a temporary `dummy` forcefield and validate build/discovery.
4. Validate capability-based codegen emission for grad+energy and energy-only wrappers.
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

5. Milestone 5:
- Add in-house grid generation interface and remove runtime requirement for external `.grid` files in that path.

6. Milestone 6:
- Ensure pure JAX path supports in-house grids without C kernel dependency.

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

5. Milestone 5: In-house grid generation (replace external `.grid` dependency)
- Implement internal grid generation pipeline.
- Keep existing external-grid path optionally for transition, but target path uses in-house grid artifacts.
- Required validation: concat scoring on first1000 + first10k must be **harness-strict** close to current references.

6. Milestone 6: In-house grid generation without C kernel (pure JAX)
- Run same in-house grid pipeline but force NB backend to JAX-only.
- Required validation: concat scoring only (first1000 + first10k), harness-strict thresholds.

7. Milestone 7: nonbon12 + cdie implementation
- Add forcefield implementation and integration points.
- No validation/test gate required in this milestone (explicit decision).
- Mark feature as provisional and testing-deferred in docs/notes.

---

### Test Cases and Scenarios

1. Scoring references (energy+gradient) already on disk:
- `/home/sjoerd/attract-namespace/test/first1000/first1000_target_nb.*.energy.npy`
- `/home/sjoerd/attract-namespace/test/first1000/first1000_target_nb.*.grad.npy`
- `/home/sjoerd/attract-namespace/test/first10k/first1000_target_nb.*.energy.npy`
- `/home/sjoerd/attract-namespace/test/first10k/first1000_target_nb.*.grad.npy`

2. Minimization references on disk (energy/pose/runtime artifacts):
- `/home/sjoerd/attract-namespace/test/first1000/minfor_jax_fused_first1000.*`
- `/home/sjoerd/attract-namespace/test/first1000/minfor_legacy_first1000.*`
- `/home/sjoerd/attract-namespace/test/first10k/minfor_jax_fused_first10k.*`
- `/home/sjoerd/attract-namespace/test/first10k/minfor_legacy_first10k.*`

3. Milestone 1-2 scoring gate:
- Use strict existing harness/score parity behavior (no relaxed threshold).

4. Milestone 1-2 minimization gate:
- No pose-by-pose correspondence requirement.
- Require similar runtime and similar energy/LRMSD distributions (existing decision).

5. Milestone 5-6 gate:
- Concat scoring only, first1000 + first10k, harness-strict closeness.

6. Milestone 7 gate:
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
8. Plan file target remains `/home/sjoerd/attract-namespace/plan-nonbonded-m1-m2.md` unless renamed later.
