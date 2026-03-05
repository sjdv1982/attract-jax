# JAX Minimization — Current Status (2026-03-02)

## Objective

- Production objective: achieve end-to-end `minfor.py` performance comparable to legacy ATTRACT on correct settings.
- Success criterion: "comparable to legacy" means adjusted per-pose runtime or adjusted wall-clock runtime is within `+/-10%` of the matched legacy baseline.
- Scope of this file: current snapshot only. Historical investigations are intentionally omitted and remain available in Git history.

## Definitions

- `Research path`: pure-JAX implementation used for rapid iteration and AD-first development.
- `Production path`: fused native NB kernel plus optimized potential-grid path.
- `Comparable to legacy`: within `+/-10%` of legacy under matched dataset, configuration, and timing protocol.
- `Measured`: value produced by an executed benchmark with saved artifacts.
- `Expected`: forward-looking estimate that still requires benchmark confirmation.
- `Adjusted timing`: concat timing minus subtraction timing (`sub4`) using the same protocol as legacy comparisons.

## Current Facts (Measured)

Measurements in this table are from 2026-02-28, 2026-03-01, and 2026-03-02.

| Path | Dataset | Metric | Result | Legacy Delta | Status (Measured/Expected) |
|---|---|---|---|---|---|
| Potential-grid only (`score_potential_batch`, zero-NB) | `first1000` active-only concat (`test/first1000/tmp_active_concat.dat`) | Adjusted wall time | `27.97 s` | `+6.83%` slower than legacy zero-NB baseline | Measured |
| Potential-grid only (`score_potential_batch`, zero-NB) | `first10k` active-only concat (`test/first10k/tmp_active_concat.dat`) | Adjusted wall time | `312.90 s` | `+8.81%` slower than legacy zero-NB baseline | Measured |
| Legacy reference (`attract --score`) | `first1000` active-only concat, normal grid (`106003` poses) | Adjusted wall time | `30.03 s` (`30.30 - 0.27`) | Baseline | Measured |
| Production path (`minfor.py --score --oracle jax --nb-kernel fused`) | `first1000` active-only concat, normal grid (`106003` poses) | Adjusted wall time | `29.07 s` (`39.26 - 10.19`) | `-3.2%` faster than matched legacy run | Measured |
| Legacy reference (`attract --score`) | `first10k` active-only concat, normal grid (`1051136` poses) | Adjusted wall time | `335.78 s` (`336.06 - 0.28`) | Baseline | Measured |
| Production path (`minfor.py --score --oracle jax --nb-kernel fused`) | `first10k` active-only concat, normal grid (`1051136` poses) | Adjusted wall time | `311.50 s` (`323.10 - 11.60`) | `-7.2%` faster than matched legacy run | Measured |
| Production path (`minfor.py --score --oracle jax --nb-kernel fused`, torque-matrix rotational reduction) | `first10k` active-only concat, normal grid (`1051136` poses) | Adjusted wall time | `308.82 s` (`318.19 - 9.37`) | `-8.0%` faster than matched legacy run, `-0.86%` vs prior fused score baseline (`311.50 s`) | Measured |
| Legacy minimization (`minfor.py`, legacy oracle) | `first1000` (`systsearch-ens1-first1000.dat`) | Total wall time | `85.6 s` | Baseline | Measured |
| Production minimization (`minfor.py --oracle jax --nb-kernel fused`) | `first1000` (`systsearch-ens1-first1000.dat`) | Total wall time | `93.8 s` | `+9.6%` slower than matched legacy run | Measured |
| Legacy minimization (`minfor.py`, legacy oracle) | `first10k` (`systsearch-ens1-first10000.dat`) | Total wall time | `457.0 s` | Baseline | Measured |
| Production minimization (`minfor.py --oracle jax --nb-kernel fused`) | `first10k` (`systsearch-ens1-first10000.dat`) | Total wall time | `414.0 s` | `-9.4%` faster than matched legacy run | Measured |
| Production minimization (post grid-gradient excision, `minfor.py --oracle jax --nb-kernel fused --autodiff-potentials`) | `first10k` (`systsearch-ens1-first10000.dat`) | Minimizer wall time / total wall time | `324.5 s` / `333.8 s` | `-21.6%` vs prior fused (`414.0 s`) and `-27.0%` vs legacy (`457.0 s`) | Measured (suspected outlier; reproducibility pending) |
| Production minimization (`--autodiff-potentials`, torque-matrix rotational reduction in fused NB) | `first10k` (`systsearch-ens1-first10000.dat`) | Minimizer wall time / total wall time | `401.4 s` / `412.2 s` | close to prior fused baseline (`414.0 s`), much slower than the isolated `333.8 s` record | Measured |
| Fused score correctness gate (pre vs post grid-gradient excision) | `first1000` active-only concat (`106003` poses) | Energy/gradient agreement | Energy identical (`max_abs=0.0`), gradients close but not bitwise-identical (`max_abs=35.11` at ~`1e9` scale) | Confirms no energy regression from excision | Measured |
| Production minimization (`minfor.py --oracle jax --nb-kernel fused --report-step-complete`) | full `systsearch-ens1.dat` (`165528` poses) | Total wall time | `5723.6 s` | no matched full-set legacy run recorded in this snapshot | Measured |
| Production minimization (post grid-gradient excision, streaming run) | full `systsearch-ens1.dat` (`165528` poses) | Early-run checkpoint only | up to tick `25` (`1421.1 s`) recorded | not comparable to final wall time; run not completed in this log | Measured (partial) |
| Production minimization (`--autodiff-potentials`, run 1) | full `systsearch-ens1.dat` (`165528` poses) | Minimizer wall time / total wall time | `6259.0 s` / `6319.3 s` | `+5.2%` slower than latest stored-gradient run (`5958.7 s` total wall) | Measured |
| Production minimization (`--autodiff-potentials`, run 2) | full `systsearch-ens1.dat` (`165528` poses) | Minimizer wall time / total wall time | `6240.5 s` / `6300.1 s` | `+5.7%` slower than latest stored-gradient run (`5958.7 s` total wall) | Measured |
| Production minimization (default stored gradients + custom JVP, run 1) | full `systsearch-ens1.dat` (`165528` poses) | Minimizer wall time / total wall time | `5901.3 s` / `5958.7 s` | fastest among the 2026-03-01 full-run trio | Measured |
| RMSD analysis (`--autodiff-potentials`, run 1) | full `systsearch-ens1.dat` top-1000-by-energy | best-10 LRMSD summary | best LRMSD `1.948`, median LRMSD(top1000) `51.292` | top-10 LRMSD list with ranks recorded | Measured |
| RMSD analysis (`--autodiff-potentials`, run 2) | full `systsearch-ens1.dat` top-1000-by-energy | best-10 LRMSD summary | best LRMSD `1.948`, median LRMSD(top1000) `51.292` | identical to run 1 on this metric | Measured |
| RMSD analysis (default stored gradients + custom JVP, run 1) | full `systsearch-ens1.dat` top-1000-by-energy | best-10 LRMSD summary | best LRMSD `3.151`, median LRMSD(top1000) `50.824` | better median, weaker best-hit than autodiff runs | Measured |
| LRMSD stability check (first10k, torque-matrix vs prior post-excision run) | `systsearch-ens1-first10000.dat` (`10000` poses) | per-pose LRMSD delta (`torque - prior`) | mean `+0.012`, median `0.000`, mean abs `0.258`, unchanged `5511/10000` poses | no material LRMSD distribution shift | Measured |

Current measured conclusion:

- Potential-grid path remains near parity (`~7-9%` slower than legacy on zero-NB benchmarks).
- Production fused-NB score path remains faster than legacy on measured `first10k` active-concat scoring (`308.82 s` adjusted vs legacy `335.78 s`).
- Grid-gradient excision (`--autodiff-potentials`) preserved fused energy outputs on `first1000` pre-vs-post comparison (`max_abs=0.0`), while gradients changed (not bitwise-identical).
- Full-set controlled reruns (2026-03-01) show stored gradients + custom JVP is faster than autodiff on wall time for this workload (`5958.7 s` vs `~6300 s`).
- Full-set RMSD trade-off (top-1000-by-energy): autodiff found a lower best LRMSD hit (`1.948`) while stored-gradient mode had a slightly better median LRMSD (`50.824` vs `51.292`).
- First10k minimization rerun on 2026-03-02 with torque-matrix rotational reduction (`401.4 s` minimizer / `412.2 s` total) is consistent with the historical fused baseline (`414.0 s`) and does not reproduce the isolated `333.8 s` record.
- The 2026-02-28 `333.8 s` first10k post-excision result is treated as a suspected outlier until reproduced under the same controlled benchmark protocol.
- Pure-JAX NB remains a research path; production scoring/minimization should use `--nb-kernel fused`.
- Full production-scale minimization on `165528` poses completes successfully with step-complete progress reporting enabled.

## 2026-03-02 Four-Case Script Rerun (Measured)

Rerun root:

- `/tmp/status_case_runs_retry_20260302_163727/`
- Consolidated validation: `/tmp/status_case_runs_retry_20260302_163727/validation_summary.json`

Case timings (`minfor.py` wall timings captured by the per-case scripts):

| Case | Pregrad | Autodiff | STATUS reference | Check |
|---|---:|---:|---:|---|
| `first1000` concat score (`tmp_active_concat.dat`) | `54.229 s` | `51.792 s` | `39.26 s` (pregrad style) | Slower than STATUS baseline (`+38.1%`) |
| `first10k` concat score (`tmp_active_concat.dat`) | `426.294 s` | `426.722 s` | `323.10 s` (pregrad style) | Slower than STATUS baseline (`+31.9%`) |
| `first1000` minimization (`systsearch-ens1-first1000.dat`) | `86.333 s` | `69.857 s` | `93.8 s` (pregrad style) | Pregrad faster than STATUS baseline (`-8.0%`) |
| `first10k` minimization (`systsearch-ens1-first10000.dat`) | `378.791 s` | `402.703 s` | `414.0 s` pregrad, `333.8 s` autodiff | Pregrad faster (`-8.5%`), autodiff slower than isolated `333.8 s` record (`+20.6%`) |

Energy/gradient verification from `.score` outputs:

- `first1000` legacy vs rerun pregrad score (`106003` poses): energy `max_abs=1828.3125`, gradient `max_abs=6621184.0`; median absolute deltas are near zero (`energy 2.84e-14`, `grad 4.24e-08`).
- `first1000` rerun pregrad vs rerun autodiff score: energy identical (`max_abs=0.0`), gradients differ (`max_abs=35.1103`, `mean_abs=0.7077`), consistent with prior grid-gradient excision behavior.
- `first10k` rerun pregrad vs rerun autodiff score (`1051136` poses): energy identical (`max_abs=0.0`), gradients differ (`max_abs=36.6500`, `mean_abs=0.7798`).
- Existing `first10k` reference pregrad score file at `test/first10k/score_jax_fused_first10k_concat_pregridexcise_style.score` is currently empty (`0` bytes), so this specific direct file-vs-file score comparison was skipped.

Minimization energy-array verification against on-disk references:

- `first1000` pregrad rerun vs `test/first1000/minfor_jax_fused_first1000.energy.npy`: `max_abs=10.6788`, `mean_abs=0.1082`, `median_abs=1.91e-06`.
- `first10k` pregrad rerun vs `test/first10k/minfor_jax_fused_first10k.energy.npy`: `max_abs=13.5494`, `mean_abs=0.1555`, `median_abs=1.91e-06`.
- `first10k` autodiff rerun vs `test/minfor_jax_fused_first10k_after_gridexcise.energy.npy`: `max_abs=18.4129`, `mean_abs=0.2014`, `median_abs=5.72e-06`.

LRMSD verification:

- `first10k` autodiff rerun LRMSD vs STATUS artifact `test/minfor_jax_fused_first10k_after_gridexcise.lrmsd`: `max_abs=25.119`, `mean_abs=0.2577`, `median_abs=0.0` (large overlap, with outlier tail differences).
- `first10k` rerun summaries:
  - pregrad: best `14.413`, median `63.5435`, p10 `43.5241`
  - autodiff: best `11.017`, median `64.1995`, p10 `43.8858`

## 2026-03-05 Timing Rerun (after milestone 4a, Measured)

Rerun roots:

- `/home/sjoerd/attract-namespace/attract-jax/validation/m4a/timings_rerun_first1000/`
- `/home/sjoerd/attract-namespace/attract-jax/validation/m4a/timings_rerun_first10k/`

Command protocol:

- `test/first1000/test_first1000_concat_score.sh <OUT_DIR>`
- `test/first10k/test_first10k_concat_score.sh <OUT_DIR>`

Concat-score timing snapshot (after milestone 4a):

| Case | Pregrad | Autodiff | STATUS pregrad reference | Check |
|---|---:|---:|---:|---|
| `first1000` concat score (`tmp_active_concat.dat`) | `67.604 s` | `60.837 s` | `39.26 s` | Pregrad `+72.20%`, autodiff `+54.96%` vs STATUS pregrad reference |
| `first10k` concat score (`tmp_active_concat.dat`) | `401.676 s` | `428.477 s` | `323.10 s` | Pregrad `+24.32%`, autodiff `+32.61%` vs STATUS pregrad reference |

Harness status lines:

- `first1000_concat_pregrad_vs_status WARN measured=67.604s expected=39.260s delta_pct=+72.20% tol_pct=20.0%`
- `first10k_concat_pregrad_vs_status WARN measured=401.676s expected=323.100s delta_pct=+24.32% tol_pct=20.0%`

## Energy-Only Score Output Update (2026-03-04)

Objective of this update:

- Add `--energy-only` support in `minfor.py` for `--score` mode.
- Ensure both subsystems run in energy-only mode under this flag:
  - JAX potential-grid path: no potential-gradient AD path.
  - NB kernel path: energy wrapper (`nb_kernel_euler_energy`) only.
- Keep score output in legacy-style energy blocks without `Gradients:` lines.

Final measured timings (`conda run -n jax python util/minfor.py ... --score --energy-only --oracle jax --nb-kernel nonbon8`):

| Case | Poses | Wall time |
|---|---:|---:|
| `first1000` concat (`test/first1000/tmp_active_concat.dat`) | `106003` | `50.455 s` |
| `first10k` concat (`test/first10k/tmp_active_concat.dat`) | `1051136` | `362.386 s` |

Energy comparison against stored gradient-block references (roundoff-scale differences accepted):

| Case | Stored reference | Energy lines | `max_abs` | `mean_abs` | `Gradients:` present in new output |
|---|---|---:|---:|---:|---|
| `first1000` concat | `/tmp/status_case_runs_retry_20260302_163727/1_first1000_concat/score_jax_fused_first1000_concat_pregridexcise_style.score` | `106003` | `0.0625` | `1.373e-06` | `no` |
| `first10k` concat | `/tmp/status_case_runs_retry_20260302_163727/2_first10k_concat/score_jax_fused_first10k_concat_pregridexcise_style.score` | `1051136` | `64.0` | `9.433e-05` | `no` |

Observed speedup versus the intermediate run where NB still used grad wrappers (`2026-03-04`):

- `first1000`: `98.950 s -> 50.455 s` (`~1.96x` faster).
- `first10k`: `383.840 s -> 362.386 s` (`~5.6%` faster).

Artifacts:

- Final run outputs/timings: `/tmp/m4_energy_only_v5_20260304_124319/`
- Final comparison JSON: `/tmp/m4_energy_only_v5_20260304_124319/energy_compare.json`

## Architecture Status

### Research Path (JAX-only)

The current JAX NB implementation remains a research/prototyping path.

- Bucketing remains a viable potential optimization for this path (not a rejected idea).
- Bucketing is currently removed from the `minfor.py` production-oriented stack and can be revisited later with a crocodile-style design if needed.
- Reference source for a future JAX NB-grid bucketing implementation: `crocodile-score-attract-jax.py` (in `attract-jax`).
- A different bucketed scheme also existed in the `minfor.py`-related codebase before 2026-02-28 (in `util/reproduce_grid_score.py` and `util/jax_scorer.py`); that scheme has been removed from the current production-oriented path.

### Production Path (Fused NB Kernel)

- The fused NB kernel design evaluates neighbors using real per-voxel counts (`k = nr_neigh[voxel]`) and loops over actual neighbors, not a fixed cap loop.
- The fused design consumes DOFs directly in C++ and performs coordinate transform, voxel lookup, and NB accumulation in one pass.
- Expected impact (from first1000 short-circuit profiling context): remove Python-side hit-list build and sorting from the NB critical path by eliminating intermediate target-list materialization.

Status:

- Integrated in `minfor.py` via `--nb-kernel fused` (with `--oracle jax`).
- Potential-gradient mode is now explicit in CLI:
  - default: stored gradients + custom JVP (pre-switch behavior)
  - optional: `--autodiff-potentials` (AD path, no stored grid gradients)
- End-to-end score and minimization benchmarks on `first1000` and `first10k` have been executed.
- Legacy-comparable criterion (`+/-10%`) is met on the measured `first10k` runs in this snapshot.
- Full-set repeatability check has been run for both modes (two autodiff runs, one stored-gradient run).

### Full-Set RMSD Detail (2026-03-01)

Top-1000-by-energy, best 10 LRMSD values and their energy ranks:

`--autodiff-potentials` run 1 and run 2 (identical):

1. LRMSD `1.948` at energy-rank `47`
2. LRMSD `2.857` at energy-rank `28`
3. LRMSD `3.118` at energy-rank `14`
4. LRMSD `3.155` at energy-rank `13`
5. LRMSD `6.824` at energy-rank `679`
6. LRMSD `7.147` at energy-rank `559`
7. LRMSD `17.912` at energy-rank `368`
8. LRMSD `17.943` at energy-rank `367`
9. LRMSD `18.559` at energy-rank `199`
10. LRMSD `18.565` at energy-rank `200`

Default stored-gradients run:

1. LRMSD `3.151` at energy-rank `23`
2. LRMSD `3.182` at energy-rank `7`
3. LRMSD `5.653` at energy-rank `49`
4. LRMSD `5.734` at energy-rank `405`
5. LRMSD `6.891` at energy-rank `771`
6. LRMSD `11.112` at energy-rank `225`
7. LRMSD `14.004` at energy-rank `507`
8. LRMSD `15.259` at energy-rank `82`
9. LRMSD `16.665` at energy-rank `295`
10. LRMSD `17.331` at energy-rank `538`

Artifacts:

- `test/minfor_jax_fused_full_systsearch_autodiff1.{log,time,dat,energy.npy,lrmsd,top1000_best10_lrmsd.json}`
- `test/minfor_jax_fused_full_systsearch_autodiff2.{log,time,dat,energy.npy,lrmsd,top1000_best10_lrmsd.json}`
- `test/minfor_jax_fused_full_systsearch_pregrad1.{log,time,dat,energy.npy,lrmsd,top1000_best10_lrmsd.json}`

## What Is Obsolete

The following statements are intentionally retired from this status snapshot:

- Cap40-based full-run timing as a production target metric.
- Framing `cap180` as a requirement for the fused native NB production path.
- Historical deep-dive narratives on bucketing/interior-culling inside this status file.
- Speculative GPU performance assessments without fresh measured benchmarks.

## Remaining Gaps To Legacy-Comparable minfor.py

Blocking items:

1. Complete formal correctness/regression suite beyond timing parity (pose-level and rank-level analyses on agreed benchmark sets).
2. Clean remaining obsolete helper scripts in `test/` that still reference removed `minfor-{eval,dump}*` utilities.

Non-blocking but useful:

- Keep JAX research path available for prototyping and AD validation.
- Keep benchmark artifacts machine-readable for repeatability.
- Keep `--nb-kernel jax` as fallback/research mode; use `--nb-kernel fused` for production.

## Validation Plan (Required Before Claiming Parity)

### 1) Correctness Gate (Pass/Fail)

- NB kernel energy/gradient agreement against current NB targets for step-replay datasets.
- Default tolerances (unless updated by explicit decision): energy `atol=1e-6`, `rtol=1e-12`; gradient `atol=1e-6`, `rtol=1e-12`. UPDATE: for very large energies, tolerances are irrelevant. When comparing a pose between program A and B, if both A and B assign energy >10000, then infinite tolerances are acceptable.
- Minimization outcome parity checks against legacy (`RMSD`, `fnat`, and energy ranking consistency on matched inputs).

Pass condition:

- No tolerance failures and no material minimization regressions on agreed benchmark sets.

### 2) Performance Gate (Pass/Fail)

- Report adjusted wall time and adjusted per-pose time for both production path and legacy baseline.
- Use matched datasets and protocol (same concat/subtraction method, same pose counts, same grid mode).
- Evaluate against the fixed comparability rule: within `+/-10%`.

Pass condition:

- Production `minfor.py` is within `+/-10%` of legacy on agreed representative workloads, or faster.

### 3) Reproducibility Gate (Pass/Fail)

- Record exact commands, environment name, thread settings, and key runtime flags.
- Record artifact paths for raw benchmark outputs (`json/csv/log` and output data files).
- Record benchmark date as an absolute date in every report summary.

Pass condition:

- A second run with the same setup reproduces conclusions within normal run-to-run variance.

### Document QA Checklist

1. Contradiction check: no unqualified claim that production NB depends on `cap180`.
2. Claim hygiene: every numeric performance statement includes dataset/context and measured/expected status.
3. Obsolescence check: no restart command blocks or exploratory deep-dive sections in this file.
4. Decision completeness: a reader can quickly answer where we are, what is done, what blocks parity, and what metric defines done.
5. Length/clarity guard: keep this file concise and snapshot-focused.

## Decision Log (2026-02-28)

Decisions adopted:

1. Snapshot-only status policy for `STATUS_JAX_MINIMIZATION.md`.
2. "Comparable to legacy" threshold fixed at within `10%`.
3. Production narrative is anchored to fused native NB kernel integration; JAX-only caveats are retained but contained.

Document contract changes:

1. This file is current-state only; historical detail belongs to Git history.
2. Historical investigations are no longer duplicated here.
3. Performance claims in this file must use explicit measured/expected labeling and the fixed comparability definition.

Assumptions/defaults in force:

1. Historical detail remains recoverable from Git and is intentionally omitted here.
2. The `10%` comparability threshold remains fixed until explicitly revised.
3. This snapshot is CPU-production oriented for current hardware context.
4. GPU commentary is excluded unless backed by fresh measured benchmarks.
