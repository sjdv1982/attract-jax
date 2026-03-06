# Plan: Port `crocodile/WIP/score.sh` to `minfor.py` with ligand-ensemble support

## 1) Target outcome

> Status report (2026-03-06): Implemented in code. `crocodile/WIP/score.sh` now routes through `attract-jax/util/minfor.py --score --energy-only`, preserves `poses-1.ene` plus `poses-1.ene.npy`, and uses `score-attract-jax.py` as the output-behavior reference. The percentile-based WIP equivalence check is not yet automated end-to-end.

- Replace `protocols/score-attract-jax.py` in `crocodile/WIP/score.sh` with `attract-jax/util/minfor.py --score`.
- Keep first-step pose conversion from `poses-1.npy + offsets-1.dat` to ATTRACT `.dat` as scoring input.
- Preserve WIP output artifacts:
  - `poses-1.ene`
  - `poses-1.ene.npy`
- Maintain energy-equivalence criteria in `energy-equivalence-test.txt`:
  - `x = log(clip(e + 18, min=0.01, max=inf))`
  - compare selected percentiles (p1..p99 excluding the last as specified).

## 2) Required ligand-input behavior

> Status report (2026-03-06): Implemented. `minfor.py` now supports legacy `--ligand-pdb`, coordinate-array `--ligand-ensemble` with required `--ligand-atomtypes`, optional zero-default `--ligand-charges`, and external `--ligand-conformers`. `--ligand-pdb-list` was also added with a same-metadata restriction across listed PDBs.

- `minfor.py` should support:
  - single reduced-ligand `.pdb` (ATTRACT-style reduced ligand PDB with atom type and charge column, existing path),
  - ligand coordinate ensemble `.npy`:
    - single pose `(N, 3)`,
    - conformer library `(C, N, 3)`.
- For coordinate `.npy` modes, atom metadata must be supplied explicitly:
  - `--ligand-atomtypes` required (shape `(N,)`)
  - `--ligand-charges` optional (shape `(N,)`, default 0 if absent).
- Optional future-compatible ligand list mode:
  - `--ligand-pdb-list` (list of PDB files) may be added after parity.
- Per-pose conformer indices come from `poses-1-conformers.npy` (currently uint16, 1-based in WIP files).

## 2.1) Required receptor-input behavior (mirroring section 2)

> Status report (2026-03-06): Implemented. Existing `--receptor-ens-list` behavior is preserved, and `minfor.py` now also accepts `--receptor-pdb`, wrapping a single receptor into a one-line ensemble list for the JAX path.

- `minfor.py` currently supports receptor ensembles via `.dat` first-field conformer index in combination with `--receptor-ens-list`.
- Receptor source modes to preserve:
  - receptor ensemble list (existing path): one `.pdb` per line in `--receptor-ens-list`,
  - single receptor `.pdb` when no ensemble is provided (legacy minimizer/scoring compatibility behavior).
- For receptor ensemble mode:
  - `.dat` first structure field maps to receptor conformer index used by existing parser/oracle behavior.
  - receptor PDBs can already be mixed in type/size as supported by current `minfor.py` flow.
- For ligand-porting this matters because `.dat` conformer indexing must stay receptor-only; ligand conformers stay external in WIP.

### 2.2) Conformer-index semantics (important correction)

> Status report (2026-03-06): Implemented. The `.dat` receptor field remains receptor-only, ligand conformers remain external via `--ligand-conformers`, and the ligand line’s optional leading field is parsed but not used as the scoring conformer selector.

- `.dat` has a first per-structure field currently used for receptor conformer index.
- In current `minfor.py`, that field is receptor-specific and effectively supports receptor ensembles.
- In current `score.sh`, ligand conformers are passed separately (`--ligand-conformers`).
- For this port, ligand conformer selection must remain a separate argument stream and must not be read from `.dat` ligand field.

## 3) Mandatory regression checkpoints after section 2

> Status report (2026-03-06): Partially complete. Static checks pass and a 4-pose end-to-end ligand-ensemble parity run matches the legacy ligand-PDB path exactly, but the full `first1000` and `first10k` concat checkpoints below are still pending.

- Before full-scale equivalence checks, run these mandatory regression checks first:
  - `first1000` concat scoring test
  - `first10k` concat scoring test
- For both subsets:
  - compare to baseline using the existing `energy-equivalence-test.txt` percentile transformation;
  - require pass before moving to broader validation.

### 3.1 `first1000` regression protocol (from `prompt.md` + `test/first1000/*`)

> Status report (2026-03-06): Pending full concat execution. I have not yet run the full active-only `first1000` benchmark/equivalence pass from this subsection.

- Run `test/first1000/run_task5_dump_traj_legacy.sh` from `test/first1000`.
- This script:
  - runs `minfor-dump-traj.py` with `--oracle legacy --traj` on `systsearch-ens1-first1000.dat`
  - writes `first1000_dumptraj.traj.<step>.dat` and `first1000_dumptraj.dump.<step>.idx.txt`
  - runs `minfor.py --score --oracle legacy` per step on each `.dat`
  - asserts per-step `Energy:` equality with `## Energy:` in the corresponding `.dat`.
- For active-only concat, build the scoring input from only active indices in each `*.dump.<step>.idx.txt`, i.e. from non-converged poses for each step.
- Use the prepared inputs:
  - `test/first1000/tmp_active_concat.dat` (`106003` poses)
  - `test/first1000/tmp_dumptraj_4poses.dat` (4 poses subtraction timing file)
- Execute `bash test/first1000/test_first1000_concat_score.sh <out_dir>` (example: `bash test/first1000/test_first1000_concat_score.sh test/first1000/Z`).
- Scoring command emitted by the script is `minfor.py test/first1000/tmp_active_concat.dat --score --oracle jax --nb-kernel nonbon8 --attract-par-npz attract-jax/attract-par.npz --receptor-ens-list test/partner1-ensemble.list --ligand-pdb test/ligandr.pdb` with pregrad and autodiff variants.
- Optional seamless-equivalent path is available via `test/first1000/test_first1000_concat_score-SEAMLESS.sh`.
- Validate measured timings against historical files:
  - `test/first1000/tmp_legacy_concat_benchmark.json`
  - `test/first1000/tmp_legacy_concat_benchmark_4core_wrapper.json`
  - `test/tmp_fair_active_concat_benchmarks.json` (if reproducing fair active-only concat benchmark tables).

### 3.2 `first10k` regression protocol (from `nb-kernel-plan.md` + `test/first10k/*`)

> Status report (2026-03-06): Pending. The full `first10k` concat regression from this subsection has not yet been run.

- Use input (first10k example):
  - `test/first10k/tmp_active_concat.dat` (`1051136` poses)
- Execute `bash test/first10k/test_first10k_concat_score.sh <out_dir>` (first10k example).
- Script scoring command is `minfor.py test/first10k/tmp_active_concat.dat --score --oracle jax --nb-kernel nonbon8 --attract-par-npz attract-jax/attract-par.npz --receptor-ens-list test/partner1-ensemble.list --ligand-pdb test/ligandr.pdb` with pregrad and autodiff variants.

## 4) `score.sh` control flow (WIP)

> Status report (2026-03-06): Implemented. The WIP shell script now performs decode-to-matrix, deterministic header generation, `mat4_to_dat.py` conversion, `minfor.py --score --energy-only` scoring, `Energy:` extraction, and artifact writing/cleanup.

- Step A: decode rotamer representation
  - `python crocodile/code/decode_rotamer_matrices.py --poses poses-1.npy --offsets offsets-1.dat --sequence UG --output <tmp_matrix.npy>`
- Step B: convert to ATTRACT `.dat`
  - write deterministic header with:
    - `#pivot auto`
    - `#centered receptor: false`
    - `#centered ligands: false`
  - `python attract-jax/util/mat4_to_dat.py <tmp_matrix.npy> --template-dat <tmp_header.dat> --output <tmp_pose.dat>`
- Step C: score with `minfor.py` using ligand ensemble
  - `python attract-jax/util/minfor.py <tmp_pose.dat> --score --energy-only --oracle jax --grid 1b7f_dom2-aar.grid --attract-par-npz <attract-par.npz> --receptor-ens-list <tmp_receptor.list> --ligand-ensemble fraglib-UG-ex1b7f.npy --ligand-conformers poses-1-conformers.npy --ligand-atomtypes UG-atomtypes.npy --ligand-charges /tmp/ligand-charges.npy > <tmp_score>`
  - fallback: if no ligand-ensemble args are provided, keep existing single-ligand reduced-PDB path.
- Step D: output
  - parse `Energy:` lines and write:
    - `poses-1.ene` (`%.3f`)
    - `poses-1.ene.npy` (float64)
- Step E: cleanup temp files.

## 5) `minfor.py` CLI and input-resolution plan

> Status report (2026-03-06): Implemented. Input resolution is now explicit and validated before oracle construction.

### 5.1) Add/adjust CLI arguments

> Status report (2026-03-06): Implemented. The new ligand and receptor arguments are present, mutual exclusions are enforced, legacy-oracle incompatibilities fail early, and conformer/vector shape mismatches are reported before scoring starts.

- Add:
  - `--ligand-ensemble`
  - `--ligand-conformers`
  - `--ligand-atomtypes`
  - `--ligand-charges`
  - `--ligand-pdb-list` (optional future parity mode)
- Validation rules:
  - if `--ligand-ensemble` is set:
    - require `--ligand-atomtypes`
    - if `--ligand-conformers` is set, its length must match number of scored poses
    - shape checks for `(N,3)` and `(C,N,3)` are enforced
  - if neither `--ligand-ensemble` nor `--ligand-pdb-list` is set, require `--ligand-pdb` (legacy mode)
  - `--ligand-pdb` and `--ligand-ensemble` are mutually exclusive unless explicitly implementing a fallback mode.

### 5.2) Source-mode precedence

> Status report (2026-03-06): Implemented. Resolution order is `--ligand-ensemble`, then `--ligand-pdb-list`, then legacy `--ligand-pdb`.

- `--ligand-ensemble` first (coordinate-array mode).
- else `--ligand-pdb-list` (optional).
- else existing `--ligand-pdb` (legacy).
- In all modes:
  - create internal coordinate arrays in `(C,N,3)` shape for kernel paths.
  - apply dummy-atom masking (`type != 99`) exactly once in preprocessing.
  - preserve atom ordering expected by provided `--ligand-atomtypes`.

### 5.3) `.dat` reader amendment (required)

> Status report (2026-03-06): Implemented. `parse_dat_two_body` now enforces exactly two numeric lines per pose, validates 6-vs-7 field receptor/ligand semantics, and raises informative errors for malformed receptor DOFs or missing ligand DOFs.

- `minfor.py` currently assumes a compact two-line-per-pose DOF layout; it must be made explicit and strict.
- For each structure in `.dat`, parse two lines:
  - line 1: receptor state
  - line 2: ligand state
- For each of these two lines, enforce field count is either 6 (no ensemble index) or 7 (ensemble index + 6 DOF).
- Receptor line rules:
  - if 6 fields, all 6 receptor DOF fields must be zero.
  - if 7 fields, first field is receptor conformer ID and the final 6 fields must all be zero.
- Ligand line rules:
  - if 6 fields, ligand DOFs are in these 6 fields.
  - if 7 fields, first field is ligand ensemble marker (unused in WIP; should not be treated as scoring conformer), final 6 are ligand DOFs.
- Add explicit validation and informative errors when:
  - a ligand line is missing pose DOF fields,
  - receptor line has non-zero DOFs in non-ensemble mode,
  - receptor line has non-zero DOFs when provided as 7-field ensemble record.
- Preserve backward compatibility with current legacy `--score` formatting while guaranteeing receptor/ligand field semantics for this port.

## 6) `jax_scorer.JaxScoreOracle` extension plan

> Status report (2026-03-06): Implemented. The oracle accepts coordinate ensembles plus sidecar metadata and can score per-pose ligand conformers without breaking the legacy single-ligand path.

### 6.1) Constructor/input storage

> Status report (2026-03-06): Implemented. The constructor now accepts `ligand_ensemble`, `ligand_atomtypes`, and `ligand_charges`, normalizes `(N,3)` to `(1,N,3)`, applies the dummy-atom mask once, and stores conformer-indexable ligand coordinate tensors.

- Extend constructor to accept either:
  - legacy single ligand via `ligand_pdb`, or
  - coordinate-ensemble mode via `ligand_ensemble + ligand_atomtypes (+ ligand_charges)` and per-pose `conformers`.
- Store conformer-aware ligand tensors:
  - `_coor_lig_ens_j` as `(C, N, 3)`
  - `_lig_atomtypes_ff_j` from sidecar atom types
  - `_lig_charges*` from sidecar charges or zeros.

### 6.2) API changes

> Status report (2026-03-06): Implemented. `score_batch` now accepts `conformers=None`, and `score_single` accepts an optional `conformer`; the old no-conformer behavior remains available.

- Add optional conformer array to score interface:
  - `score_batch(ens, dofs, conformers=None)`
  - `score_single(ens_id, dof, conformer=0)` (or equivalent wrapper)
- Backward compatibility:
  - `conformers=None` means all conformer index `0`.
- Thread conformer path through:
  - `_energy_batch_raw`
  - merged-ensemble paths
  - `_pot_vg_batch` / `_pot_vg_single`
  - nonbon8/C backend equivalents where available.

### 6.3) Performance-preserving batching

> Status report (2026-03-06): Implemented. When conformers are supplied, scoring is grouped by `(receptor_ens_id, ligand_conformer_id)` and processed in the existing batch size windows; when conformers are absent, the old merged-ensemble path is still used.

- Group poses by `(receptor_ens_id, ligand_conformer_id)` and evaluate grouped slices in batches.
- Use the same energy batching controls (`energy_batch`, kernel padding).
- Preserve old non-ensemble code path exactly when no conformer array is supplied.

## 7) Behavioral parity validation

> Status report (2026-03-06): Partial. A duplicated-coordinate ligand ensemble with conformer indices reproduces the single-ligand PDB energies exactly on `test/first1000/tmp_dumptraj_4poses.dat`. The percentile-based WIP dataset comparison from `energy-equivalence-test.txt` still needs to be run on the intended larger datasets.

- Run baseline-vs-newenergies comparison with percentile checks from `energy-equivalence-test.txt`:
  - identical length and finite values;
  - `x = log(clip(e + 18, 0.01, inf))`;
  - report per-percentile deltas and MAE/MaxAE.
- If mismatch persists:
  - check `poses-1-conformers.npy` indexing conversion (1-based vs 0-based),
  - check centered/non-centered convention assumptions,
  - confirm receptor-list and `.dat` pose ordering alignment.

## 8) Open decisions before execution

> Status report (2026-03-06): Resolved/updated. `--ligand-charges` is optional with a zero default in coordinate-array mode. `--ligand-pdb-list` was implemented with uniform atom-type/charge requirements across conformers. The percentile tolerance calibration remains open until the larger validation runs are executed.

- Keep `--ligand-charges` required or optional (with zero default) for coordinate arrays.
- Confirm `--ligand-pdb-list` scope in v1 (optional / deferred).
- Set explicit percentile tolerance window after first calibration run.
