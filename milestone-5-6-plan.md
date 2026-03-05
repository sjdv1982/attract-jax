# Milestone 5 + 6 Implementation Plan

## Design Principle

All physics evaluations go through `score_pairs()` dispatch.  No physics is
re-implemented in `grid_generator.py`.  The privileged initial path is
`backend="kernel"` (C++ nb_kernel), which dispatches to the already-present
`nb_kernel_euler_clamp_grad` / `nb_kernel_euler_clamp_energy` symbols.

---

## Step 1 — Extend `score_pairs_kernel` to return gradients

**File:** `util/score_pairs.py`

Currently `score_pairs_kernel` always calls the energy-only symbol and ignores
`out_grad`.  For potential-grid construction we need the stored-gradient mode
(`return_gradients=True`).

Changes:

- When `return_gradients=True`, select `nb_kernel_euler_{plateau_mode}_grad`
  and allocate `out_grad = np.zeros((N, 6))`.
- After the kernel call, extract `out_grad[:, 3:6]` (translation DOF gradients
  = Cartesian gradient of the probe atom) as the spatial gradient.
- Return `(energies, out_grad[:, 3:6])`.

The grad symbol is already in the `.so`:

```
Available symbols: [
  'nb_kernel_euler_clamp_energy', 'nb_kernel_euler_clamp_grad',
  'nb_kernel_euler_correction_energy', 'nb_kernel_euler_correction_grad',
  ...
]
```

**Potential problem — gradient sign:**
The C kernel computes `dE/d(tx, ty, tz)` for a probe whose single atom sits at
the origin and the pose translation is `(tx, ty, tz)`.  The legacy grid stores
`(gx, gy, gz)` as the *negative* spatial force convention, and
`potential_atom_energies_jvp` in `reproduce_grid_score.py` applies
`atom_grad = -vdw_eg[:, :, 1:4]` (grid channel 1–3 = negative gradient).

**Resolution:** Store `+dE/d(tx)` (positive kernel output) in the potential
grid channels 1–3 **with a negation** before storage so that it matches the
legacy `-(dE/dx)` convention consumed by `potential_atom_energies_jvp`.
This must be verified against the legacy grid's gradient values before the
validation run.

> **Note — legacy gradient bug (discovered 2026-03-05):** `grid_calculate.cpp`
> contains a math error in the plateau region of both `_calc_potential` and
> `_calc_potential_elec`. After clamping `dsq` and recomputing `rr2`, the
> scaled displacement `dd` is formed with the *post-clamp* `rr2` instead of
> the original `1/dsq_orig`. This means the gradient direction is scaled by
> `dsq_orig / plateaudissq` (which is 1 outside the plateau and < 1 inside).
> **Consequence:** gradient channels in the legacy `.grid` files are incorrect
> for voxels inside the plateau sphere. Agreement with legacy grid gradient
> channels is therefore **not** a validation target. The C kernel
> (`pose_loop.h`, `nb_kernel_euler_clamp_grad`) uses the correct formula.

---

## Step 2 — Build neighbour lists for potential-grid computation

**File:** `util/grid_generator.py` (new file)

Two separate KDTree queries — not via `score_pairs`:

### 2a. Potential-grid corner neighbours (50 Å, for calling `score_pairs`)

Two separate KDTree queries are needed — one for LJ, one for electrostatics:

**LJ neighbour list (all receptor atoms, 50 Å):**

- Voxel corners at `origin + (i, j, k) * gridspacing` (no half-voxel offset).
- KDTree over **all** receptor atoms, query radius = **50 Å** (the hard VDW
  cutoff used inside `_calc_potential` in `grid_calculate.cpp`).
- Results stored as `nr_neigh_lj (Nvox,)`, `nb_start_lj (Nvox,)`,
  `nb_concat_lj (total_nb,)`.

**Electrostatics neighbour list (charged receptor atoms only, 50 Å):**

- Same voxel corners.
- KDTree over **charged receptor atoms only** (atoms where `q_rec != 0`),
  query radius = **50 Å**.
- Results stored as `nr_neigh_elec (Nvox,)`, `nb_start_elec (Nvox,)`,
  `nb_concat_elec (total_nb_elec,)`.
- The atom indices in `nb_concat_elec` must index into the **full** receptor
  atom array (not into a compressed charged-only array), so that
  `rec_coords[nb_concat_elec[i]]` gives the correct Cartesian position.

Both sets of arrays are transient; only used during grid precomputation, not
stored in the final `Grid` object.

### 2b. NB-correction centre neighbours (neighbourdis, for scoring)

- Voxel centres at `origin + (i + 0.5, j + 0.5, k + 0.5) * gridspacing`.
- Query radius = `neighbourdis` (typically 12 Å).
- Results stored as `neighbour_grid (dx, dy, dz, max_K)` uint16, and
  `nr_neighbours (dx, dy, dz)` — these *are* part of the output `Grid`.

**Potential problem — flat index ordering:**
The C kernel looks up the neighbour list voxel by:

```
vox_idx = floor((tx - origin[i]) / spacing + 0.5)
flat_idx = vox_x + dim[0] * (vox_y + dim[1] * vox_z)
```

The Python KDTree is queried over a ravelled grid enumerated as
`(i, j, k)` in x-major order.  The `nr_neigh_corner` / `nb_start_corner`
arrays must be ravelled in the **same** order: `x + dx*(y + dy*z)`.
Use `np.mgrid` with explicit axis order and verify flat index matches.

---

## Step 3 — VDW potential-grid evaluation via `score_pairs`

For each alphabet channel `ch=0..n_alpha-1` (receptor atomtype
`alphabet_atomtypes[ch]`, ligand atomtype probe `alphabet_atomtypes[ch]`):

```python
energies, grads = score_pairs(
    query_coords   = corner_coords,         # (Nvox, 3)
    rec_coords     = rec_coords[ens],
    rec_atomtypes_idx = rec_atomtypes_ff,   # mapped to ff rec dimension
    rec_charges_scaled = np.zeros(M),       # suppress electrostatics
    query_atomtype_idx = ch,                # probe is this lig atomtype
    query_charge   = 0.0,                   # suppress electrostatics
    ff_params      = ff_params,
    ff_module      = nonbon8_module,
    nr_neigh       = nr_neigh_corner,
    nb_start       = nb_start_corner,
    nb_concat      = nb_concat_corner,
    origin         = origin,
    gridspacing    = gridspacing,
    dim            = dim,
    plateaudis_sq  = plateaudis**2,
    plateau_mode   = "clamp",
    backend        = "kernel",
    return_gradients = True,
)
inner_potential_grid[ch] = np.stack([-energies, -grads[:,0], -grads[:,1], -grads[:,2]], axis=-1)
    .reshape(dx, dy, dz, 4)
```

Notes:

- `rec_charges_scaled = zeros` and `query_charge = 0.0` → electrostatics term
  is zero in both JAX and kernel paths; no code change needed.
- The ff_params `rc/ac` arrays must be **full** `(nrec_types, nlig_types)` size
  (as in jax_scorer.py) so that the slice `[rec_type, query_atomtype_idx]`
  always resolves correctly for every receptor atom type.

---

## Step 4 — Electrostatic potential-grid evaluation via `score_pairs`

**Problem:** LJ and electrostatics are evaluated jointly in `score_pairs`.  To
isolate the electrostatic channel:

- Pass `ff_params_zero_lj` with `rc=0, ac=0, emin=0, rmin2=1.0, ivor=0` for
  all pairs.  With `rc=ac=0` → `vlj=0`, so `e_lj=0` regardless of the
  ivor/rmin2 branch.
- `rec_charges_scaled = q_rec_raw * ffelec**2` (so that at scoring time
  `q_lig_raw * grid_elec_val = q_lig_raw * q_rec * ffelec^2 * dd`).
- `query_charge = 1.0`.
- `query_atomtype_idx = 0` (arbitrary; LJ is zero).

```python
energies_elec, grads_elec = score_pairs(
    query_coords      = corner_coords,
    rec_coords        = rec_coords[ens],
    rec_atomtypes_idx = np.zeros(M, dtype=np.int32),  # irrelevant
    rec_charges_scaled = q_rec_raw * ffelec**2,
    query_atomtype_idx = 0,
    query_charge       = 1.0,
    ff_params          = ff_params_zero_lj,
    ...
    plateau_mode = "clamp",
    backend      = "kernel",
    return_gradients = True,
)
inner_elec_grid = np.stack([-energies_elec, -grads_elec[:,0], -grads_elec[:,1], -grads_elec[:,2]], axis=-1)
    .reshape(dx, dy, dz, 4)
```

**Potential problem — elec plateau in clamp mode:**
In the C kernel the plateau scaling for electrostatics in clamp mode applies
`ratio = sqrt(dsq/plateaudissq)` and then computes `dd = (rr2 * ratio^2 -
inv50sq)`.  In the legacy `_calc_potential_elec`, the clamping is:

```cpp
if (dsq < plateaudissq) { ratio = sqrt(dsq/plateaudissq); dsq = plateaudissq; rr2 = 1/plateaudissq; }
dd = { d*rr2*ratio, ... }
charge0 = charge * ffelec^2
elec(cdie=false, charge0, rr2, dd_x*ratio, ...)   // rdie: E = charge * rr2
```

The Python `score_pairs_jax` electrostatics clamp is `dsq_eff = max(dsq, pdsq)` then
`dd = rr2_eff - inv50sq`.  The two formulations are **not identical** for
`dsq < plateaudissq` — the legacy C uses `rr2 = 1/plateaudissq` but `dd` uses the
original direction, while the Python uses `1/pdsq` directly.  The kernel C
(`elec.h`) must be checked against `grid_calculate.cpp` to confirm they match.
**Action:** Read `elec.h` and `elec_grad.h` before coding and put an assertion
in the validation test.

---

## Step 5 — Outer grid

Outer grid voxel `(i, j, k)` has corner position:

```python
outer_corner = origin + (2*i - gridextension, 2*j - gridextension, 2*k - gridextension) * gridspacing
# equivalently (matching test-calc-grid-energy.py):
outer_corner = (np.array([i,j,k]) - gridextension/2.0) * 2*gridspacing + origin
```

Outer dimensions:

```python
dx2 = int((dx + 2*gridextension) / 2) + 1
dy2 = int((dy + 2*gridextension) / 2) + 1
dz2 = int((dz + 2*gridextension) / 2) + 1
```

Same 50 Å KDTree + same VDW and elec `score_pairs` calls as the inner grid,
with `outer_corner_coords` replacing `corner_coords`.  The neighbour list for
the outer grid is separate from the inner grid's.

**Note:** The outer grid has no neighbour-correction grid; `neighbour_grid` in
the `Grid` namedtuple is inner only.

---

## Step 6 — Assemble `Grid` namedtuple

The output must be compatible with what `read_grid_with_electro` returns so
that `JaxScoreOracle` and `build_kernel` work unchanged.  Fields:

| Field | Shape | dtype | Source |
|---|---|---|---|
| `inner_potential_grid` | `(n_alpha, dx, dy, dz, 4)` | float32 | Step 3 |
| `outer_potential_grid` | `(n_alpha, dx2, dy2, dz2, 4)` | float32 | Step 5 |
| `inner_elec_grid` | `(dx, dy, dz, 4)` | float32 | Step 4 |
| `outer_elec_grid` | `(dx2, dy2, dz2, 4)` | float32 | Step 5 |
| `neighbour_grid` | `(dx, dy, dz, max_K)` | uint16 | Step 2b |
| `nr_neighbours` | `(dx, dy, dz)` | int16 | Step 2b |
| `max_nr_neighbours` | scalar | int | Step 2b |
| `alphabet_atomtypes` | `(n_alpha,)` | int32 | input |
| `plateaudis` | scalar | float | input |
| `gridspacing` | scalar | float | input |
| `dim` | `(3,)` | int32 | computed |
| `dim2` | `(3,)` | int32 | computed |
| `origin` | `(3,)` | float64 | computed from rec coords |
| `gridextension` | scalar | int | constant 32 |
| `neighbourdis` | scalar | float | input |
| `alphabet` | `(99,)` | bool | derived from alphabet_atomtypes |
| `natoms` | scalar | int | len(rec_coords) |
| `neighbour_grid_ravel` | None | — | set to None (filled by JaxScoreOracle) |

---

## Step 7 — Extend `JaxScoreOracle` to accept a `Grid` object

**File:** `util/jax_scorer.py`

- Add `grid_object=None` parameter to `__init__`.
- If `grid_object` is not None, skip `read_grid_with_electro(Path(grid_file).read_bytes())`.
- If both `grid_file` and `grid_object` are None, raise `ValueError`.
- All downstream code is unchanged.

---

## Step 8 — Add `--generate-grid` to `minfor.py`

**File:** `util/minfor.py`

New CLI flag: `--generate-grid` (mutually exclusive with `--grid`).  Requires:

- `--receptor-ens-list` (already exists)
- `--attract-par-npz` (already exists)
- `--ligand-pdb` or equivalent for alphabet determination

When used the oracle is constructed as:

```python
from grid_generator import generate_grid
grid_obj = generate_grid(
    rec_pdb_list=ens_list_path,
    attract_par_npz=par_npz,
    epsilon=args.epsilon,
    backend=args.nb_kernel,
)
oracle = JaxScoreOracle(..., grid_object=grid_obj, grid_file=None)
```

---

## Step 9 — M6 unit test for cross-backend consistency

**File:** `native/nb_kernel/tests/test_score_pairs_m6.py` (new)

Tests:

1. `score_pairs(..., backend="jax")` vs `score_pairs(..., backend="kernel")` on
   synthetic small receptor — energies must agree within 1e-5.
2. `generate_grid(..., backend="kernel")` vs `generate_grid(..., backend="jax")`
   on the test receptor — potential grid values must agree within 1e-4 (float32
   storage).
3. Full scoring via `JaxScoreOracle` with in-house grid must match reference
   `.score` files within the harness tolerance (1e-6 atol/rtol on energies).

---

## Step 10 — Validation runs

### M5 validation (kernel backend, in-house grid)

Run both concat scoring cases with `--generate-grid` and compare against the
existing reference `.score` files using `compare_scores.py`:

- `test/first1000/score_jax_fused_first1000_concat_pregridexcise_style.score`
- `test/first10k/score_jax_fused_first10k_concat_pregridexcise_style.score`

Accept atol=1e-4 / rtol=1e-4 (float32 grid storage introduces small rounding
vs the legacy float32 grid; legacy comparison itself already uses 1e-6).

### M6 validation (JAX backend, in-house grid)

Same as M5 but with `--nb-kernel jax`.  Grids must match within 1e-4; scores
must pass `compare_scores.py` at the same tolerance.

---

## Milestone 6b — Gradient convention (completed)

### Problem

The legacy `.grid` binary stores **forces** `−(dE/d_i)` in channels 1–3.
`score_pairs` returns analytic **gradients** `+(dE/d_i)`. After Steps 3–4 the
generated NPZ was storing the negated (force-convention) values to match, but
this added accidental complexity to all consumers.

### Resolution

Adopt **gradient convention `+(dE/d_i)` everywhere** in memory:

1. `util/grid_generator.py` `_eval_voxels` — store `+grads` directly.
2. `util/reproduce_grid_score.py` `read_grid_with_electro` — negate channels
   1:4 after reading legacy binary (force → gradient conversion at the boundary).
3. `util/reproduce_grid_score.py` `potential_atom_energies_jvp` — remove the
   negation of VDW channels and the subtraction of `g_el` (was `−g_el`, now
   `+g_el`).

### Validation

- **Energy consistency at fixed poses:** max |ΔE| < 5 × 10⁻⁸ kcal/mol across
  all three scoring modes (legacy stored-grad, NPZ stored-grad, NPZ autodiff).
- **Gradient consistency (stored-grad modes):** max |Δg| < 1 × 10⁻⁷ — legacy
  and NPZ computed gradients are bit-identical to float32 precision.
- **Autodiff gradient discrepancy (~25%):** expected — derivative of a piecewise-
  linear interpolant ≠ interpolant of the analytic derivative. Stored-gradient
  path is the recommended production mode.
- **Minimization:** three successful runs (5 poses, maxfun=500); trajectories
  diverge to different local minima due to gradient differences, which is expected.

1. **Gradient sign convention:** Do not block on this. The placeholder negations
   (`-grads[:,0]` etc.) in Steps 3–4 must be verified numerically by comparing
   a single voxel from the in-house generator against the legacy
   `test/receptorgrid.grid` before running full validation. If the sign is wrong,
   flip it — it is a one-line fix and cannot hide behind other errors.

2. **Receptor ensemble:** Always generate from **ensemble member 0 only**,
   matching the legacy `make-grid` behaviour (single receptor PDB input).
   Ensemble variation at scoring time is handled by the NB correction path
   (different `ens` index selects different receptor coords), not by the grid.

3. **Electrostatics plateau — mandatory pre-coding check:**
   Before writing any electrostatics grid generation code, read `elec.h`
   (kernel clamp formula) and `grid_calculate.cpp` (`_calc_potential_elec`
   clamp formula) side by side.

   - **If they match:** use `backend="kernel"` for electrostatics precomputation
     (same as LJ). No split needed.
   - **If they differ:** use `backend="jax"` for electrostatics precomputation
     only, implementing `grid_calculate.cpp`'s formula directly. Keep
     `backend="kernel"` for LJ. This split is localised entirely inside
     `grid_generator.py` — `score_pairs()` itself does not change.

   Do not assume they match and proceed. Do the check first.
