# NumPy pose scoring

`minfor.py --oracle numpy --score --energy-only` scores poses without importing
JAX or loading the compiled nonbonded kernel. NumPy is the only runtime
dependency for this path when running the scripts from this checkout. The
existing package metadata still declares JAX as a dependency.

New grids generated in-house default to a 10 Å plateau and 12 Å pairlist
distance, at 0.9 Å spacing. The legacy grid-generation example scripts also use
10–12. Precomputed grids retain the parameters stored in their headers.

Supply a precomputed ATTRACT binary `.grid` or an in-house `.npz` grid, or select
`--no-grid` to evaluate receptor-ligand pairs directly. With a grid, the score
includes the same inner/outer grid interpolation and short-range neighbour
correction as the JAX scorer, including electrostatic scaling and ligand pivots.
Floating-point reduction order can differ. Parity tests allow an absolute
score difference of 0.001 (including scores below 100), rather than requiring
bitwise equality. This mode does not generate grids, compute gradients, or minimize.

With `--no-grid`, no grid is loaded or generated. The calculation sums the full
8/6 LJ potential over all atom pairs and uses the existing shifted 50 Å cutoff
for rdie electrostatics. As in the legacy explicit scorer, squared distances
below 0.001 Å² are floored at that value. Pair buffers are bounded by processing
one pose and chunks of receptor and ligand atoms at a time. Grid-free scores
can differ from grid scores by more than 0.001: they remove interpolation and
include LJ contributions beyond the grid generator's 50 Å cutoff.

Use the same input options as JAX scoring, replacing the oracle selection:

```bash
python util/minfor.py poses.dat \
  --oracle numpy --score --energy-only \
  --receptor-ens-list receptor.list --ligand-pdb ligandr.pdb \
  --grid receptorgrid.grid \
  --attract-par-npz attract-original/attract-par.npz
```

For direct evaluation, replace `--grid receptorgrid.grid` with `--no-grid`.
The two options are mutually exclusive.

For an `(N, 6)` pose array, replace `poses.dat` with
`--input-npy poses.npy --input-format euler` (or `rotvec`). Array-based receptor
and ligand inputs, ligand conformers, `--pool-conformers`, pose slicing,
`--score-batch-size`, and `--output-npy energies.npy` also work. NumPy output uses
the existing float32 energy format; the Python API returns float64 energies.

```python
from jax_scorer import JaxScoreOracle  # util/ must be on sys.path

oracle = JaxScoreOracle(
    backend="numpy",
    energy_only=True,
    receptor_ens_list="receptor.list",
    ligand_pdb="ligandr.pdb",
    grid_file="receptorgrid.grid",
    attract_par_npz="attract-original/attract-par.npz",
    lig_pivot=pivot,  # (3,)
)
energies, gradients = oracle.score_batch(ensemble_ids, dofs)
# ensemble_ids: 1-based receptor indices; dofs: (N, 6), non-centered ATTRACT DOFs.
# Optional conformers=... supplies 0-based ligand conformer indices.
# gradients has shape (0, 6); score_single returns (energy, None).
```

For the grid-free Python API, replace `grid_file=...` with `grid_free=True` and
omit both `grid_file` and `grid_object`.

The existing paths remain `--oracle jax --nb-kernel jax` for pure JAX and
`--oracle jax --nb-kernel compiled` for JAX grid interpolation with compiled
neighbour corrections. Omit `--nb-kernel` when selecting NumPy.

Validation:

```bash
python -m pytest tests/test_energy_only.py tests/test_numpy_scoring.py -q
```

With JAX installed, the tests compare NumPy and JAX on ensembles, conformers,
both rotation conventions, grid boundaries, charged/uncharged ligands, and
zero-neighbour grids. A subprocess test blocks every JAX import and exercises
the CLI, including streamed NumPy output. The independent NumPy tests also run
in an environment where JAX is absent; parity tests skip there.

Grid-free tests also compare against an independent scalar all-pairs reference,
exercise both rotation conventions, mixed ensembles/conformers, chunking,
charge thresholds, dummy atoms, and exact overlaps. The grid-free CLI test
blocks imports of both JAX and the grid generator.

## Comparison on the four 1B7F poses

The previously computed grid scores used `1B7F-bound/receptorr.pdb` (863 atoms),
`1B7F-bound/frag5r.pdb` (18 atoms), and a legacy grid built with plateau distance
5 Å and neighbour distance 7 Å. All three grid paths (NumPy, CPU JAX, and the
compiled neighbour kernel) produced identical float32 output. Direct scoring
uses the same structures, ligand-centroid pivot, epsilon 15, and Euler DOFs:

```python
poses = np.array([
    [0., 0., 0., 0., 0., 0.],
    [.01, .02, -.01, .1, -.2, .1],
    [-.02, .01, .02, -.2, .1, -.1],
    [.1, -.1, .2, 1., -1., .5],
])
```

| Pose | Grid score | Direct NumPy | Direct − grid |
| --- | ---: | ---: | ---: |
| 1 | -27.980139 | -27.994417 | -0.014278 |
| 2 | -26.157488 | -26.170915 | -0.013427 |
| 3 | -26.706350 | -26.790114 | -0.083764 |
| 4 | -16.652702 | -16.687740 | -0.035038 |

These differences exceed the 0.001 tolerance used for NumPy/JAX agreement on
the same grid. The direct NumPy float64 scores independently agree with legacy
`attract-infinite --score` to within 1.2e-13 on all four poses. Full precision
values and both comparisons are saved in
[numpy-grid-free-comparison.csv](numpy-grid-free-comparison.csv).

### Larger plateau: 10–12 grid

Rebuilding the legacy grid with plateau 10 Å and pairlist distance 12 Å, with
the same alphabet and all other generation settings, gives the following
comparison. Both grids have spacing 0.9 Å and extension 32, verified from their
binary headers. NumPy, CPU JAX, and the compiled neighbour backend again
produce identical float32 scores on the 10–12 grid.

| Pose | Direct NumPy | 10–12 grid | Absolute error, 5–7 | Absolute error, 10–12 |
| --- | ---: | ---: | ---: | ---: |
| 1 | -27.994417 | -27.994753 | 0.014278 | 0.000336 |
| 2 | -26.170915 | -26.170822 | 0.013427 | 0.000093 |
| 3 | -26.790114 | -26.785679 | 0.083764 | 0.004435 |
| 4 | -16.687740 | -16.686569 | 0.035038 | 0.001171 |

The larger plateau substantially reduces the difference from explicit scoring:
the maximum absolute error falls from 0.083764 to 0.004435, and mean absolute
error from 0.036627 to 0.001509. Two of four poses meet the 0.001 threshold at
this grid spacing. A larger plateau moves more of the varying short-range
interaction into explicit pair corrections and leaves a smoother potential
for interpolation; it does not guarantee an error below 0.001. The CSV above
also includes all three 10–12 backend scores and their differences from direct
scoring.

## Larger sample: 2,048 candidate poses

An additional test used reproducible rigid-body perturbations around the same
bound 1B7F complex. Rotation-vector components were sampled independently from
a normal distribution with sigma 0.15 radians, and translation components
with sigma 1 Å. The following reproduces the input DOFs:

```python
rng = np.random.default_rng(20261027)
poses = np.concatenate([
    np.column_stack((rng.normal(0., .15, (1024, 3)),
                     rng.normal(0., 1., (1024, 3))))
    for _ in range(2)
])
poses[0] = 0.  # identity control
```

The ligand-centroid pivot, structures, force field, epsilon 15 and 0.9 Å grids
were unchanged. Of the 2,048 candidates, **1,221 had direct energy ≤100**;
the other 827 were excluded from all statistics below. Requiring energy ≤100
in every grid calculation retains exactly the same set. No generated pose
has an atom outside the inner grid. All scores in this test are float64.

| Absolute error against direct scoring | 5–7 | 10–12 |
| --- | ---: | ---: |
| Mean | 0.130512 | 0.005721 |
| Median | 0.104003 | 0.004673 |
| 95th percentile | 0.336362 | 0.014133 |
| Maximum | 0.534947 | 0.024835 |
| Poses within 0.001 | 10 / 1,221 (0.82%) | 129 / 1,221 (10.57%) |

The mean absolute error is about 23 times smaller with 10–12, but this grid
does not generally reproduce explicit scores within 0.001 across these
perturbations. This comparison measures the grid approximation: NumPy and CPU
JAX agree within 6.5e-12 on both grids, and direct NumPy agrees with legacy
`attract-infinite --score` within 8.2e-12 on all 1,221 retained poses.

[Per-pose scores and DOFs](numpy-grid-free-2048.csv) contain only retained
poses, with their original candidate indices. The
[summary and generation metadata](numpy-grid-free-2048-summary.json) include
additional percentiles, signed bias, RMSE, filter counts, and backend checks.
