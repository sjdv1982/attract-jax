# JAX Minimization — Status Document

**Date**: 2026-02-17 (updated)
**Goal**: Minimize all 165,528 poses from `test/systsearch-ens1.dat` using the JAX oracle + VA13 minimizer.

---

## What's Done

### 1. JAX Scorer (`attract-jax/util/jax_scorer.py`)

- **Analytical gradients** via `jax.vmap(jax.value_and_grad(single_energy))` through `main_ad` (fully JIT-compilable kernel).
- **Pre-converted receptor arrays** to JAX (eliminates per-call numpy→jax conversion).
- **`max_nb_cap=40`**: caps neighbour offsets from 180→40, gives identical energies for all tested poses, ~4× faster.
- **`energy_batch=8`**: best throughput per kernel call (~11ms/call for 8 poses with `block_until_ready`).
- **`block_until_ready()`** added to timing code for accurate kernel measurements.

### 2. Minimizer (`attract-jax/util/minfor.py`)

- `minfor_minimize_batched()`: proven VA13 (label 110/135 fix verified against Fortran).
- **`--pose-offset N`** argument added for parallel chunking.
- **Per-tick timing** in trace output: `[kernel=Xs python=Ys]`.
- Output: `.dofs.npy`, `.energy.npy`, `.ens.npy`, `.nfev.npy`, `.mat4.npy`, `.dat`.

### 3. Kernel (`attract-jax/util/reproduce_grid_score.py`)

- `nb_energy_vectorized`: processes all (atom, offset) pairs simultaneously.
- `main_ad`: fully JIT-compilable (no Python control flow), used for AD gradients.
- `padded_nb_size` + `max_nb_cap` parameters on `build_kernel()`.

### 4. Parallel Launcher (`attract-jax/util/launch_parallel.sh`)

- Splits 165,528 poses into N chunks using `--pose-offset` and `--max-poses`.
- Launches with single-threaded XLA (`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`).
- Merges results after all chunks complete.
- **Currently hardcoded to 6 processes — needs adjustment** (see scaling below).

### 5. Verified Correctness

- JAX vs legacy oracle energy at same DOFs: max diff = 0.050, mean = 0.013.
- AD gradients differ from FD (and legacy) due to `custom_jvp` on `potential_atom_energies` — this is expected and both are valid for minimization.
- 1000-pose run: best energy -22.216, matching earlier runs.

---

## Key Performance Numbers

### Profiling (1000 poses, eb=8, cap=40, all XLA threads)

- **Total: 175.5s** → kernel 169.9s (96.8%), Python state updates 1.6s (0.9%)
- Mean nfev per pose: 103
- ~130 kernel calls per tick, ~11ms per call

### XLA Threading: No Benefit

- Single-threaded: 1.484s per score_batch (1000 poses)
- Multi-threaded (default): 1.48s per score_batch (1000 poses)
- Conclusion: per-call work is too small for thread parallelism

### Parallel Process Scaling (500 poses per process, small-chunk test)

| Processes | Throughput  | Notes |
|-----------|-------------|-------|
| 1         | 4.9 poses/s | baseline |
| 2         | 6.4 poses/s | +30% — best scaling observed |
| 3         | 5.9 poses/s | memory bandwidth limited |
| 4         | 4.9 poses/s | no benefit |
| 6         | 5.2 poses/s | no benefit |

**CAVEAT**: These were measured with only 500 poses per process. JIT warmup (~10s) is a big fraction of total time at that size. With 27k+ poses per chunk, the init cost is negligible, so **scaling might be better with real chunk sizes**. This was never tested.

### Projected Wall Clock (165,528 poses)

- 1 process: ~9.4 hours (conservative, based on 4.9 poses/s)
- 2 processes: ~7.2 hours (based on 6.4 poses/s — may be pessimistic)
- True throughput with large chunks: **UNKNOWN, needs testing**

---

## Full Run Results (2026-02-17)

The full 165,528-pose minimization **completed** in **~8 hours** (single process, old oracle code with energy-batch=8, cap40).

### Performance vs Legacy ATTRACT

| Config                  | JAX (cap40) | Legacy single-thread | Ratio           |
|-------------------------|-------------|----------------------|-----------------|
| Wall clock (165k poses) | ~8 hours    | ~1 hour              | **~10× slower** |

### Correctness Issue: cap40 vs cap180

The full run used `max_nb_cap=40`, which was validated to give identical energies for a small set of tested poses. However, **only cap180 (i.e. no capping) gives correct energies in general**. Since the nb kernel dominates >80% of wall time, and cap180 has ~3.4× more nb kernel time than cap40 (measured), the **true slowdown with correct settings is ~30-35×** vs legacy ATTRACT.

### What Remains

1. **Result validation** (RMSD, fnat) against legacy ATTRACT output.
2. **Re-run with cap180** for correct energies (estimated ~32 hours with current code — impractical without kernel improvements).
3. **Nb grid kernel optimization** — see below.

---

## Oracle Optimization — Merged-Ensemble Batching (2026-02-15)

**Goal**: Reduce the ~10× JAX-vs-legacy speed gap to 2-3× without degrading GPU performance.

### Problem Analysis

Profiling the old oracle (energy-batch=8, per-ensemble loop) on 30 poses revealed:

- **11 kernel calls per tick** (one per ensemble), each processing 8 poses
- Per-ensemble batches had only ~3 real poses + 5 padded → **62% compute wasted on padding**
- With 11 ensembles × 6 sub-batches = 66 dispatch calls per tick for 500 active poses
- Python dispatch overhead: ~0.5ms × 66 = 33ms per tick
- Multi-threading (`--xla_cpu_multi_thread_eigen=true`): **zero benefit** — individual operations too small for thread parallelism
- Float32: **zero benefit** — workload is memory-bound (grid lookups), not compute-bound

### Solution: Merged-Ensemble Oracle

Changed `_vg_batch` from `vmap(in_axes=(0, None, None))` to `vmap(in_axes=(0, 0, 0))`:

- Each pose carries **its own** receptor coordinates/charges (indexed from stacked ensemble arrays)
- All ensembles processed in a **single kernel call** — no per-ensemble Python loop
- Adaptive batch sizing via `_round_up_batch()` — pads to discrete sizes (8, 16, 32, ..., 1024) for JIT cache reuse

### Files Changed

- **`attract-jax/util/jax_scorer.py`**:
  - `_vg_batch`: `in_axes=(0, None, None)` → `in_axes=(0, 0, 0)` (per-pose rec data)
  - `score_batch()`: eliminated ensemble loop; gathers per-pose rec data from stacked numpy arrays; sub-batches by `energy_batch`
  - `_round_up_batch()`: pads to discrete sizes for JIT shape stability
  - XLA multi-threading override (neutral on this hardware, may help on servers)
  - `_vg_ensemble()`: updated to broadcast shared rec data for the per-pose vmap signature

- **`attract-jax/util/minfor.py`**:
  - `--energy-batch` default: 8 → 256 (controls sub-batch size for very large pose counts)

### Benchmark Results

All benchmarks single-threaded, PID 134725 paused, i5-1235U.

**score_batch micro-benchmark** (30 ticks, post-warmup):

| Config | 30 poses | 200 poses | 500 poses |
|--------|----------|-----------|-----------|
| OLD (batch=8, per-ens) | 208 ms, 144 p/s | — | — |
| NEW (merged, EB=256) | 41 ms, 739 p/s | 325 ms, 615 p/s | 699 ms, 715 p/s |
| Speedup | **5.1×** | — | — |

**Mini-minimization** (maxfun=50):

| Poses | energy_batch | Time | Rate | vs baseline |
|-------|-------------|------|------|-------------|
| 200 | 256 | 17.2s | **11.6 p/s** | **2.4×** |
| 1000 | 256 | 74.0s | **13.5 p/s** | **2.8×** |
| 1000 | 1024 | 97.6s | 10.2 p/s | 2.1× |

**Optimal energy_batch = 256**: Fits L3 cache (256 × 80 atoms × 40 nb × ~12 B ≈ 10 MB). Larger batches (1024) degrade due to cache pressure.

**Approaches tested but rejected**:

- Float32 computation: +3-5% (grid data already float32; workload is memory-bound)
- XLA CPU multi-threading: 0% (operations too small per vmap iteration)
- pmap with 2-4 CPU devices: +10-30% (mostly from less padding, not true parallelism; not worth the API complexity and GPU incompatibility)

### Speedup Breakdown

The 2.4-2.8× improvement comes from two sources:

1. **Eliminated padding waste** (~1.5-2×): Old code padded each per-ensemble sub-batch to 8 poses; merged approach pads the full batch to the next power-of-2 → dramatically less wasted compute
2. **Per-pose efficiency gain** (~1.5×): Larger vmap batches enable XLA to generate more efficient code (better amortization of loop overhead, prefetching)

### GPU Impact

All changes are **GPU-positive**:

- Larger batches → better GPU occupancy
- No per-ensemble Python loop → fewer kernel launches
- No hardware-specific flags (multi-threading is CPU-only)

### Legacy ATTRACT Benchmark (for comparison)

| Config | Rate | vs JAX optimized |
|--------|------|-----------------|
| Legacy single-thread (5k poses, .grid) | 47.0 p/s | 3.5-4.0× faster |
| Legacy --np 12 (10k poses, SHM grid) | 61.9 p/s | 4.6-5.3× faster |

Legacy --np 12 achieved only **1.32× parallel scaling** (user predicted this — E-core CPU + memory contention).

### Recommended Restart Command

Kill PID 134725 and restart with optimized oracle:

```bash
cd /home/sjoerd/attract-namespace/attract-jax/util
kill 134725  # kill old run
nohup env XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" OMP_NUM_THREADS=1 \
  conda run --no-capture-output -n jax python -u minfor.py \
  /home/sjoerd/attract-namespace/test/systsearch-ens1.dat \
  --grid /home/sjoerd/attract-namespace/test/receptorgrid.grid \
  --attract-par-npz /home/sjoerd/attract-namespace/attract-jax/attract-par.npz \
  --receptor-ens-list /home/sjoerd/attract-namespace/test/partner1-ensemble.list \
  --ligand-pdb /home/sjoerd/attract-namespace/test/ligandr.pdb \
  --oracle jax --epsilon 15.0 --energy-batch 256 --max-nb-cap 40 \
  --trace-every 100 \
  --out-prefix /home/sjoerd/attract-namespace/test/jax_full_run/full \
  > /home/sjoerd/attract-namespace/test/jax_full_run/full_v2.log 2>&1 &
```

Estimated wall time: **~4 hours** (vs ~9 hours with old code).

---

## Interior Voxel Culling Investigation (2026-02-15)

**Goal**: Identify "deeply interior" grid voxels whose neighbour-list correction is always strongly repulsive, and zero their nb lists to reduce `max_nb_cap` or skip them during `nb_energy_vectorized`.

### Scripts Created

- `attract-jax/util/analyze_grid_nb.py` — Grid neighbour count distribution analysis.
- `attract-jax/util/probe_kdtree_radii.py` — Receptor atom density around voxels at various radii.
- `attract-jax/util/precompute_interior_voxels.py` — Full pipeline: KD-tree pre-filter → vectorized LJ nb energy → identify interior voxels.
- `attract-jax/util/precompute_interior_voxels_v1.py` — Backup of earlier version (before KD-tree addition).

### Key Findings

**Grid structure** (66×63×67 = 278,586 voxels, spacing=0.9 Å):

- 148,456 active voxels (53.3% have nb > 0).
- 55,301 (37.3% of active) have nb ≥ 40 — these are the expensive ones under `max_nb_cap=40`.
- Max nb = 180, currently capped at 40.

**Cannot use potential grid to detect interior**: The potential grid stores the long-range LJ potential, which is typically *negative* (attractive) at interior voxels. Only the short-range nb correction (pairwise at `d < plateaudis`) is repulsive at deeply buried positions.

**KD-tree pre-filter with all-atom PDBs**:

- Uses `partner1-ensemble-aa.list` (1,688 atoms/member) instead of reduced PDBs (937 atoms).
- Radius = 1.5 + √3·0.9/2 ≈ 2.28 Å, min_nearby_atoms = 3.
- Rejects 136,899 / 148,456 active voxels (92.2%) as definitely *not* interior.
- Only 11,557 candidates (7.8%) pass to the expensive energy evaluation.
- Of these candidates, 11,550 (99.9%) have nb ≥ 40 — confirming they are deeply buried.

**Energy evaluation results** (11 ensemble × 31 alphabet types = 341 combos):

- Conservative margin: ±0.779 Å (half voxel diagonal) on all distances.
- At threshold = 100 kcal/mol: **10,616 voxels confirmed interior** (7.2% of all active).
- Energy distribution of candidates: 92% have min energy > 50; 74% > 500; 34% > 10,000.

### Impact Assessment

| Metric | Before | After culling |
|--------|--------|---------------|
| Active voxels | 148,456 | 137,840 |
| Voxels with nb ≥ 40 | 55,301 | ~43,751 |
| % of nb≥40 removed | — | **20.9%** |
| Max nb | 180 | 177 |

**Conclusion**: Culling removes ~20.9% of the expensive high-nb voxels. However, this does **not** directly translate to a 20% speedup because:

1. The JAX kernel evaluates `nb_energy_vectorized` for all atoms simultaneously, iterating up to `max_nb_cap=40` offsets per atom regardless.
2. Interior voxels are only visited by a fraction of ligand atoms per pose — the saving depends on how many atoms actually land on culled voxels during docking.
3. `max_nb` drops only from 180→177, so `max_nb_cap` cannot be lowered.

**Practical benefit**: Modest — estimated ~5-10% kernel speedup depending on how many ligand atoms visit deep-interior voxels. The culled grid can be produced by re-running without `--dry-run`. The optimization is *correct* (conservative margin, all ensemble members, all alphabet types) but may not justify the complexity for this test case.

---

## The Nb Grid Bottleneck

The nonbonded neighbour-list correction (`nb_energy_vectorized`) accounts for **>80% of wall time**. This is the inner loop: for each ligand atom, look up its grid voxel, iterate over up to `max_nb_cap` neighbour offsets, and compute pairwise LJ corrections.

The current JAX kernel processes all (atom, offset) pairs as a flat vmap — elegant and GPU-friendly, but on CPU it suffers from:

1. **No early termination**: padded offsets (nb=0) are computed and discarded. With cap180, most atoms have far fewer than 180 real neighbours, so significant compute is wasted.
2. **Memory-bound access pattern**: each atom looks up scattered grid voxels and receptor coordinates. XLA cannot optimize this into cache-friendly loops.
3. **vmap overhead on CPU**: XLA unrolls vmap into sequential operations; there is no CPU vectorization benefit for scattered memory access.

---

## Bucketing Analysis (2026-02-17)

Two bucketing implementations exist. Analysis of why one worked and the other didn't.

### Implementation 1: Crocodile Scorer (working)

**Source**: `crocodile-protocol/.../score-attract-jax.py` (tested on 1b7f nucleotide docking).

**Approach**: flattens all (pose, atom) pairs across the entire batch (e.g., 2M structures × 22 atoms = 44M entries), sorts by nb count in numpy, then processes chunks of ~1M entries. Each chunk dispatches `nb_energy()` with `ncontacts` as a **static argument** — XLA compiles a separate, fully-unrolled trace per distinct bucket threshold.

**Key design choices**:

- Sort in numpy via `pure_callback` (fast on CPU, avoids XLA's slow argsort).
- `ncontacts` is static → each bucket gets optimal XLA code with no loop overhead.
- Bucket thresholds are fine-grained at low end: `(0,1,2,3,4,5,10,15,20,30,...)`.
- Chunk size (`NB_CHUNK_SIZE=1M`) is large → sort cost amortized over huge batches.
- Adaptive chunk quadrupling when most atoms in a chunk share the same high bucket.

**Workload it was designed for**: millions of small-ligand (22 atoms) surface poses scored in bulk. No minimization, no per-pose gradient. Gradient (when enabled) is taken on the full batch via `jax.value_and_grad(main)`.

### Implementation 2: Current jax_scorer.py Bucketed Mode (~45× slower)

**Source**: `attract-jax/util/reproduce_grid_score.py` (bucketed branch in `main_ad`).

**Approach**: within a single `main_ad` call (which is vmapped for AD gradients), flattens (B poses × A atoms), sorts by bucket, dispatches `nb_energy_vectorized_k` per bucket via `fori_loop` over fixed-size chunks (`NB_BUCKET_CHUNK=4096`).

**Why it's slow**:

1. **`jnp.argsort` instead of numpy**: required because this path is vmapped and `pure_callback` doesn't support vmap batching. `jnp.argsort` on CPU is much slower than numpy.
2. **`fori_loop` + `dynamic_slice_in_dim`**: each step has overhead from slicing, validity masking, and `scatter.add`. Many steps for small buckets.
3. **Cannot batch AD gradients**: bucketed control flow causes memory explosion under vmapped reverse-mode AD. `jax_scorer.py` falls back to **sequential single-pose gradients** (`_vg_batch = None`), losing all batching benefit.
4. **Sort happens inside JIT**: cannot be hoisted to Python without breaking the vmap/AD chain.

**Net effect**: the bucketing saves some nb work but the overhead (slow sort, fori_loop, no gradient batching) more than negates it → 45× slower than legacy.

### Comparison

| Aspect       | Crocodile (working)               | Current bucketed (broken)             |
|--------------|-----------------------------------|---------------------------------------|
| Batch unit   | Flat atom pool across all poses   | Per-pose (vmapped for AD)             |
| Sort scope   | 44M entries, once, in numpy       | B×A entries, inside JIT, jnp.argsort  |
| Inner loop   | Static `ncontacts` (unrolled)     | Static `k` but in `fori_loop` chunks  |
| AD gradients | `value_and_grad` on full batch    | vmap of single-pose grad → blowup     |
| Amortization | Sort cost / 44M atoms → tiny      | Sort cost / (256×80) atoms → large    |
| Sweet spot   | Huge batch, small ligand, scoring | None (broken for all practical uses)  |

### The Fundamental Tension

Bucketing requires **sorting**, which is data-dependent control flow. JAX's functional model makes this awkward inside vmapped AD:

- The crocodile approach works by keeping the sort **outside JIT/vmap**, in Python/numpy. This is only possible when the entire scoring pass is a single forward call (no per-pose AD gradient needed, or AD taken on the full flattened batch).
- During minimization, `score_batch()` is called repeatedly with ~256 active poses. AD gradients are needed per pose. The sort must either happen inside JIT (slow) or the AD must be restructured to work on the flattened atom pool (complex but potentially viable).

### Bucketing vs Special Kernel: Conclusion

**Current bucketed mode in jax_scorer.py**: should be **removed**. It is slower than fixed mode in all tested configurations.

**Crocodile-style bucketing is viable for GPU production**, including minimization. The key insight: `value_and_grad` on the batch energy sum gives per-pose gradients correctly (cross-derivatives are zero since each pose's energy depends only on its own DOFs). This is already proven in the crocodile code. So the AD restructuring needed for minimization is straightforward — no vmap needed.

For ensemble grids (max_nb=180), bucketing to e.g. (8, 16, 32, 64, 128, 180) saves dramatically vs padding to 180. A surface atom with nb=5 processes 8 offsets instead of 180 — a ~20× reduction for that atom. The sort cost (numpy argsort of ~20k entries for 256 poses × 80 atoms) is negligible.

**A CUDA kernel is strictly better but complementary**: even a CUDA kernel benefits from pre-sorting atoms by nb count to reduce warp divergence (adjacent threads in a warp with nb=5 and nb=150 cause the fast thread to idle). So bucketing isn't replaced by native kernels — it's a useful technique at both levels.

**For CPU production**: C kernel with early termination dominates. No bucketing needed.

---

## Paths Forward: Nb Kernel Optimization

### Path A: Crocodile-Style Bucketing for GPU (pure JAX)

Adapt the proven crocodile bucketing approach for minimization:

- **Flatten (pose, atom) pairs** across the batch, sort by nb count in numpy (outside JIT).
- **Dispatch per bucket** with static `ncontacts` — XLA compiles optimal code per bucket.
- **AD via `value_and_grad` on the batch energy sum** — gives per-pose gradients correctly (cross-derivatives are zero). No vmap needed. Already proven in crocodile code.
- **Atom-level culling**: atoms with nb=0 are naturally excluded by the sort.

**Pros**: pure JAX, works on any GPU, AD-compatible, proven approach. Large speedup over fixed-cap for ensemble grids (surface atoms with nb=5 process 8 offsets instead of 180).
**Cons**: still wastes some compute on intra-bucket padding. Multiple JIT traces (one per bucket threshold). Sort overhead per `score_batch` call (negligible at 256×80 = 20k entries).

**Target**: default GPU production path for all energy functions.

### Path B: C Kernel for CPU

Write a C kernel for the nb energy+gradient, callable from Python via ctypes/cffi:

- Nested for-loops with early termination (break when nb list ends).
- Cache-friendly sequential atom traversal.
- Hand-coded analytical gradients (reference: Fortran `nonbon8.f`).
- Expected to match or approach legacy ATTRACT speed on CPU.

**Target**: CPU production. Straightforward to implement given the Fortran reference.

### Path C: CUDA Kernel for GPU (when justified)

Write a CUDA kernel for the nb energy+gradient:

- One thread per (pose, atom) pair — embarrassingly parallel.
- Pre-sort atoms by nb count (reuse bucketing logic) to reduce warp divergence.
- Per-thread loop with early termination.
- Shared memory for receptor coordinate caching if beneficial.
- Potentially **faster than legacy ATTRACT** due to massive parallelism.

**Target**: specific energy functions that bottleneck the heaviest GPU workloads (see assessment below).

### GPU Performance Assessment: Bucketed JAX vs C/CUDA Kernel

Assessment by Claude Opus 4.6, 2026-02-17.

#### The question

For production GPU use, is bucketed JAX (Path A) sufficient, or does one realistically always need a C/CUDA kernel (Paths B/C)? This matters because new energy functions will be added, and the answer determines whether each one requires a hand-written kernel or can rely on JAX with bucketing.

#### Bucketed JAX efficiency relative to a CUDA kernel

For the nb kernel specifically, XLA generates good GPU code from the bucketed JAX path. Each bucket dispatch becomes a GPU kernel where every thread processes one atom through a fully-unrolled loop of `ncontacts` gather-compute-accumulate steps. The memory access pattern (scattered reads from the grid and receptor arrays) is identical to what a hand-written CUDA kernel would do.

The efficiency loss from bucketed JAX vs CUDA is **intra-bucket padding**: an atom with nb=17 in a bucket capped at 20 wastes 3 iterations. With fine-grained thresholds (1,2,3,4,5,10,15,20,...), this waste averages roughly 10-30% of actual work. So bucketed JAX operates at approximately **70-90% of theoretical CUDA efficiency** for the nb kernel itself.

However, this relative efficiency says nothing about the absolute gap vs legacy ATTRACT, which is the real question.

#### Absolute performance: the XLA constant-factor gap

On CPU, JAX with cap40 (fixed, no bucketing) is already ~12× slower than legacy multi-core ATTRACT for the full minimization run. This gap comes from XLA overhead: vmap unrolling into sequential operations, poor codegen for scattered memory access, Python/JIT dispatch. Bucketing eliminates padding waste but does not address this ~12× constant factor.

On GPU, some of this gap shrinks: vmap maps to real parallelism, and GPUs handle scattered gathers via latency hiding. But legacy ATTRACT is already multi-core, and a CUDA kernel would also have early termination (no padding at all). **The magnitude of the GPU speedup over JAX-on-CPU for this code is unknown — no GPU benchmarks exist.**

#### Per-workload verdict

**Ensemble + large ligand (this test case, max_nb=180)**:

- JAX-CPU cap180 fixed: ~27 hours (~35× slower than legacy).
- JAX-CPU cap180 bucketed: ~8 hours (bucketing reduces effective cap to ~20-30 average, comparable to cap40 fixed). Still ~12× slower than legacy.
- JAX-GPU bucketed: unknown, but the ~12× CPU constant factor is unlikely to fully close on GPU. Estimated ~3-5× slower than legacy multi-core. **Not competitive for production.** A C kernel (CPU) or CUDA kernel (GPU) is needed.

**Single-conformation + large ligand (max_nb ~20-30)**:

- Padding waste is small even without bucketing (pad to 32).
- The ~12× XLA constant factor still applies on CPU, but the absolute time is much lower.
- On GPU, JAX fixed-cap may be adequate since the nb kernel is a smaller fraction of total work. **Unknown — needs benchmarking.**

**Small ligand (any grid)**:

- Nb kernel is cheap per pose regardless of max_nb.
- JAX-GPU likely adequate. Bucketing may not even be needed.
- **Probably viable for production without a special kernel.**

#### CPU: bucketing does not help

On CPU, bucketing addresses the wrong bottleneck. The ~12× constant-factor gap (cap40 JAX vs legacy) is not caused by padding — it comes from XLA's CPU codegen: vmap unrolled to sequential operations, scattered memory access that XLA cannot optimize into cache-friendly loops, and Python/JIT dispatch overhead. Bucketing eliminates padding waste but leaves the ~12× untouched.

Even in the best case (single-conf, max_nb ~25): padding waste is only ~25% (pad to 32), so bucketing saves at most ~1.2× on the nb kernel. The ~12× constant factor remains. **On CPU, a C kernel is the only viable production path. Bucketing is strictly a GPU technique** — it trades padding waste for parallelism, which only pays off with thousands of GPU threads.

#### Conclusion

Bucketed JAX is not a universal production solution. For the heaviest workload (ensemble + large ligand), the XLA constant-factor overhead (~12× on CPU, unknown on GPU) means a C/CUDA kernel is needed for production performance. For lighter workloads (single-conf or small ligand), bucketed JAX may be sufficient — but this is unproven.

**Implication for new energy functions**: implement in JAX first (research mode). For production, measure the actual wall-time gap. If the energy function dominates and the workload is heavy, a C/CUDA kernel will be needed. The architecture should make it easy to swap in a native kernel per-function, but many energy functions may never need one — it depends on how much time they consume relative to the nb kernel.

### Interface

All paths expose the same oracle API: `score_batch(ens, dofs) → (energies, gradients)`. The oracle selects the kernel at init time based on hardware and configuration (`--nb-kernel jax|jax-bucketed|c|cuda`).

### Dual-Purpose Architecture

The codebase has a **dual function**:

1. **Research mode** (JAX, fixed cap): implement and iterate on energy functions in Python. JAX provides automatic differentiation and GPU parallelization for free. Correctness is the priority; performance is secondary.
2. **Production mode** (bucketed JAX on GPU, C on CPU, or CUDA when justified): for workloads where performance matters. Validated against the JAX reference implementation.

This duality is appropriate and should be embraced. The JAX implementation serves as the reference/ground-truth, and production kernels are validated against it. New energy functions follow the path: JAX implementation → bucketing for GPU production → CUDA only if profiling shows it's the bottleneck at scale.

---

## Workload Analysis

The nb kernel workload varies dramatically depending on the docking scenario. Three factors dominate:

### Factor 1: Plateau Distance (Ensemble vs Single Conformation)

- **Ensemble docking** (e.g., xylanase demo): a single receptor grid is built with a large plateau distance (~10 Å; nb stored up to ~12 Å). This exploits the fact that long-range LJ contributions are conformer-independent, but it pushes more work into the nb correction. The exact closest-neighbour list differs per conformer — this is why `max_nb_cap=40` gave incorrect results (the xylanase grid has max_nb=180).
- **Single-conformation docking** (more typical): plateau distance ~5 Å, nb stored up to ~7 Å. Far fewer neighbours per voxel — max_nb is much lower (TBD: measure typical values). A lower cap may be sufficient and correct.

**Implication**: the ensemble grid is the worst case for the nb kernel. Single-conformation grids may not need bucketing at all if max_nb is already small.

### Factor 2: Ligand Size

- **Protein–protein docking**: ligand has hundreds of atoms (comparable to receptor). Each pose processes many atoms through the nb kernel — high per-pose cost, but a single pose provides enough atoms for GPU parallelism.
- **Peptide / nucleotide docking**: ligand has tens of atoms (1–2 orders of magnitude smaller). Per-pose nb cost is low, but more poses are needed to fill GPU warps/batches efficiently.

**Implication**: for small ligands, the nb kernel is cheap per pose, and the overhead of bucketing/sorting may not be justified. For large ligands, the nb kernel dominates and optimization has the highest payoff.

### Factor 3: Minimization Trajectory

During minimization, a pose passes through three regimes:

1. **Far from receptor**: no atoms in the grid → nb cost is zero. Dominated by the potential grid lookup.
2. **Clashing (line search overshoot)**: many atoms deeply buried → uniformly high nb counts → low variance → padding waste is modest even without bucketing.
3. **Near-surface (converged)**: mix of buried, surface, and exposed atoms → high variance in nb counts → bucketing provides the most benefit on GPU.

The near-surface regime is where poses spend most function evaluations (converging), so it dominates wall time.

### Workload Matrix

| Scenario                        | Plateau | Ligand size | max_nb | C/CUDA kernel value | JAX-vs-legacy gap   |
|---------------------------------|---------|-------------|--------|---------------------|---------------------|
| Protein–protein, ensemble       | 10 Å    | large       | ~180   | **High**            | ~35× (measured)     |
| Protein–protein, single conf    | 5 Å     | large       | ~20-30 | **High**            | ~6-7×? (hypothesis) |
| Peptide/nucleotide, ensemble    | 10 Å    | small       | ~180   | Medium              | unknown             |
| Peptide/nucleotide, single conf | 5 Å     | small       | ~20-30 | Low                 | unknown             |

**Note**: only the ensemble protein–protein case has been benchmarked. The single-conformation estimates (~6× less nb work due to lower cap) are hypothetical — the actual JAX-vs-legacy gap needs to be measured. Other constant-factor overheads (grid lookup, potential evaluation, Python dispatch) may keep the gap larger than the nb reduction alone would predict.

### Optimization Priority by Hardware

**CPU production** → Path B (C kernel). Early termination in nested for-loops eliminates padding waste entirely. Straightforward port from Fortran `nonbon8.f`.

**GPU production (light workloads)** → Path A (bucketed JAX). Adequate for single-conformation grids and small-ligand docking. Default starting point for new energy functions.

**GPU production (heavy workloads)** → Path C (CUDA kernel). Required for ensemble + large ligand at production scale, where the XLA constant-factor overhead makes bucketed JAX uncompetitive. The architecture should make it easy to swap in a CUDA kernel per energy function.

**Research / prototyping** → current JAX kernel (fixed cap) is adequate. No optimization needed for correctness testing on small pose counts.

---

## Next Steps

1. **Prototype C nb kernel** (Path B) — port `nonbon8.f` logic to C with ctypes/cffi interface. Validate against JAX kernel, benchmark against legacy ATTRACT.
2. **Adapt crocodile bucketing for minimization** (Path A) — restructure AD path to use `value_and_grad` on batch sum. Validate per-pose gradients match vmapped approach.
3. **Remove broken bucketed mode** — delete the `nb_mode="bucketed"` path from `reproduce_grid_score.py` and `jax_scorer.py`, along with the sequential single-pose gradient fallback (`_vg_batch = None` / `for j in range(m)` loop) that only exists to work around it. The vmapped `_vg_batch` handles all cases including batch size 1.
4. **Result validation** — compare cap40 full run output (RMSD, fnat) against legacy ATTRACT.
5. **Correct full run** — re-run 165k poses with cap180 (correct energies) using C kernel or bucketed JAX. Expected wall time: ~1 hour (C kernel). Bucketed JAX wall time is unknown — depends on the actual nb distribution during minimization (needs profiling, see below).
6. **Measure single-conformation grid** — build a grid with plateau=5 Å, check max_nb (~20-30 expected), validate that fixed-cap JAX kernel is adequate for this case.
7. **CUDA kernel** (Path C, long-term) — once C kernel is proven, port to CUDA with pre-sorted atoms for GPU production.

---

## Hardware

- **CPU**: 12th Gen Intel i5-1235U (10 cores / 12 threads)
- **RAM**: 30GB (22GB free), ~2GB per JAX process
- **No GPU** — CPU-only JAX
