# Plan: Nonbonded Force Field Evaluation and Development

## 1. Motivation and Goals

### 1.1 The Problem

The current NB kernel (`nb_kernel.cpp` inside `attract-jax/native/`) hardcodes the ATTRACT force field: 8/6 Lennard-Jones
with distance-dependent dielectric (rdie), Euler angle rotation, and always computes gradients.
This makes it impossible to support alternative force fields, rotation parameterizations, or
energy-only evaluation without duplicating the entire 400-line pose loop.

At the same time, force field development today requires deep C++ knowledge. The target audience
for adding new force fields — computational scientists comfortable with Python — should not need
to understand templates, build systems, or wrapper boilerplate. They should write physics, and
the tooling should handle everything else.

### 1.2 What We Want

1. **Multiple force fields** as drop-in modules, each defined by a small set of `.h` files
   containing pure physics (LJ-like and electrostatics interactions).

2. **Multiple rotation parameterizations** (Euler angles, rotation vectors, potentially others)
   selectable independently of the force field.

3. **Energy-only evaluation** as a first-class mode, not a hack on top of energy+gradient.
   This matters because potential grid precomputation only needs energies, and skipping all
   gradient machinery (torque matrix, dR/dq derivatives, force accumulation) is a significant
   speedup.

4. **Zero performance cost** compared to hand-duplicated, specialized kernels. Every force field
   / rotation / grad combination compiles to its own fully-inlined, monomorphized code path.

5. **A progressive Python-side optimization pipeline** that lets developers start with naive
   JAX evaluation and incrementally add acceleration (neighbor list grids, potential grids,
   NB kernels) — each stage validated against the naive reference.

6. **Minimal developer burden**: write a Python energy function, let an AI agent generate
   the C `.h` files, run a codegen script, and the build system handles the rest.

---

## 2. C++ Kernel Architecture

### 2.1 The Three Template Axes

The pose loop is parameterized by three compile-time axes:

- **RotPolicy** — how rotation DOFs map to a 3x3 rotation matrix and its derivatives.
- **FFPolicy** — the nonbonded force field (LJ variant + electrostatics variant).
- **ComputeGrad** (bool) — whether to compute gradients or energy only.

```cpp
template <typename RotPolicy, typename FFPolicy, bool ComputeGrad>
inline void run_pose_loop_fused(
    const NbFusedStepData *step,
    const NbFusedGridData *grid,
    const NbGlobalData *global,
    double *out_energy,
    double *out_grad);
```

The pose loop lives in `pose_loop.h`. It is never modified when adding a new force field or
rotation scheme.

### 2.2 Rotation Policies

Each rotation policy is a struct in its own `.h` file with two static inline functions:

```cpp
// euler_rot.h
struct EulerRot {
    // Compute rotation matrix only (energy-only path).
    static inline void rot_only(const double *dofs, double R[9]);

    // Compute rotation matrix AND dR/dq derivatives (gradient path).
    static inline void rot_torque(const double *dofs, double R[9], double pm2[3][3][3]);
};
```

The pose loop calls `rot_only` when `ComputeGrad` is false (avoiding all `pm2` computation)
and `rot_torque` when true. The `pm2` tensor encodes `dR/d(q_j)` for the specific
parameterization; the torque contraction `g_j = sum_{k,l} pm2[k][j][l] * torque[k][l]`
is parameterization-agnostic and stays in the pose loop.

**Why this split matters:** `rot_only` skips computing 27 derivatives that are never used
in energy-only mode. For rotation vectors, the Rodrigues formula derivatives involve
significantly more work than the rotation matrix itself, so this is not trivial savings.

Initial implementations:

- `euler_rot.h` — the existing ATTRACT Euler convention (phi, ssi, rot).
- `rotvec_rot.h` — rotation vector via Rodrigues formula, with Taylor expansion near θ=0.

### 2.3 Force Field Policies

Each force field is defined by up to four `.h` files in a dedicated directory:

```
forcefields/nonbon8/
    lj.h          — lj_energy(): energy-only LJ evaluation (REQUIRED)
    elec.h        — elec_energy(): energy-only electrostatics (REQUIRED)
    lj_grad.h     — lj_grad(): energy+gradient LJ evaluation (OPTIONAL)
    elec_grad.h   — elec_grad(): energy+gradient electrostatics (OPTIONAL)
    ff.yaml       — declares which rotation schemes are supported
```

The function signatures are fixed and documented in the skeleton files (see Section 2.5).
The functions are pure physics: they receive interaction parameters, distance information,
and produce energy (and optionally gradient) outputs. They know nothing about rotation,
poses, grids, or parallelism.

**Why four separate files instead of one:** The target developer may be a Python-comfortable
scientist using an AI agent to translate their Python code to C. If `lj_grad.h` is missing,
the compiler error is unambiguous: `#include "lj_grad.h": No such file or directory`. A
missing function inside an existing file produces cryptic linker errors. The file boundary
is the cleanest possible contract.

**Why energy+gradient rather than gradient-only:** The energy and gradient computations share
all expensive intermediates (powers of `1/r`, LJ terms, etc.). The gradient is typically just
a few extra multiplies on top. Separating them would either duplicate the expensive work or
require passing intermediates through an awkward interface.

> **Note — legacy gradient bug (discovered 2026-03-05):** The legacy
> `grid_calculate.cpp` (`_calc_potential` and `_calc_potential_elec`) contains
> a math error in the plateau region: after overwriting `dsq = plateaudissq`
> and `rr2 = 1/plateaudissq`, the scaled displacement `dd` is computed with
> the post-clamp `rr2` instead of the original `1/dsq_orig`. This incorrectly
> scales gradient magnitudes inside the plateau sphere by `dsq_orig / plateaudissq`.
> **Consequence:** gradient channels in legacy `.grid` files are wrong for
> voxels inside the plateau sphere. The C kernel (`pose_loop.h`) uses the
> correct formula. **Do not use legacy `.grid` gradient channels as a
> reference** when validating new gradient implementations.

**Concrete initial force fields:**

- `nonbon8/` — 8/6 LJ with rdie (distance-dependent dielectric). This is the classic
  ATTRACT potential.
- `nonbon12/` — 12/6 LJ with cdie (constant dielectric). Standard molecular mechanics.

The `cdie`/`rdie` choice is baked into the electrostatics implementation, not a runtime flag.
This is deliberate: it's a physics choice that affects the functional form, and making it
compile-time allows the compiler to eliminate the unused branch entirely.

### 2.4 Design Constraint: Fixed Parameter Layout

Each pairwise interaction function (LJ and electrostatics) receives exactly **8 doubles**
as its per-type-pair parameters, passed by value. These are stored as flat arrays indexed
by `rec_type * nlig_types + lig_type` (8 arrays of length `nrec_types * nlig_types`, or
equivalently one array of 8-tuples). The pose loop loads them by fixed offset and passes
them to the force field function.

For the current ATTRACT force field, the LJ parameters are (rc, ac, emin, rmin2, ivor,
unused, unused, unused). For electrostatics, the charge is a per-atom quantity combined
at runtime, not a type-pair parameter. A different force field may use all 8 slots.

This fixed-width layout is a deliberate constraint. It allows the compiler to:

- Load parameters at known offsets (no pointer chasing in the inner loop).
- Keep all 8 values in registers (64 bytes, one cache line).
- Vectorize the inner loop without variable-stride memory access.

Unused parameter slots carry negligible cost — they are loaded but the compiler
eliminates any dead computation on them.

**Unused parameter slots on the Python side:** Python provides each of the 8 parameter
slots as a 2D array (or flattened 1D array) of shape `(nrec_types, nlig_types)`. For
parameters that a force field does not use, Python must still pass a valid array —
passing NULL would cause undefined behavior in C, even if the compiler would optimize
away the load at higher optimization levels (UB is UB regardless of `-O` level, and the
compiler may exploit it to break other code). Instead, Python should pass a zeroed-out
dummy array of the correct size:

```python
dummy = np.zeros(nrec_types * nlig_types)
```

This is allocated once and shared across all unused parameter slots. The pose loop loads
a zero from it, the force field function ignores the value, and the compiler eliminates
any dead computation. Zero runtime cost, no undefined behavior.

**Consequence for force field developers:** A new force field must fit its per-type-pair
parameters into 8 doubles. This is generous for typical pairwise potentials. If a force
field needs fundamentally different parameterization — say, per-atom rather than per-pair
parameters, or more than 8 parameters per pair — it would require modifying
`pose_loop.h`, which breaks the "never touch the template" principle. Such cases are
expected to be rare and would be handled as special-purpose extensions.

On the Python side, this constraint is softer. JAX vmap over pairs can pass arbitrary
parameter structures without performance penalty (JAX traces and compiles the full
computation graph regardless of parameter shape). So the Python path can accommodate
force fields with non-standard parameterization even when the C kernel path cannot.

### 2.5 The Pose Loop Template

The template `run_pose_loop_fused` in `pose_loop.h` contains the grid lookup, neighbor
iteration, plateau correction, and OpenMP parallelization. It calls into the policies via:

```cpp
// Rotation
if constexpr (ComputeGrad)
    RotPolicy::rot_torque(pose_dofs, R, pm2);
else
    RotPolicy::rot_only(pose_dofs, R);

// Force field (inner loop, per neighbor pair)
if constexpr (ComputeGrad) {
    FFPolicy::lj_grad(..., e0, gx0, gy0, gz0);
    // ... plateau evaluation ...
    FFPolicy::elec_grad(..., ee0, egx0, egy0, egz0);
    // ... force/torque accumulation ...
} else {
    FFPolicy::lj_energy(..., e0);
    // ... plateau evaluation ...
    FFPolicy::elec_energy(..., ee0);
    // no force accumulation, no torque matrix, no dR/dq contraction
}
```

The `if constexpr` branches are resolved at compile time. In the energy-only instantiation,
the compiler eliminates: the `pm2` array, the `torque[3][3]` matrix, all `fhx/fhy/fhz`
accumulation, the torque contraction loop, the translation gradient accumulation, and the
force reduction. The generated code is identical to a hand-written energy-only kernel.

### 2.5 Codegen Script

A Python script `codegen_ff.py` with two modes:

**Init mode:** `python codegen_ff.py init <name> <directory>`

Creates the directory and generates:

- `lj.h` — skeleton with the `lj_energy` signature, parameter documentation, and a TODO body.
- `elec.h` — skeleton with the `elec_energy` signature.
- `ff.yaml` — default configuration declaring supported rotation schemes and noting that
  `lj_grad.h` / `elec_grad.h` should be created to enable gradient support.

The skeleton files serve as documentation. A developer (or AI agent) reading `lj.h` sees
exactly what function to implement, what every parameter means, and what the return
semantics are. The file also explains that creating `lj_grad.h` with the `lj_grad` signature
will unlock gradient-capable kernel variants, and likewise for `elec_grad.h`.

**Codegen mode:** `python codegen_ff.py codegen <name> <directory>`

Reads `ff.yaml`, inspects which `.h` files exist, and generates:

- A single `.cpp` file that `#include`s the pose loop template, the relevant rotation policy
  headers, and the force field `.h` files. It instantiates the template for each valid
  combination and emits `extern "C"` wrapper functions.
- The wrapper function names follow the convention: `nb_kernel_<rot>_<grad|energy>`.
  For example, `nb_kernel_euler_grad`, `nb_kernel_rotvec_energy`.

If `lj_grad.h` and `elec_grad.h` both exist, grad wrappers are generated. If either is
missing, only energy wrappers are generated. The YAML controls which rotation schemes
to instantiate (e.g. a force field might only support `euler` initially).

### 2.6 Build System

The Makefile scans for force field directories (those containing `ff.yaml`), invokes the
codegen script if the generated `.cpp` is stale, and compiles each into a shared library:

```
nb_kernel_nonbon8.so    — contains nb_kernel_euler_grad, nb_kernel_euler_energy,
                          nb_kernel_rotvec_grad, nb_kernel_rotvec_energy
nb_kernel_nonbon12.so   — same set of symbols
```

One `.so` per force field. Python loads the `.so` by name and probes for available symbols
using `ctypes`. This means:

- The build produces a clean, self-contained artifact per force field.
- Python discovers capabilities at runtime — no hardcoded function lists.
- Adding a new force field requires zero changes to existing build rules.

### 2.7 File Layout

```
nb_kernel/
    include/
        pose_loop.h          — template pose loop (never modified for new FFs)
        euler_rot.h           — Euler rotation policy
        rotvec_rot.h          — rotation vector policy
        nb_kernel.h           — shared data structures (NbFusedStepData, etc.)
    forcefields/
        nonbon8/
            lj.h, elec.h, lj_grad.h, elec_grad.h, ff.yaml
        nonbon12/
            lj.h, elec.h, lj_grad.h, elec_grad.h, ff.yaml
    codegen_ff.py
    Makefile
```

---

## 3. Python-Side Progressive Optimization Pipeline

### 3.1 Overview

A force field developer follows a progression of increasingly optimized evaluation modes.
Each mode is validated against the previous one, with naive mode as the ultimate ground truth.
The developer is only required to write Python code; C kernel support is optional and
comes later (or via AI-assisted translation).

The stages, in order:

```
Stage 0: Python energy function (developer writes this)
    |
Stage 1: Naive mode — vmap over all atom pairs
    |
Stage 2: Neighbor list grid — KD-tree precomputation, JAX evaluation
    |
Stage 3: Potential grid — precomputed energy grids, JAX evaluation
    |
Stage 4: NB kernel — compiled C kernel with neighbor list grid
    |
Stage 5: NB kernel potential grid — C kernel for grid precomputation
```

Stages 2-5 are independent optimizations that can be mixed. The Python framework handles
orchestration; the developer only ever writes Stage 0.

### 3.2 Stage 0: The Developer's Python Function

The developer writes a single Python function for each interaction type: one for LJ-like
interactions and one for electrostatics. These operate on a single atom pair and return
an energy scalar. Example:

```python
def lj_energy(rc, ac, emin, rmin2, ivor, dsq, rr2):
    """8/6 Lennard-Jones energy for a single atom pair."""
    rr23 = rr2 * rr2 * rr2
    rep = rc * rr2
    vlj = (rep - ac) * rr23
    if dsq < rmin2:
        return vlj + (ivor - 1) * emin
    return ivor * vlj
```

This function is pure physics, no framework dependencies beyond basic math. It is the
ground truth definition of the force field.

### 3.3 Stage 1: Naive Mode

The framework applies the developer's function over all ligand-receptor atom pairs using
`jax.vmap`. This is the reference implementation:

- Correct by construction (it evaluates every pair).
- Slow (O(N*M) with no cutoff, no spatial indexing).
- Serves as the ground truth for validating all subsequent stages.

**Why vmap and not Python for-loops:** JAX traces and compiles vmapped functions efficiently.
Python for-loops over atom pairs would be prohibitively slow even for small systems, and
JAX's tracing overhead on loops is severe.

The developer may also define a dampening/cutoff range at this stage. This does not change
the naive mode computation (all pairs are still evaluated), but it defines the transition
where the potential smoothly goes to zero. This range determines the neighbor distance used
in all subsequent spatial acceleration stages.

### 3.4 Stage 2: Neighbor List Grid

**What:** Precompute a spatial grid over the receptor, where each voxel stores the list of
receptor atoms within the cutoff distance. Ligand atom evaluation then only considers
neighbors from the relevant voxel.

**Why:** Reduces the inner loop from O(N_rec) to O(k) where k is the average neighbor count.
For typical biomolecular systems with a ~10 Angstrom cutoff, k is perhaps 50-200 while
N_rec may be thousands.

**How:** Using `scipy.spatial.cKDTree` (or similar), build a KD-tree from receptor atom
coordinates, query it against grid voxel centers with the cutoff radius, and store the
result as a (count, offset) + concatenated neighbor index structure — the same format the
NB kernel already consumes. Prototype code for this already exists in
`test-calc-grid-energy.py` (lines 94-103, single atom type). This needs to be generalized
to multiple atom types and packaged as a reusable component.

**Performance note:** JAX evaluation over neighbor list grids is currently ~100x slower than
the C NB kernel in worst-case benchmarks. However, it may be competitive for smaller systems,
systems with low neighbor counts, or when GPU acceleration is available. This is worth
investigating — JAX-only evaluation may be a legitimate performance contender in some regimes.

**Bucketing:** Voxels can be grouped by neighbor count so that JAX processes batches of
similar-sized neighbor lists without excessive padding. This can significantly reduce wasted
computation in the JAX path.

### 3.5 Stage 3: Potential Grid

**What:** Precompute the energy contribution of each atom type on a spatial grid. At
evaluation time, ligand atom energies are looked up from the grid (with interpolation)
rather than computed from pairwise interactions.

**Why:** Reduces per-ligand-atom cost from O(k) neighbor evaluations to an O(1) grid lookup.
This is the classic ATTRACT acceleration strategy and is critical for large-scale docking
where millions of poses are evaluated.

**How:** For each receptor atom type, evaluate the force field function at every grid voxel
(using the neighbor list grid from Stage 2 to avoid redundant computation), and store the
result. Existing prototype code for single-atom-type potential grid computation exists
inside the ATTRACT-JAX tests and needs to be generalized.

In legacy ATTRACT, potential grid computation takes ~10 seconds on all cores. With the NB
kernel available, similar performance is expected here.

**Interaction with NB kernel:** The potential grid precomputation machinery should detect
the presence of the required NB kernel `.so` and use it if available (specifically, the
energy-only variant). If no kernel is found, it falls back to JAX evaluation over the
neighbor list grid. This is where the energy-only kernel variant pays for itself —
potential grid precomputation is the primary consumer of energy-only evaluation.

**Potential grid modes:** Potential grids support three evaluation modes, depending on
what the application needs and what the force field provides:

1. **Energy-only potential grid.** The grid stores only energies. At evaluation time,
   ligand atom energies are looked up and summed. This is sufficient for scoring (ranking
   poses without minimization) and is the simplest mode. It requires only the energy-only
   Python functions (`lj.py`, `elec.py`) that the developer writes at Stage 0. No gradient
   functions are needed.

2. **Autodiff potential grid.** The grid stores only energies, but gradients are obtained
   at evaluation time via JAX automatic differentiation through the grid interpolation.
   JAX differentiates the lookup + interpolation operation with respect to the ligand
   coordinates and rotation DOFs. This requires no additional work from the developer —
   the energy-only Python functions are sufficient, and JAX handles the chain rule through
   the grid lookup automatically. The gradients are with respect to the *interpolated*
   grid values, so they are approximate (limited by grid resolution), but this is often
   acceptable for minimization.

3. **Stored-gradient potential grid.** The grid stores both energies and their spatial
   gradients (dE/dx, dE/dy, dE/dz) at each voxel. At evaluation time, both energies and
   gradients are looked up directly — no autodiff needed. This is more accurate than
   autodiff mode (the gradients are computed from the actual pairwise interactions, not
   from interpolated grid values) and faster at evaluation time (no backward pass).

   **This mode requires a Python gradient function.** The developer must provide a
   function that computes both energy and its Cartesian gradient for a single atom pair.
   This is a separate development step beyond Stage 0:

   ```python
   # myff/lj_grad.py (OPTIONAL — only needed for stored-gradient potential grids)
   def lj_grad(rc, ac, emin, rmin2, ivor, dsq, rr2, dx, dy, dz):
       """LJ energy and Cartesian gradient for a single atom pair.

       Additional parameters beyond lj_energy:
       dx, dy, dz : float — (atom_i - atom_j) / dsq (scaled displacement)

       Returns
       -------
       energy : float
       gx, gy, gz : float — gradient components
       """
       ...
   ```

   The gradient function receives the same parameters as the energy function, plus the
   scaled displacement vector (dx, dy, dz), and returns energy plus three gradient
   components. The framework vmaps this over pairs to compute the gradient field at each
   voxel, then stores it in the grid alongside the energy.

**When to write the Python gradient function:** The developer does NOT need a Python
gradient function for energy-only scoring, nor for autodiff potential grids. The Python
gradient function is needed when the developer wants:

- Stored-gradient potential grids (the gradient field must be explicitly computed and
  stored, which requires a function that returns gradients).
- To produce C kernel gradient `.h` files — the C gradient files are *ports* of the
  Python gradient functions, not independent implementations. The expected workflow is:
  developer writes and validates the Python gradient function first, then translates it
  to C (manually or with AI assistance). Writing C gradient code de novo without a Python
  reference is possible (an AI agent could derive the gradient symbolically) but is not
  the standard path.

In practice, this means:

- Energy-only scoring: no gradient function needed at any stage, neither Python nor C.
- Minimization via autodiff potential grids: no gradient function needed.
- Minimization via stored-gradient potential grids: Python gradient function needed
  (`lj_grad.py`, `elec_grad.py`).
- Minimization via NB kernel: Python gradient function needed first, then ported to C
  gradient `.h` files (`lj_grad.h`, `elec_grad.h`).

The init skeleton should clearly document this progression — the developer creates
`lj_grad.py` and `elec_grad.py` when they move beyond energy-only evaluation, and then
creates the C `.h` files as translations of the Python functions when they want kernel
acceleration.

### 3.6 Stage 4: NB Kernel with Neighbor List Grid

**What:** The compiled C kernel from Section 2, called from Python via `ctypes`, evaluating
energy and/or gradients using the precomputed neighbor list grid.

**Why:** This is the fastest path for energy+gradient evaluation during minimization. The
C kernel with OpenMP parallelization and fully inlined, monomorphized code is the
performance ceiling.

**How:** Python loads `nb_kernel_<ffname>.so`, probes for available symbols, constructs the
`NbFusedStepData` / `NbFusedGridData` / `NbGlobalData` structures, and calls the
appropriate function. The existing harness code demonstrates this pattern.

### 3.7 Stage 5: NB Kernel for Potential Grid Precomputation

**What:** Use the energy-only NB kernel to accelerate potential grid precomputation.

**Why:** The potential grid must be recomputed when the receptor ensemble changes, which
may happen frequently in flexible docking workflows. Fast precomputation (seconds, not
minutes) makes this practical.

**How:** The potential grid precomputation code (Stage 3) checks for the energy-only kernel
symbol and dispatches to it when available. The grid computation is embarrassingly parallel
over voxels, which maps naturally to the kernel's existing OpenMP pose loop (each "pose" is
a voxel evaluation).

**Stored-gradient grids with NB kernel:** When both the energy-only and gradient C kernels
are available, the stored-gradient potential grid can be precomputed entirely via the C
kernel — the gradient kernel evaluates energy+gradient at each voxel, and both are stored.
This is the fastest path for precomputing stored-gradient grids.

### 3.8 Merging the Paths

Stages 3 and 4 are complementary, not competing:

- Potential grids (Stage 3) give O(1) per-atom lookup but require precomputation time and
  memory.
- NB kernel (Stage 4) gives exact pairwise evaluation with gradients, at O(k) per atom.

In practice, a docking run might use potential grids for coarse scoring of millions of poses,
then switch to NB kernel evaluation for gradient-based refinement of the top candidates.
The framework should make both paths available and let the application choose.

The following table summarizes what each evaluation path requires from the developer and
what it provides:

```
Path                          | Requires (Python)         | Requires (C)      | Provides
------------------------------|---------------------------|-------------------|------------------
Naive (Stage 1)               | energy functions          | —                 | Energy
Neighbor list grid (Stage 2)  | energy functions          | —                 | Energy
Potential grid, energy-only   | energy functions          | —                 | Energy
Potential grid, autodiff      | energy functions          | —                 | Energy + gradient
Potential grid, stored-grad   | gradient functions        | —                 | Energy + gradient
NB kernel, energy-only        | energy functions (ref)    | energy .h files   | Energy
NB kernel, energy+grad        | gradient functions (ref)  | gradient .h files | Energy + gradient
```

Note: "(ref)" means the Python functions serve as the reference implementation that the
C `.h` files are ported from. The Python functions are written and validated first; the
C files are translations of them.

Energy-only evaluation (scoring without minimization) is a first-class use case at every
stage. A developer who only needs scoring never writes gradient functions — neither in
Python nor in C.

---

## 4. Validation Strategy

### 4.1 Principle

Every optimization stage must be numerically validated against a reference. However, the
reference depends on where in the pipeline a cutoff or dampening range is introduced.

**Before cutoff is defined (Stage 1 only):** Naive mode evaluates all pairs with no
distance cutoff. This is the purest reference — the raw force field as written.

**After cutoff/dampening is defined:** Introducing a hard cutoff or dampening range
changes the physics — pairs beyond the cutoff no longer contribute, or their contribution
is smoothly attenuated. This is a deliberate, irreversible choice by the developer that
changes the numerical results. Once defined, naive mode *with* the cutoff/dampening
applied becomes the new reference. All subsequent stages (neighbor list grid, potential
grid, NB kernel) must match this modified naive mode.

The validation chain is therefore:

1. Stage 1 (naive, no cutoff) — ground truth for the raw force field.
2. Developer defines cutoff/dampening — this intentionally changes results.
3. Stage 1 (naive, with cutoff) — new ground truth for the short-range force field.
4. Stages 2-5 must match Stage 1-with-cutoff within numerical tolerance.

This distinction matters because without a cutoff, Stage 2 (neighbor list grid) cannot
match Stage 1 unless an infinite neighbor list distance is used (which defeats the
purpose). The cutoff is what makes spatial acceleration valid — it guarantees that pairs
outside the neighbor list distance contribute zero energy.

**Note on potential grids (Stage 3):** The cutoff/dampening distance used for neighbor
list construction does not necessarily constrain the potential grid. The neighbor list
grid is an intermediate used during potential grid *precomputation*, but the resulting
potential grid captures the sum of all contributions within the neighbor distance at each
voxel. A sufficiently large neighbor distance (e.g. 30 Å in the current implementation)
captures all physically significant interactions, making the potential grid effectively
exact even though a finite cutoff was used during its construction. The cutoff may even
be relaxed at the potential grid stage compared to direct neighbor list evaluation
(Stage 2), since the precomputation cost is paid once rather than per pose.

### 4.2 The Codegen Init Skeleton Should Generate a Test

When `codegen_ff.py init` creates a new force field directory, it should also generate a
test script (or test configuration) that:

1. Runs naive mode on a small test system.
2. Runs each available acceleration stage on the same system.
3. Compares energies (and gradients where applicable) within a tolerance.

This test is part of the developer workflow: implement the Python function, run the test
in naive mode to verify it works, then progressively enable acceleration stages and confirm
they match.

### 4.3 Kernel Validation

The NB kernel (Stages 4-5) must be validated against the JAX neighbor list grid evaluation
(Stage 2), which is itself validated against naive mode. This two-level validation catches
both physics errors (wrong force field implementation) and infrastructure errors (wrong
grid construction, off-by-one in neighbor lists, etc.).

---

## 5. Python Force Field Module Structure

### 5.1 What the Developer Creates

A force field is a Python package (directory with `__init__.py`) that lives under a
configurable location. The developer creates this package, and the framework discovers
and loads it at runtime. The package contains:

```
myff/
    __init__.py        — exports the force field interface
    lj.py              — LJ-like pairwise interaction (REQUIRED)
    elec.py            — electrostatics pairwise interaction (REQUIRED)
    params.py          — parameter loading/conversion (REQUIRED)
    ff.yaml            — metadata: cutoff, plateau, supported features
```

### 5.2 The Python Function Contract

Each interaction module (`lj.py`, `elec.py`) exports a single function that computes the
energy for one atom pair. These functions must be JAX-traceable — no Python control flow
that depends on array values (use `jnp.where` instead of `if`), no side effects, no
non-JAX operations.

```python
# myff/lj.py
import jax.numpy as jnp

def lj_energy(rc, ac, emin, rmin2, ivor, dsq, rr2):
    """LJ energy for a single atom pair.

    Parameters
    ----------
    rc : float     — repulsive coefficient for this atom type pair
    ac : float     — attractive coefficient for this atom type pair
    emin : float   — energy at the LJ minimum for this pair
    rmin2 : float  — squared distance at the LJ minimum
    ivor : int     — 1 for attractive pairs, 0 for repulsive
    dsq : float    — squared distance between the two atoms
    rr2 : float    — 1/dsq (precomputed reciprocal)

    Returns
    -------
    energy : float — pairwise energy contribution
    """
    rr23 = rr2 * rr2 * rr2
    rep = rc * rr2
    vlj = (rep - ac) * rr23
    return jnp.where(dsq < rmin2,
        vlj + (ivor - 1) * emin,
        ivor * vlj)
```

```python
# myff/elec.py
import jax.numpy as jnp

def elec_energy(charge, rr2, dsq):
    """Electrostatic energy for a single atom pair.

    Parameters
    ----------
    charge : float — product of the two atom charges, prescaled by felec
    rr2 : float    — 1/dsq
    dsq : float    — squared distance

    Returns
    -------
    energy : float — pairwise electrostatic energy contribution
    """
    ...
```

The function signatures are fixed by convention. The framework vmaps these over atom pairs
— the developer never writes loops, vmaps, or batching logic.

**Why `jnp.where` instead of `if`:** JAX traces functions symbolically. A Python `if`
on a traced value causes a tracing error. `jnp.where` evaluates both branches and selects
the result, which JAX can compile and differentiate. This is the standard JAX pattern for
conditionals.

### 5.3 Parameter Module

The `params.py` module handles loading force field parameters from an `.npz` file and
deriving any secondary quantities. It exports a function that returns a standardized
parameter container:

```python
# myff/params.py
import numpy as np
from collections import namedtuple

FFParams = namedtuple("FFParams", ("rc", "ac", "ivor", "emin", "rmin2"))

def load_params(npz_path):
    """Load force field parameters from an NPZ file.

    Returns an FFParams namedtuple with arrays indexed by
    (rec_type, lig_type).
    """
    par = np.load(npz_path)
    rc = par["rc"]
    ac = par["ac"]
    ivor = par["ivor"]
    emin = -27.0 * ac**4 / (256.0 * rc**3)
    rmin2 = 4.0 * rc / (3.0 * ac)
    return FFParams(rc, ac, ivor, emin, rmin2)
```

The `.npz` file itself (e.g. `attract-par.npz`) is not part of the force field module —
it is a data file supplied separately. The current `attract-par.npz` already exists and
is converted from legacy ATTRACT `.par` files via `attract-original/convert-attract-par.py`.

### 5.4 Force Field Metadata (ff.yaml)

```yaml
# myff/ff.yaml
name: myff
cutoff: 50.0              # squared distance cutoff for neighbor lists
plateau: true             # whether plateau correction is applied
plateau_distance_sq: 50.0 # distance² at which plateau kicks in
cdie: false               # constant dielectric (true) or distance-dependent (false)
potshape: 8               # LJ repulsive exponent (8 or 12)
```

This metadata is consumed by:

- The Python framework (to configure neighbor list construction and plateau correction).
- The C codegen script (to configure which kernel features to enable).
- The validation harness (to set up test conditions).

### 5.5 The Init Script Generates Python Skeletons Too

When `codegen_ff.py init myff forcefields/myff/` is run, it creates not only the C `.h`
skeletons (Section 2.5) but also the Python package:

- `myff/__init__.py` — imports and re-exports the interface
- `myff/lj.py` — skeleton with the `lj_energy` signature, docstring, and a TODO body
- `myff/elec.py` — skeleton with the `elec_energy` signature and a TODO body
- `myff/params.py` — skeleton parameter loader
- `myff/ff.yaml` — default metadata with comments explaining each field

The Python skeletons are the *starting point* for the developer. They implement the Python
functions first, validate them in naive mode, and only later (optionally) create the C
`.h` files for kernel acceleration.

### 5.6 Force Field Discovery and the CLI

**The problem:** The Python framework (currently `util/minfor.py` and `util/jax_scorer.py`)
needs to find and load the developer's force field module at runtime. The module could live
anywhere — in the attract-jax source tree, in the developer's project directory, or in a
separate package.

**The solution:** A CLI argument `--forcefield` (or `--ff`) that specifies a Python import
path or a filesystem path to the force field package. The framework resolves this to a
module and loads it.

```bash
# As a filesystem path (adds parent to sys.path, imports the directory as a package):
python util/minfor.py --ff ./forcefields/nonbon8 ...

# As a Python import path (for installed packages):
python util/minfor.py --ff myff ...
```

The existing `--attract-par-npz` argument in `util/minfor.py` already points to the
parameter file. This stays separate — the `.npz` is data, the force field module is code.
Together they fully define the physics.

**Discovery logic (pseudocode):**

```python
def load_forcefield(ff_spec):
    """Load a force field from a path or import string."""
    path = Path(ff_spec)
    if path.is_dir() and (path / "__init__.py").exists():
        # Filesystem path: add parent to sys.path, import as package
        sys.path.insert(0, str(path.parent))
        module = importlib.import_module(path.name)
    else:
        # Try as a Python import path
        module = importlib.import_module(ff_spec)

    # Validate the module exports the required interface
    assert hasattr(module, 'lj_energy'), f"{ff_spec} missing lj_energy"
    assert hasattr(module, 'elec_energy'), f"{ff_spec} missing elec_energy"
    assert hasattr(module, 'load_params'), f"{ff_spec} missing load_params"
    return module
```

**Kernel .so discovery:** When a force field module is loaded, the framework also looks
for a compiled kernel at a conventional location relative to the module:

```python
def find_kernel_so(ff_module):
    """Look for nb_kernel_<name>.so next to the force field module."""
    ff_dir = Path(ff_module.__file__).parent
    so_path = ff_dir / f"nb_kernel_{ff_module.__name__}.so"
    if so_path.exists():
        return ctypes.CDLL(str(so_path))
    return None
```

If found, the framework probes for available symbols (`nb_kernel_euler_grad`,
`nb_kernel_euler_energy`, etc.) and uses the fastest available path. If not found,
it falls back to JAX evaluation. This is transparent to the developer.

---

## 6. Developer Workflow Summary

Adding a new nonbonded force field, end-to-end:

1. **Run `codegen_ff.py init myff forcefields/myff/`** — creates the Python package
   skeleton (`lj.py`, `elec.py`, `params.py`, `__init__.py`, `ff.yaml`) and the C
   skeleton files (`lj.h`, `elec.h`). The skeletons contain full docstrings explaining
   the function signatures and expected behavior.

2. **Implement the Python energy functions** — edit `lj.py` and `elec.py`, filling in
   the function bodies. Implement `params.py` to load parameters from an `.npz` file.
   Define cutoff and plateau parameters in `ff.yaml`.

3. **Validate in naive mode** — run the generated test script, which applies the force
   field over all atom pairs via `jax.vmap` on a small test system. Confirm physically
   reasonable energies.

4. **Enable neighbor list grid evaluation** — the framework handles this automatically
   once `ff.yaml` defines a cutoff. Validate against naive mode.

5. **Enable potential grid precomputation** — validate against naive mode.

6. **(Optional) Create C kernel `.h` files** — either manually or with AI assistance,
   translate the Python functions to C. The skeleton `.h` files document exactly what's
   needed. Only `lj.h` and `elec.h` (C versions) are required for energy-only kernels.
   Add `lj_grad.h` and `elec_grad.h` for gradient support.

7. **Run `codegen_ff.py codegen myff forcefields/myff/`** — generates the C++ wrapper code.

8. **Run `make`** — builds `nb_kernel_myff.so`.

9. **Validate kernel against JAX** — the test script automatically detects the `.so` and
   includes kernel evaluation in the comparison.

Steps 1-5 require only Python. Steps 6-9 require creating (or AI-generating) up to four
small `.h` files and running two commands. The pose loop template, build system, Python
framework, and validation harness are never modified.

---

## 6. Implementation Order

### Phase 1: Refactor the C++ kernel

Extract the current `nb_kernel.cpp` into the template architecture: `pose_loop.h`,
`euler_rot.h`, `nonbon8/lj.h`, `nonbon8/elec.h`, etc. Validate that the refactored kernel
produces identical results to the current one.

### Phase 2: Add energy-only support

Implement the `ComputeGrad=false` path in the pose loop template. Add energy-only wrappers.
Validate against the grad version (energies must match).

### Phase 3: Add rotation vector support

Implement `rotvec_rot.h`. Validate by comparing rotvec gradients against finite-difference
derivatives of the energy.

### Phase 4: Add nonbon12

Implement `nonbon12/lj.h` and `nonbon12/elec.h` (12/6 LJ with cdie). Validate against
a Python reference.

### Phase 5: Codegen script

Implement `codegen_ff.py` with init and codegen modes.

### Phase 6: Build system

Implement Makefile rules for scanning force field directories, invoking codegen, and
building `.so` files.

### Phase 7: Python neighbor list grid component

Generalize the existing KD-tree prototype from `test-calc-grid-energy.py` into a reusable
component supporting multiple atom types.

### Phase 8: Python progressive evaluation pipeline

Implement the Python framework that orchestrates naive mode, neighbor list grid, potential
grid, and kernel evaluation. Implement auto-detection of kernel `.so` availability.

### Phase 9: Validation harness

Implement the test generation and cross-stage validation framework.

### Phase 10: Potential grid precomputation

Generalize the existing potential grid prototype into a reusable component. Integrate with
kernel auto-detection for acceleration.
