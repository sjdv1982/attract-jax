#!/usr/bin/env python3
"""
codegen_ff.py — Force field code generation for nb_kernel.

Usage:
    python codegen_ff.py init    <name> <directory>
    python codegen_ff.py codegen <name> <directory>

'init' creates a new force field directory with skeleton files:
  C headers:      lj.h, elec.h  (required, energy-only; create lj_grad.h /
                  elec_grad.h optionally to enable gradient kernel variants)
  Python package: __init__.py, lj.py, elec.py, params.py
  Metadata:       ff.yaml  (contains cpp_namespace, supported_rotations, etc.)

'codegen' reads <directory>/ff.yaml, inspects which .h files are present, and
writes <directory>/nb_kernel_<name>.cpp with:
  - #includes for rotation policies and FF physics headers
  - An FFPolicy struct that delegates to the FF C++ namespace
  - extern "C" wrappers for each (rotation × grad|energy) combination
  - Input validation helpers (validate_grad_inputs / validate_energy_inputs)

Gradient wrappers are only emitted when BOTH lj_grad.h and elec_grad.h exist.
The actual .h files in the directory always take precedence over has_lj_grad /
has_elec_grad flags in ff.yaml.

The generated .cpp is designed to be compiled from the native/nb_kernel/
root directory (where this script lives), with -I. so that includes are
resolved relative to that root.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Directory where this script lives = native/nb_kernel/ root
SCRIPT_DIR = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Rotation policy metadata
# ---------------------------------------------------------------------------
ROT_HEADER = {
    "euler": "include/euler_rot.h",
}
ROT_POLICY = {
    "euler": "EulerRot",
}
PLATEAU_MODES = {
    "correction": "Correction",
    "clamp": "Clamp",
}

# ---------------------------------------------------------------------------
# Validation helpers (static C++ source — identical for all force fields)
# ---------------------------------------------------------------------------
_VALIDATION_SRC = r"""
// ---------------------------------------------------------------------------
// Input validation helpers
// ---------------------------------------------------------------------------
namespace
{

  inline int validate_grad_inputs(
      const NbFusedStepData *step, const NbFusedGridData *grid,
      const NbGlobalData *global, const NbRunConfig *cfg,
      double *out_energy, double *out_grad)
  {
    if (!step || !grid || !global || !cfg || !out_energy || !out_grad)
      return 1;
    if (step->nposes < 0 || step->natoms < 0 || global->nrec <= 0 || global->nens <= 0)
      return 2;
    if (!step->dofs || !step->ens || !step->lig_coords || !step->lig_pivot ||
        !step->lig_type || !step->lig_charge)
      return 3;
    if (!grid->nr_neigh || !grid->nb_start || grid->spacing <= 0.0 ||
        grid->dim[0] <= 0 || grid->dim[1] <= 0 || grid->dim[2] <= 0)
      return 4;
    if (!global->nb_concat || !global->rec_coord || !global->rec_type ||
        !global->rec_charge || !global->rc || !global->ac || !global->emin ||
        !global->rmin2 || !global->ivor)
      return 5;
    if (global->nrec_types <= 0 || global->nlig_types <= 0)
      return 6;
    if (global->plateaudissq <= 0.0)
      return 7;
    return 0;
  }

  // Energy-only validation: out_grad may be NULL.
  inline int validate_energy_inputs(
      const NbFusedStepData *step, const NbFusedGridData *grid,
      const NbGlobalData *global, const NbRunConfig *cfg,
      double *out_energy)
  {
    if (!step || !grid || !global || !cfg || !out_energy)
      return 1;
    if (step->nposes < 0 || step->natoms < 0 || global->nrec <= 0 || global->nens <= 0)
      return 2;
    if (!step->dofs || !step->ens || !step->lig_coords || !step->lig_pivot ||
        !step->lig_type || !step->lig_charge)
      return 3;
    if (!grid->nr_neigh || !grid->nb_start || grid->spacing <= 0.0 ||
        grid->dim[0] <= 0 || grid->dim[1] <= 0 || grid->dim[2] <= 0)
      return 4;
    if (!global->nb_concat || !global->rec_coord || !global->rec_type ||
        !global->rec_charge || !global->rc || !global->ac || !global->emin ||
        !global->rmin2 || !global->ivor)
      return 5;
    if (global->nrec_types <= 0 || global->nlig_types <= 0)
      return 6;
    if (global->plateaudissq <= 0.0)
      return 7;
    return 0;
  }

} // namespace
"""

# ---------------------------------------------------------------------------
# Skeleton templates for 'init' mode
# ---------------------------------------------------------------------------


def _lj_h_skeleton(name: str) -> str:
    guard = f"{name.upper()}_LJ_H"
    return f"""\
/**
 * lj.h — Energy-only LJ evaluation for {name} (generated skeleton).
 *
 * Function signature (fixed contract; do not change):
 *
 *   double lj_energy(rc, ac, emin, rmin2, ivor, dsq, rr2)
 *
 * Parameters
 * ----------
 * rc    : repulsive coefficient
 * ac    : attractive coefficient
 * emin  : energy at the LJ minimum for this type pair
 * rmin2 : squared distance at the LJ minimum (Å^2)
 * ivor  : 0 for purely repulsive pair, 1 for attractive pair
 * dsq   : squared interatomic distance (Å^2)
 * rr2   : 1/dsq (Å^{{-2}}), precomputed by the pose loop
 *
 * Returns: pairwise LJ energy contribution (kcal/mol)
 *
 * To unlock gradient-capable kernels, also create lj_grad.h with the same
 * namespace and a 'lj_grad(rc, ac, emin, rmin2, ivor, dsq, rr2, dx, dy, dz,
 * energy, gx, gy, gz)' function (see PLAN_NONBONDED_FORCEFIELD.md §2.3).
 * IMPORTANT: gx/gy/gz must be force components −dE/d(lig_xyz) (force
 * direction, i.e. negative gradient).  The pose loop negates them before
 * accumulating into out_grad (gradient convention, +dE/dx).
 */
#ifndef {guard}
#define {guard}

namespace {name}
{{

    // TODO: implement lj_energy for {name}
    static inline double lj_energy(
        double rc, double ac, double emin, double rmin2,
        int ivor, double dsq, double rr2)
    {{
        (void)rc; (void)ac; (void)emin; (void)rmin2;
        (void)ivor; (void)dsq; (void)rr2;
        return 0.0; // TODO
    }}

}} // namespace {name}

#endif // {guard}
"""


def _elec_h_skeleton(name: str) -> str:
    guard = f"{name.upper()}_ELEC_H"
    return f"""\
/**
 * elec.h — Energy-only electrostatics for {name} (generated skeleton).
 *
 * Function signature (fixed contract; do not change):
 *
 *   double elec_energy(charge, rr2)
 *
 * Parameters
 * ----------
 * charge : product of the two prescaled atomic charges
 * rr2    : 1/dsq (Å^{{-2}}), precomputed by the pose loop
 *
 * Returns: pairwise electrostatic energy contribution (kcal/mol)
 *
 * To unlock gradient-capable kernels, also create elec_grad.h.
 */
#ifndef {guard}
#define {guard}

namespace {name}
{{

    // TODO: implement elec_energy for {name}
    static inline double elec_energy(double charge, double rr2)
    {{
        (void)charge; (void)rr2;
        return 0.0; // TODO
    }}

}} // namespace {name}

#endif // {guard}
"""


def _ff_yaml_skeleton(name: str) -> str:
    return f"""\
# {name} force field metadata.
# Consumed by codegen_ff.py and the Python framework.
name: {name}
description: "TODO: describe the {name} force field"

# C++ namespace used in the physics header files (lj.h, elec.h, ...).
# Change only if your headers use a different namespace than the FF name.
cpp_namespace: {name}

# Squared distance cutoff for neighbour list construction (Å^2).
# 'auto' = read from the grid file at runtime; or set a numeric value.
cutoff_sq: auto

# Whether the pose loop applies plateau correction.
plateau: true

# Dielectric model: false = rdie (distance-dependent), true = cdie (constant).
cdie: false

# Repulsive LJ exponent.
potshape: 8

# Rotation schemes to instantiate wrapper functions for.
# 'euler' is always available. 'rotvec' is planned for a later milestone.
supported_rotations:
  - euler

# Gradient support flags.  These are overridden at codegen time by inspecting
# whether lj_grad.h / elec_grad.h actually exist in this directory.
has_lj_grad: false
has_elec_grad: false
"""


def _init_py(name: str) -> str:
    return f'''\
"""
{name} — Python force field module ({name} skeleton generated by codegen_ff.py).

Follows the Section 5.1 force field module contract:
  - lj_energy(rc, ac, emin, rmin2, ivor, dsq, rr2) -> float
  - elec_energy(charge, rr2, dsq=None)              -> float
  - load_params(npz_path)                           -> FFParams

Gradient functions are optional.  If lj_grad.py and elec_grad.py are present,
they are imported automatically and exported as lj_grad / elec_grad.
Without them the module still loads; score_pairs will raise AttributeError
if gradient scoring is requested (use --autodiff-potentials instead).

Edit lj.py, elec.py, and params.py to implement the physics, then run:
  python codegen_ff.py codegen {name} <this_directory>
  make nb_kernel_{name}.so
"""

from .lj import lj_energy
from .elec import elec_energy
from .params import load_params, FFParams

try:
    from .lj_grad import lj_grad
except ImportError:
    lj_grad = None  # type: ignore[assignment]

try:
    from .elec_grad import elec_grad
except ImportError:
    elec_grad = None  # type: ignore[assignment]

__all__ = ["lj_energy", "elec_energy", "lj_grad", "elec_grad", "load_params", "FFParams"]
'''


def _lj_py(name: str) -> str:
    return f'''\
"""
lj.py — Energy-only LJ for {name} (generated skeleton).

Implement lj_energy as a JAX-traceable function.
Use jnp.where instead of Python if/else so JAX can trace and differentiate.

The C counterpart lives in lj.h; both must implement identical physics.
"""

import jax.numpy as jnp


def lj_energy(rc, ac, emin, rmin2, ivor, dsq, rr2):
    """LJ energy for a single atom pair.

    Parameters
    ----------
    rc     : repulsive coefficient
    ac     : attractive coefficient
    emin   : energy at the LJ minimum
    rmin2  : squared distance at the LJ minimum
    ivor   : 1 for attractive pairs, 0 for repulsive
    dsq    : squared interatomic distance
    rr2    : 1/dsq, precomputed

    Returns
    -------
    energy : float — pairwise LJ energy contribution
    """
    # TODO: implement {name} LJ energy
    return jnp.zeros_like(dsq)
'''


def _elec_py(name: str) -> str:
    return f'''\
"""
elec.py — Energy-only electrostatics for {name} (generated skeleton).

Implement elec_energy as a JAX-traceable function.
"""

import jax.numpy as jnp


def elec_energy(charge, rr2, dsq=None):
    """Electrostatic energy for a single atom pair.

    Parameters
    ----------
    charge : product of the two prescaled atomic charges
    rr2    : 1/dsq, precomputed
    dsq    : squared interatomic distance (optional, for API compatibility)

    Returns
    -------
    energy : float — pairwise electrostatic energy contribution
    """
    # TODO: implement {name} electrostatic energy
    return jnp.zeros_like(rr2)
'''


def _params_py(name: str) -> str:
    return f'''\
"""
params.py — Force field parameter loading for {name} (generated skeleton).

Follows the Section 5.3 parameter module contract:
    load_params(npz_path) -> FFParams
"""

from collections import namedtuple

import numpy as np

# Add or remove fields to match your force field\'s per-type-pair parameters.
FFParams = namedtuple("FFParams", ("rc", "ac", "ivor", "emin", "rmin2"))


def load_params(npz_path):
    """Load {name} force field parameters from an NPZ file.

    Parameters
    ----------
    npz_path : str or Path

    Returns
    -------
    FFParams
    """
    # TODO: adjust parameter loading and derived quantities for {name}
    par = np.load(npz_path)
    rc = par["rc"]
    ac = par["ac"]
    ivor = par["ivor"]
    emin = -27.0 * ac**4 / (256.0 * rc**3)
    rmin2 = 4.0 * rc / (3.0 * ac)
    return FFParams(
        rc=rc.astype(np.float64, copy=False),
        ac=ac.astype(np.float64, copy=False),
        ivor=ivor.astype(np.float64, copy=False),
        emin=emin.astype(np.float64, copy=False),
        rmin2=rmin2.astype(np.float64, copy=False),
    )
'''


# ---------------------------------------------------------------------------
# 'init' mode
# ---------------------------------------------------------------------------


def init_mode(name: str, ff_dir: Path) -> None:
    """Create a new force field skeleton under ff_dir."""
    ff_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "ff.yaml": _ff_yaml_skeleton(name),
        "lj.h": _lj_h_skeleton(name),
        "elec.h": _elec_h_skeleton(name),
        "__init__.py": _init_py(name),
        "lj.py": _lj_py(name),
        "elec.py": _elec_py(name),
        "params.py": _params_py(name),
    }

    for filename, content in files.items():
        out_path = ff_dir / filename
        if out_path.exists():
            print(f"  SKIP (exists): {out_path}")
        else:
            out_path.write_text(content)
            print(f"  CREATE: {out_path}")

    print(f"\nForce field '{name}' skeleton created in {ff_dir}.")
    print("Next steps:")
    print(f"  1. Implement lj_energy in {ff_dir}/lj.py  and {ff_dir}/lj.h")
    print(f"  2. Implement elec_energy in {ff_dir}/elec.py  and {ff_dir}/elec.h")
    print(f"  3. Implement load_params in {ff_dir}/params.py")
    print(f"  4. (Optional) Create lj_grad.h + elec_grad.h for gradient support")
    print(f"  5. Run: python codegen_ff.py codegen {name} {ff_dir}")
    print(f"  6. Run: make nb_kernel_{name}.so")


# ---------------------------------------------------------------------------
# 'codegen' mode — generate nb_kernel_<name>.cpp
# ---------------------------------------------------------------------------


def _ff_policy_struct(cap_name: str, namespace: str, has_grad: bool) -> str:
    """Generate the FFPolicy struct body."""
    lines = []
    lines.append(f"struct {cap_name}")
    lines.append("{")

    # lj_energy
    lines.append("  static inline double lj_energy(")
    lines.append("      double rc, double ac, double emin, double rmin2,")
    lines.append("      int ivor, double dsq, double rr2)")
    lines.append("  {")
    lines.append(
        f"    return {namespace}::lj_energy(rc, ac, emin, rmin2, ivor, dsq, rr2);"
    )
    lines.append("  }")
    lines.append("")

    # lj_grad (only when gradient headers exist)
    if has_grad:
        lines.append("  static inline void lj_grad(")
        lines.append("      double rc, double ac, double emin, double rmin2,")
        lines.append("      int ivor, double dsq, double rr2,")
        lines.append("      double dx, double dy, double dz,")
        lines.append("      double &energy, double &gx, double &gy, double &gz)")
        lines.append("  {")
        lines.append(f"    {namespace}::lj_grad(rc, ac, emin, rmin2, ivor, dsq, rr2,")
        lines.append("                    dx, dy, dz, energy, gx, gy, gz);")
        lines.append("  }")
        lines.append("")

    # elec_energy
    lines.append("  static inline double elec_energy(double charge, double rr2)")
    lines.append("  {")
    lines.append(f"    return {namespace}::elec_energy(charge, rr2);")
    lines.append("  }")
    lines.append("")

    # elec_grad (only when gradient headers exist)
    if has_grad:
        lines.append("  static inline void elec_grad(")
        lines.append("      double charge, double rr2,")
        lines.append("      double dx, double dy, double dz,")
        lines.append("      double &energy, double &gx, double &gy, double &gz)")
        lines.append("  {")
        lines.append(
            f"    {namespace}::elec_grad(charge, rr2, dx, dy, dz, energy, gx, gy, gz);"
        )
        lines.append("  }")
        lines.append("")

    lines.append("};")
    return "\n".join(lines)


def _grad_wrapper(
    name: str,
    rot: str,
    rot_policy: str,
    cap_name: str,
    plateau_mode: str,
    symbol: Optional[str] = None,
) -> str:
    """Generate the extern "C" energy+gradient wrapper."""
    sym = symbol or f"nb_kernel_{rot}_{plateau_mode}_grad"
    mode_cpp = PLATEAU_MODES[plateau_mode]
    return f"""\
/**
 * {sym} — energy + gradient, {rot} rotation, {name} force field, {plateau_mode} plateau mode.
 */
extern "C" int {sym}(
    const NbFusedStepData *step,
    const NbFusedGridData *grid,
    const NbGlobalData *global,
    const NbRunConfig *cfg,
    double *out_energy,
    double *out_grad)
{{
  const int rc = validate_grad_inputs(step, grid, global, cfg, out_energy, out_grad);
  if (rc != 0)
    return rc;

  std::memset(out_energy, 0, static_cast<size_t>(step->nposes) * sizeof(double));
  std::memset(out_grad, 0, static_cast<size_t>(step->nposes) * 6 * sizeof(double));

  const int32_t nthreads = cfg->num_threads > 0 ? cfg->num_threads : 1;
  omp_set_num_threads(nthreads);
  run_pose_loop_fused<{rot_policy}, {cap_name}, true, PlateauMode::{mode_cpp}>(
      step, grid, global, out_energy, out_grad);
  return 0;
}}
"""


def _energy_wrapper(
    name: str,
    rot: str,
    rot_policy: str,
    cap_name: str,
    plateau_mode: str,
    symbol: Optional[str] = None,
) -> str:
    """Generate the extern "C" energy-only wrapper."""
    sym = symbol or f"nb_kernel_{rot}_{plateau_mode}_energy"
    mode_cpp = PLATEAU_MODES[plateau_mode]
    return f"""\
/**
 * {sym} — energy only, {rot} rotation, {name} force field, {plateau_mode} plateau mode.
 * out_grad is accepted for ABI uniformity but not written to; may be NULL.
 */
extern "C" int {sym}(
    const NbFusedStepData *step,
    const NbFusedGridData *grid,
    const NbGlobalData *global,
    const NbRunConfig *cfg,
    double *out_energy,
    double *out_grad) /* not written; may be NULL */
{{
  const int rc = validate_energy_inputs(step, grid, global, cfg, out_energy);
  if (rc != 0)
    return rc;

  std::memset(out_energy, 0, static_cast<size_t>(step->nposes) * sizeof(double));

  const int32_t nthreads = cfg->num_threads > 0 ? cfg->num_threads : 1;
  omp_set_num_threads(nthreads);
  run_pose_loop_fused<{rot_policy}, {cap_name}, false, PlateauMode::{mode_cpp}>(
      step, grid, global, out_energy, nullptr);
  (void)out_grad;
  return 0;
}}
"""


def codegen_mode(name: str, ff_dir: Path) -> None:
    """Generate nb_kernel_<name>.cpp from the FF directory."""
    try:
        import yaml
    except ImportError:
        print(
            "ERROR: PyYAML is required for codegen mode.  Install with: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)

    ff_dir = ff_dir.resolve()
    yaml_path = ff_dir / "ff.yaml"
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(yaml_path) as fh:
        cfg = yaml.safe_load(fh)

    namespace: str = cfg.get("cpp_namespace", name)
    rotations: list = cfg.get("supported_rotations", ["euler"])

    # Inspect actual .h files (override yaml flags)
    has_lj_grad = (ff_dir / "lj_grad.h").exists()
    has_elec_grad = (ff_dir / "elec_grad.h").exists()
    has_grad = has_lj_grad and has_elec_grad

    # Compute include path prefix relative to the kernel root (SCRIPT_DIR).
    # The generated .cpp is compiled from SCRIPT_DIR, so includes must be
    # relative to SCRIPT_DIR.
    try:
        rel_ff_dir = ff_dir.relative_to(SCRIPT_DIR)
    except ValueError:
        # ff_dir is outside the kernel tree; use the absolute path.
        # The user may need to adjust -I flags manually.
        rel_ff_dir = ff_dir
    ff_prefix = str(rel_ff_dir).replace("\\", "/")

    # Capitalise FF name for the C++ struct identifier: nonbon8 → Nonbon8FF
    cap_name = name[0].upper() + name[1:] + "FF"

    parts: list[str] = []

    # --- header comment ---
    parts.append(
        f"""\
/**
 * nb_kernel_{name}.cpp — GENERATED by codegen_ff.py
 *
 * Force field : {name}
 * Namespace   : {namespace}
 * Has gradient: {'yes' if has_grad else 'no'}
 * Rotations   : {', '.join(rotations)}
 *
 * Regenerate with:
 *   python codegen_ff.py codegen {name} {ff_prefix}/
 *
 * Do not edit manually.
 */\
"""
    )

    # --- includes ---
    parts.append('#include "nb_kernel.h"\n')

    parts.append("// Rotation policies")
    for rot in rotations:
        if rot in ROT_HEADER:
            parts.append(f'#include "{ROT_HEADER[rot]}"')
    parts.append("")

    parts.append(f"// Force field headers: {name}")
    parts.append(f'#include "{ff_prefix}/lj.h"')
    if has_grad:
        parts.append(f'#include "{ff_prefix}/lj_grad.h"')
    parts.append(f'#include "{ff_prefix}/elec.h"')
    if has_grad:
        parts.append(f'#include "{ff_prefix}/elec_grad.h"')
    parts.append("")

    parts.append("// Pose loop template (must follow all physics + rotation headers)")
    parts.append('#include "include/pose_loop.h"\n')

    parts.append("#include <cstddef>")
    parts.append("#include <cstdint>")
    parts.append("#include <cstring>")
    parts.append("#include <omp.h>\n")

    # --- FFPolicy struct ---
    parts.append(
        "// ---------------------------------------------------------------------------"
    )
    parts.append(f"// FFPolicy: {name}")
    parts.append(f"// Delegates to pure physics functions in {namespace}::*")
    parts.append(f"// (generated from {ff_prefix}/lj.h, {ff_prefix}/elec.h, ...)")
    parts.append(
        "// ---------------------------------------------------------------------------"
    )
    parts.append(_ff_policy_struct(cap_name, namespace, has_grad))
    parts.append("")

    # --- validation helpers ---
    parts.append(_VALIDATION_SRC)

    # --- exported wrappers ---
    parts.append(
        "// ---------------------------------------------------------------------------"
    )
    parts.append("// Exported wrapper symbols — generated by codegen_ff.py")
    parts.append(f"// Naming convention: nb_kernel_<rot>_<plateau_mode>_<grad|energy>")
    parts.append(
        f"// Backward-compatible aliases: nb_kernel_<rot>_<grad|energy> (correction mode)"
    )
    parts.append(
        "// ---------------------------------------------------------------------------"
    )
    parts.append("")

    for rot in rotations:
        if rot not in ROT_POLICY:
            print(
                f"WARNING: rotation '{rot}' has no policy header mapping; skipping.",
                file=sys.stderr,
            )
            continue
        rot_policy = ROT_POLICY[rot]
        for plateau_mode in ("correction", "clamp"):
            if has_grad:
                parts.append(
                    _grad_wrapper(
                        name,
                        rot,
                        rot_policy,
                        cap_name,
                        plateau_mode=plateau_mode,
                    )
                )
            parts.append(
                _energy_wrapper(
                    name,
                    rot,
                    rot_policy,
                    cap_name,
                    plateau_mode=plateau_mode,
                )
            )

        # Keep unsuffixed symbols as correction-mode aliases for existing callers.
        if has_grad:
            parts.append(
                _grad_wrapper(
                    name,
                    rot,
                    rot_policy,
                    cap_name,
                    plateau_mode="correction",
                    symbol=f"nb_kernel_{rot}_grad",
                )
            )
        parts.append(
            _energy_wrapper(
                name,
                rot,
                rot_policy,
                cap_name,
                plateau_mode="correction",
                symbol=f"nb_kernel_{rot}_energy",
            )
        )

    out_path = ff_dir / f"nb_kernel_{name}.cpp"
    out_path.write_text("\n".join(parts) + "\n")
    print(f"Generated: {out_path}")
    print(f"  Namespace  : {namespace}")
    print(f"  Rotations  : {', '.join(rotations)}")
    print(f"  Has grad   : {has_grad}")
    print("  Plateau    : correction, clamp")
    if has_grad:
        print(
            "  Wrappers   : "
            "nb_kernel_euler_correction_grad, nb_kernel_euler_correction_energy, "
            "nb_kernel_euler_clamp_grad, nb_kernel_euler_clamp_energy, "
            "nb_kernel_euler_grad (alias), nb_kernel_euler_energy (alias)"
        )
    else:
        print(
            "  Wrappers   : "
            "nb_kernel_euler_correction_energy, nb_kernel_euler_clamp_energy, "
            "nb_kernel_euler_energy (alias)  (no gradient headers found)"
        )
    print(f"\nBuild with:  make nb_kernel_{name}.so")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Force field code generation for nb_kernel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python codegen_ff.py init    myff forcefields/myff/\n"
            "  python codegen_ff.py codegen myff forcefields/myff/\n"
        ),
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_init = sub.add_parser("init", help="Create a new force field skeleton")
    p_init.add_argument(
        "name", help="Force field name (used as namespace and file prefix)"
    )
    p_init.add_argument("directory", help="Directory to create the skeleton in")

    p_codegen = sub.add_parser(
        "codegen", help="Generate C++ wrapper code from an existing FF"
    )
    p_codegen.add_argument("name", help="Force field name")
    p_codegen.add_argument(
        "directory", help="Force field directory (must contain ff.yaml)"
    )

    args = parser.parse_args()
    ff_dir = Path(args.directory)

    if args.mode == "init":
        init_mode(args.name, ff_dir)
    elif args.mode == "codegen":
        codegen_mode(args.name, ff_dir)


if __name__ == "__main__":
    main()
