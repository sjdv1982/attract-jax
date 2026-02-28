#!/usr/bin/env python3
"""Minimizer benchmark: custom DFP/BFGS with legacy ATTRACT --score as oracle.

Implements a quasi-Newton minimizer matching the key parameters of legacy
minfor (Harwell VA13 variable-metric method):
- Initial Hessian: H_0 = diag_value * I  (default 0.01, matching minfor)
- Armijo sufficient decrease with c1 = 0.1
- Curvature condition threshold = 0.7
- Cubic interpolation backtracking
- Step extrapolation up to 9x current step
- DFP (or BFGS) Hessian update

Uses the compiled ATTRACT binary ($ATTRACTDIR) in --score mode as energy oracle.
"""

import argparse
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# DAT parsing (same as minimize_scipy.py)
# ---------------------------------------------------------------------------
STRUCT_RE = re.compile(r"^#\d+\s*$")
PIVOT_RE = re.compile(
    r"^#pivot\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$"
)
ENERGY_RE = re.compile(r"^\s*Energy:\s*([-+0-9.eE]+)\s*$")
GRAD_RE = re.compile(r"^\s*Gradients:\s*(.*?)\s*$")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")
DAT_ENERGY_RE = re.compile(r"^##\s*Energy:\s*([-+0-9.eE]+)\s*$")


def parse_dat_two_body(path: str, max_poses: int = 0):
    header: List[str] = []
    pivots: Dict[int, np.ndarray] = {}
    ens_list: List[int] = []
    dof_list: List[Tuple[float, ...]] = []
    energy_list: List[float] = []
    centered_ligands: Optional[bool] = None
    current_lines: List[List[float]] = []
    current_energy: Optional[float] = None
    seen_first_struct = False

    def flush():
        nonlocal current_lines, current_energy
        if not current_lines:
            return
        if len(current_lines) < 2:
            raise ValueError("Expected at least two numeric DOF lines per structure")
        first = current_lines[0]
        second = current_lines[-1]
        if len(first) != 7:
            raise ValueError(f"Expected 7 fields, got {len(first)}")
        ens = int(round(first[0]))
        if len(second) == 7:
            second = second[1:]
        if len(second) != 6:
            raise ValueError(f"Expected 6 DOF fields, got {len(second)}")
        ens_list.append(ens)
        dof_list.append(tuple(float(v) for v in second))
        energy_list.append(
            float("nan") if current_energy is None else float(current_energy)
        )
        current_lines = []
        current_energy = None

    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            pm = PIVOT_RE.match(line)
            if pm:
                pivots[int(pm.group(1))] = np.array(
                    [float(pm.group(2)), float(pm.group(3)), float(pm.group(4))],
                    dtype=np.float64,
                )
            em = DAT_ENERGY_RE.match(line)
            if em:
                current_energy = float(em.group(1))
                continue
            if STRUCT_RE.match(line):
                if max_poses and len(ens_list) >= max_poses:
                    break
                seen_first_struct = True
                flush()
                continue
            if not seen_first_struct:
                header.append(raw)
                low = line.strip().lower()
                if low.startswith("#centered ligands:"):
                    if "true" in low:
                        centered_ligands = True
                    elif "false" in low:
                        centered_ligands = False
            parts = line.strip().split()
            if not parts:
                continue
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) in (6, 7):
                current_lines.append(vals)

    if not (max_poses and len(ens_list) >= max_poses):
        flush()
    if not ens_list:
        raise ValueError(f"No structures parsed from {path}")
    return (
        header,
        pivots,
        np.asarray(ens_list, dtype=np.int32),
        np.asarray(dof_list, dtype=np.float64),
        np.asarray(energy_list, dtype=np.float64),
        centered_ligands,
    )


def write_dat_two_body(path, header, ens, dofs, energies=None):
    with open(path, "w") as f:
        for line in header:
            f.write(line)
        for i in range(len(dofs)):
            f.write(f"#{i+1}\n")
            if energies is not None and np.isfinite(energies[i]):
                f.write(f"## Energy: {energies[i]:.15e}\n")
            f.write(f"{int(ens[i]):12d}{0:12d}{0:12d}{0:12d}{0:12d}{0:12d}{0:12d}\n")
            phi, ssi, rot, xa, ya, za = dofs[i]
            f.write(
                f"{phi:24.16f} {ssi:24.16f} {rot:24.16f} "
                f"{xa:24.16f} {ya:24.16f} {za:24.16f}\n"
            )


def parse_score_output(text):
    energies = []
    grads = []
    for line in text.splitlines():
        m = ENERGY_RE.match(line)
        if m:
            energies.append(float(m.group(1)))
            continue
        g = GRAD_RE.match(line)
        if g:
            vals = [
                float(tok.replace("D", "E").replace("d", "e"))
                for tok in FLOAT_RE.findall(g.group(1))
            ]
            if len(vals) == 6:
                grads.append(vals)
    e = np.asarray(energies, dtype=np.float64)
    g = np.asarray(grads[: len(e)], dtype=np.float64)
    return e, g


def print_legacy_score(energies, gradients):
    """Print score blocks compatible with legacy ATTRACT --score output.

    gradients are expected as +dE/dx; legacy prints forces (-dE/dx).
    """
    for e, g in zip(np.asarray(energies), np.asarray(gradients)):
        f = -np.asarray(g, dtype=np.float64)
        print(f" Energy: {float(e): .16f}     ")
        print(
            f"{float(e):12.3f}{0.0:12.3f}{0.0:12.3f}"
            f"{0.0:12.3f}{0.0:12.3f}{0.0:12.3f}"
        )
        print(" Gradients:" + "".join(f"{float(v):24.16E}" for v in f))


# ---------------------------------------------------------------------------
# Legacy ATTRACT --score oracle
# ---------------------------------------------------------------------------
class LegacyScoreOracle:
    def __init__(
        self,
        attract_bin,
        attract_par,
        receptor_pdb,
        ligand_pdb,
        ens_list,
        grid_header,
        header,
        tmpdir,
        cwd=".",
    ):
        self.attract_bin = os.path.abspath(attract_bin)
        self.attract_par = os.path.abspath(attract_par)
        self.receptor_pdb = os.path.abspath(receptor_pdb)
        self.ligand_pdb = os.path.abspath(ligand_pdb)
        self.ens_list = os.path.abspath(ens_list)
        self.grid_header = os.path.abspath(grid_header)
        self.header = header
        self.tmpdir = os.path.abspath(tmpdir)
        self.cwd = os.path.abspath(cwd)
        self._call_count = 0

    def _run_score(self, dat_path, expected_n):
        cmd = [
            self.attract_bin,
            dat_path,
            self.attract_par,
            self.receptor_pdb,
            self.ligand_pdb,
            "--fix-receptor",
            "--ens",
            "1",
            self.ens_list,
            "--grid",
            "1",
            self.grid_header,
            "--score",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=self.cwd
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"attract --score failed (rc={result.returncode}):\n{result.stderr}"
            )
        e, g = parse_score_output(result.stdout)
        if len(e) != expected_n:
            raise ValueError(f"Expected {expected_n} energies, got {len(e)}")
        return e, g

    def score_single(self, ens_id, dof):
        self._call_count += 1
        dat_path = os.path.join(self.tmpdir, f"_s{self._call_count}.dat")
        write_dat_two_body(
            dat_path, self.header, np.array([ens_id], np.int32), dof.reshape(1, 6)
        )
        e, g = self._run_score(dat_path, 1)
        os.unlink(dat_path)
        # ATTRACT --score prints forces (−∂E/∂x); negate to get gradient (+∂E/∂x)
        return float(e[0]), -g[0]

    def score_batch(self, ens, dofs):
        self._call_count += 1
        dat_path = os.path.join(self.tmpdir, f"_s{self._call_count}.dat")
        write_dat_two_body(dat_path, self.header, ens, dofs)
        e, g = self._run_score(dat_path, len(dofs))
        os.unlink(dat_path)
        # ATTRACT --score prints forces (−∂E/∂x); negate to get gradient (+∂E/∂x)
        return e, -g


# ---------------------------------------------------------------------------
# Packed LDL^T Hessian routines (from VA13)
# ---------------------------------------------------------------------------
def mc11e_packed(a_pack, z, w, ir, n=6):
    """Solve B*d = z where B is stored as packed LDL^T."""
    if ir < n:
        return z.copy(), w.copy()

    a = np.zeros(len(a_pack) + 1, dtype=np.float64)
    a[1:] = a_pack
    zf = np.zeros(n + 1, dtype=np.float64)
    zf[1:] = z
    wf = np.zeros(n + 1, dtype=np.float64)
    wf[1:] = w

    wf[1] = zf[1]
    if n <= 1:
        zf[1] = zf[1] / a[1]
        return zf[1:], wf[1:]

    ij = 1
    for i in range(2, n + 1):
        ij = i
        v = zf[i]
        for j in range(1, i):
            v = v - a[ij] * zf[j]
            ij = ij + n - j
        wf[i] = v
        zf[i] = v

    zf[n] = zf[n] / a[ij]
    np1 = n + 1
    for nip in range(2, n + 1):
        i = np1 - nip
        ii = ij - nip
        v = zf[i] / a[ii]
        ip = i + 1
        ij = ii
        for j in range(ip, n + 1):
            ii = ii + 1
            v = v - a[ii] * zf[j]
        zf[i] = v

    return zf[1:], wf[1:]


def mc11a_packed(a_pack, z, sig, w, ir, mk, eps, n=6, alias_zw=False):
    """Rank-1 update of packed LDL^T factorization: B_new = B + sig*z*z^T."""
    a = np.zeros(len(a_pack) + 1, dtype=np.float64)
    a[1:] = a_pack
    zf = np.zeros(n + 1, dtype=np.float64)
    zf[1:] = z
    if alias_zw:
        wf = zf
    else:
        wf = np.zeros(n + 1, dtype=np.float64)
        wf[1:] = w

    if n <= 1:
        a[1] = a[1] + sig * zf[1] ** 2
        ir = 1
        if a[1] <= 0.0:
            a[1] = 0.0
            ir = 0
        zout = zf[1:].copy()
        wout = zout.copy() if alias_zw else wf[1:].copy()
        return a[1:].copy(), zout, wout, ir

    np1 = n + 1
    if sig > 0.0:
        mm = 0
        tim = 1.0 / sig
    else:
        if sig == 0.0 or ir == 0:
            zout = zf[1:].copy()
            wout = zout.copy() if alias_zw else wf[1:].copy()
            return a[1:].copy(), zout, wout, ir

        ti = 1.0 / sig
        ij = 1
        if mk != 0:
            for i in range(1, n + 1):
                if a[ij] != 0.0:
                    ti = ti + wf[i] ** 2 / a[ij]
                ij = ij + np1 - i
        else:
            for i in range(1, n + 1):
                wf[i] = zf[i]
            for i in range(1, n + 1):
                ip = i + 1
                v = wf[i]
                if a[ij] <= 0.0:
                    wf[i] = 0.0
                    ij = ij + np1 - i
                    continue
                ti = ti + v * v / a[ij]
                if i != n:
                    for j in range(ip, n + 1):
                        ij = ij + 1
                        wf[j] = wf[j] - v * a[ij]
                ij = ij + 1

        goto40 = False
        if ir > 0:
            if ti > 0.0:
                ti = eps / sig
                if eps == 0.0:
                    ir = ir - 1
            else:
                if mk <= 1:
                    goto40 = True
                else:
                    ti = 0.0
                    ir = -ir - 1
        else:
            ti = 0.0
            ir = -ir - 1

        if goto40:
            mm = 0
            tim = 1.0 / sig
        else:
            mm = 1
            tim = ti
            for i in range(1, n + 1):
                j = np1 - i
                ij = ij - i
                if a[ij] != 0.0:
                    tim = ti - wf[j] ** 2 / a[ij]
                wf[j] = ti
                ti = tim

    ij = 1
    for i in range(1, n + 1):
        ip = i + 1
        v = zf[i]
        if a[ij] <= 0.0:
            if ir > 0 or sig < 0.0 or v == 0.0:
                ti = tim
                ij = ij + np1 - i
                continue
            ir = 1 - ir
            a[ij] = v * v / tim
            if i == n:
                break
            for j in range(ip, n + 1):
                ij = ij + 1
                a[ij] = zf[j] / v
            break

        al = v / a[ij]
        if mm != 0:
            ti = wf[i]
        else:
            ti = tim + v * al
        r = ti / tim
        a[ij] = a[ij] * r
        if r == 0.0 or i == n:
            break

        b = al / ti
        if r <= 4.0:
            for j in range(ip, n + 1):
                ij = ij + 1
                zf[j] = zf[j] - v * a[ij]
                a[ij] = a[ij] + b * zf[j]
        else:
            gm = tim / ti
            for j in range(ip, n + 1):
                ij = ij + 1
                yv = a[ij]
                a[ij] = b * zf[j] + yv * gm
                zf[j] = zf[j] - v * yv
        tim = ti
        ij = ij + 1

    if ir < 0:
        ir = -ir
    zout = zf[1:].copy()
    wout = zout.copy() if alias_zw else wf[1:].copy()
    return a[1:].copy(), zout, wout, ir


def init_hpack_diag(n=6, diag_value=0.01):
    """Initialize packed LDL^T as diag_value * I."""
    size = n * (n + 1) // 2
    h = np.zeros(size, dtype=np.float64)
    k = 0
    for i in range(n):
        h[k] = diag_value
        k += n - i
    return h


# ---------------------------------------------------------------------------
# Custom minimizer — faithful VA13 reimplementation
# ---------------------------------------------------------------------------
def minfor_minimize(
    oracle,
    ens_id,
    x0,
    maxfun=150,
    init_metric=0.01,
    acc=1e-9,
):
    """Minimize a single pose using VA13 variable-metric method.

    This is a faithful reimplementation of the Harwell VA13 algorithm used
    by legacy ATTRACT's minfor, using the same packed LDL^T Hessian
    representation and identical line search logic.

    The outer loop has two entry points matching the Fortran labels:
      - Label 110: reset xaa/fa/ga from the global best (first call +
        after failed line search with isfv >= 2).
      - Label 135: compute next search direction from the *current* xaa/fa/ga
        (after a successful Hessian update — does NOT reset to global best).

    Parameters
    ----------
    oracle : LegacyScoreOracle
    ens_id : int
    x0 : ndarray (6,)
    maxfun : int
    init_metric : float
        Initial diagonal Hessian (minfor default: 0.01)
    acc : float
        Convergence tolerance (minfor default: 1e-9)

    Returns
    -------
    x_best, f_best, nfev
    """
    n = len(x0)

    # Packed Hessian B (LDL^T factorization)
    hpack = init_hpack_diag(n, init_metric)
    w = np.zeros(n, dtype=np.float64)
    ir = n  # rank indicator (starts full rank)

    # Current best point
    x = x0.copy()
    gesa, g = oracle.score_single(ens_id, x)
    nfev = 1
    f_best = gesa
    x_best = x.copy()
    g_best = g.copy()

    dff = 0.0
    # State variables that persist across outer iterations:
    xaa = x.copy()
    fa = gesa
    ga = g.copy()
    isfv = 1
    entry_label = 110  # first entry is always label 110

    while nfev < maxfun:
        if entry_label == 110:
            # --- Label 110: reset base to global best ---
            xaa = x.copy()
            fa = gesa
            isfv = 1
            ga = g.copy()
        # else entry_label == 135: keep current xaa/fa/ga from Hessian update

        # --- Label 135: compute search direction ---
        d_in = -ga
        d, w = mc11e_packed(hpack, d_in, w, ir, n)

        cmax = float(np.max(np.abs(d)))
        dga = float(np.dot(ga, d))

        if cmax <= 0.0 or dga >= 0.0:
            # Fortran: go to 240 → isfv is 1 after label 110, so terminate
            break

        stmin = 0.0
        stepbd = 0.0
        steplb = acc / cmax
        fmin = fa
        gmin = dga

        step = 1.0
        if dff <= 0.0:
            step = min(step, 1.0 / cmax)
        else:
            step = min(step, 2.0 * dff / (-dga))

        # --- Line search ---
        entry_label = 0  # will be set before next outer iteration
        while nfev < maxfun:
            nfev += 1
            c = stmin + step
            xbb = xaa + c * d
            fb, gb = oracle.score_single(ens_id, xbb)

            # Track global best (isfv logic)
            isfv = min(2, isfv)
            if fb <= gesa:
                better = fb < gesa
                if not better:
                    gl1 = float(np.dot(g, g))
                    gl2 = float(np.dot(gb, gb))
                    better = gl2 < gl1
                if better:
                    isfv = 3
                    gesa = fb
                    x = xbb.copy()
                    g = gb.copy()
                    if fb < f_best:
                        f_best = fb
                        x_best = xbb.copy()
                        g_best = gb.copy()

            dgb = float(np.dot(gb, d))

            # Armijo sufficient decrease test
            if fb - fa <= 0.1 * c * dga:
                # --- Extrapolation ---
                stepbd -= step
                stmin = c
                fmin = fb
                gmin = dgb

                step = 9.0 * stmin
                if stepbd > 0.0:
                    step = 0.5 * stepbd

                ctmp = dga + 3.0 * dgb - 4.0 * (fb - fa) / stmin
                if ctmp > 0.0:
                    step = min(step, stmin * max(1.0, -dgb / ctmp))

                if dgb < 0.7 * dga:
                    # Curvature not satisfied — continue line search
                    if stmin + step <= steplb:
                        # Step too small; Fortran: go to 240
                        if isfv >= 2:
                            entry_label = 110  # restart from global best
                        break
                    continue

                # Curvature satisfied
                isfv = 4 - isfv
                if stmin + step <= steplb:
                    # Fortran: go to 240
                    if isfv >= 2:
                        entry_label = 110
                    break

                # VA13 two-step mc11a Hessian update
                ga_old = ga.copy()
                y = gb - ga

                denom1 = dga
                denom2 = stmin * (dgb - dga)
                if abs(denom1) < 1e-16 or abs(denom2) < 1e-16:
                    break

                h1, _, w1, ir1 = mc11a_packed(
                    hpack,
                    ga_old,
                    1.0 / denom1,
                    w,
                    -n,
                    1,
                    0.0,
                    n=n,
                )
                h2, _, _, ir2 = mc11a_packed(
                    h1,
                    y,
                    1.0 / denom2,
                    y.copy(),
                    -ir1,
                    0,
                    0.0,
                    n=n,
                    alias_zw=True,
                )
                if ir2 < n:
                    break  # rank deficient — abort

                hpack = h2
                w = w1
                ir = ir2

                # Update base point to accepted step (Fortran label 280)
                dff = fa - fb
                fa = fb
                xaa = xbb.copy()
                ga = gb.copy()
                entry_label = 135  # → label 135: next direction, NO reset
                break

            else:
                # --- Backtracking ---
                if step > steplb:
                    stepbd = step
                    ctmp = gmin + dgb - 3.0 * (fb - fmin) / step
                    disc = ctmp * ctmp - gmin * dgb
                    if disc < 0.0:
                        disc = 0.0
                    denom = ctmp + gmin - math.sqrt(disc)
                    if abs(denom) < 1e-16:
                        fac = 0.1
                    else:
                        fac = max(0.1, gmin / denom)
                    step = step * fac
                    continue
                else:
                    # Step too small; Fortran: go to 240
                    if isfv >= 2:
                        entry_label = 110
                    break

        if entry_label == 135:
            # Successful Hessian update → continue to next direction
            continue
        if entry_label == 110:
            # Failed line search but had improvement → restart from best
            continue
        # entry_label == 0: no progress, terminate
        break

    return x_best, f_best, nfev


# ---------------------------------------------------------------------------
# Batched VA13 minimizer — one oracle call per tick for all active poses
# ---------------------------------------------------------------------------
def minfor_minimize_batched(
    oracle,
    ens,
    dofs0,
    maxfun=150,
    init_metric=0.01,
    acc=1e-9,
    trace_every=0,
    traj_prefix=None,
    traj_header=None,
):
    """Batch-minimize all poses using VA13 with batched oracle calls.

    Identical minimization logic to minfor_minimize, but collects all active
    poses' trial points into one oracle.score_batch call per tick.  The
    per-pose state update (Hessian, line search decisions) is done in a
    Python loop but is negligible compared to the energy evaluation.

    Returns
    -------
    x_best : (N, 6)
    f_best : (N,)
    nfev   : (N,) int
    """
    N, n = dofs0.shape

    # --- Per-pose state arrays ---
    hpack = np.tile(init_hpack_diag(n, init_metric), (N, 1))  # (N, 21)
    w_arr = np.zeros((N, n), dtype=np.float64)
    ir_arr = np.full(N, n, dtype=np.int32)

    x = dofs0.copy()  # current best position
    g = np.zeros((N, n), dtype=np.float64)  # gradient at x
    gesa = np.full(N, np.inf, dtype=np.float64)  # energy at x
    x_best = dofs0.copy()  # global best position
    f_best = np.full(N, np.inf, dtype=np.float64)  # global best energy

    xaa = np.zeros((N, n), dtype=np.float64)  # line search base point
    d_arr = np.zeros((N, n), dtype=np.float64)  # search direction
    fa_arr = np.full(N, np.inf, dtype=np.float64)  # energy at xaa
    ga_arr = np.zeros((N, n), dtype=np.float64)  # gradient at xaa
    dga_arr = np.zeros(N, dtype=np.float64)  # directional deriv
    stmin_arr = np.zeros(N, dtype=np.float64)
    stepbd_arr = np.zeros(N, dtype=np.float64)
    steplb_arr = np.zeros(N, dtype=np.float64)
    fmin_arr = np.zeros(N, dtype=np.float64)
    gmin_arr = np.zeros(N, dtype=np.float64)
    step_arr = np.zeros(N, dtype=np.float64)
    dff = np.zeros(N, dtype=np.float64)
    isfv = np.ones(N, dtype=np.int32)
    nfev = np.zeros(N, dtype=np.int32)
    xbb = dofs0.copy()  # next trial point
    active = np.ones(N, dtype=bool)

    # --- Trajectory state: track last evaluated (dofs, energy) per pose ---
    traj_dofs = dofs0.copy()  # last evaluated point per pose
    traj_energies = np.full(N, np.nan, dtype=np.float64)

    # --- Initial batch evaluation ---
    e0, g0 = oracle.score_batch(ens, dofs0)
    nfev[:] = 1
    gesa[:] = e0
    g[:] = g0
    f_best[:] = e0
    x_best[:] = dofs0
    traj_energies[:] = e0

    if traj_prefix and traj_header is not None:
        write_dat_two_body(
            f"{traj_prefix}.{0:04d}.dat", traj_header, ens, traj_dofs, traj_energies
        )

    # --- Set up first outer iteration for all poses ---
    for i in range(N):
        xaa[i] = x[i]
        fa_arr[i] = gesa[i]
        isfv[i] = 1
        ga_arr[i] = g[i]

        d_arr[i], w_arr[i] = mc11e_packed(
            hpack[i], -ga_arr[i], w_arr[i], int(ir_arr[i]), n
        )
        cmax = float(np.max(np.abs(d_arr[i])))
        dga_arr[i] = float(np.dot(ga_arr[i], d_arr[i]))

        if cmax <= 0.0 or dga_arr[i] >= 0.0:
            active[i] = False
            continue

        stmin_arr[i] = 0.0
        stepbd_arr[i] = 0.0
        steplb_arr[i] = acc / cmax
        fmin_arr[i] = fa_arr[i]
        gmin_arr[i] = dga_arr[i]

        stp = 1.0
        if dff[i] <= 0.0:
            stp = min(stp, 1.0 / cmax)
        else:
            stp = min(stp, 2.0 * dff[i] / (-dga_arr[i]))
        step_arr[i] = stp

        c = stmin_arr[i] + step_arr[i]
        xbb[i] = xaa[i] + c * d_arr[i]

    tick = 0
    t_batch = time.time()
    _cum_kernel = 0.0
    _cum_python = 0.0

    # --- Main batch loop ---
    while active.any():
        # Budget check — exclude poses that hit maxfun
        over = active & (nfev >= maxfun)
        active[over] = False
        act_idx = np.where(active)[0]
        if len(act_idx) == 0:
            break

        # One batch oracle call for all active poses
        _t0 = time.time()
        e_batch, g_batch = oracle.score_batch(ens[act_idx], xbb[act_idx])
        _t1 = time.time()
        _cum_kernel += _t1 - _t0
        nfev[act_idx] += 1
        tick += 1

        # Update trajectory state for active poses
        if traj_prefix and traj_header is not None:
            for k, ii in enumerate(act_idx):
                traj_dofs[ii] = xbb[ii]
                traj_energies[ii] = float(e_batch[k])
            write_dat_two_body(
                f"{traj_prefix}.{tick:04d}.dat",
                traj_header,
                ens,
                traj_dofs,
                traj_energies,
            )

        # Process each active pose's line search result
        need_label_110 = []  # failed line search → reset to global best
        need_label_135 = []  # successful Hessian update → continue from current
        for k, ii in enumerate(act_idx):
            i = int(ii)
            fb = float(e_batch[k])
            gb = g_batch[k]

            # --- isfv / best-tracking (identical to per-pose code) ---
            isfv[i] = min(2, isfv[i])
            if fb <= gesa[i]:
                better = fb < gesa[i]
                if not better:
                    gl1 = float(np.dot(g[i], g[i]))
                    gl2 = float(np.dot(gb, gb))
                    better = gl2 < gl1
                if better:
                    isfv[i] = 3
                    gesa[i] = fb
                    x[i] = xbb[i].copy()
                    g[i] = gb.copy()
                    if fb < f_best[i]:
                        f_best[i] = fb
                        x_best[i] = xbb[i].copy()

            c = stmin_arr[i] + step_arr[i]
            dgb = float(np.dot(gb, d_arr[i]))

            # --- Armijo test ---
            if fb - fa_arr[i] <= 0.1 * c * dga_arr[i]:
                # Sufficient decrease — extrapolation phase
                stepbd_arr[i] -= step_arr[i]
                stmin_arr[i] = c
                fmin_arr[i] = fb
                gmin_arr[i] = dgb

                new_step = 9.0 * stmin_arr[i]
                if stepbd_arr[i] > 0.0:
                    new_step = 0.5 * stepbd_arr[i]

                ctmp = dga_arr[i] + 3.0 * dgb - 4.0 * (fb - fa_arr[i]) / stmin_arr[i]
                if ctmp > 0.0:
                    new_step = min(new_step, stmin_arr[i] * max(1.0, -dgb / ctmp))

                if dgb < 0.7 * dga_arr[i]:
                    # Curvature NOT satisfied — continue or break
                    if stmin_arr[i] + new_step <= steplb_arr[i]:
                        # Step too small
                        if gesa[i] < fa_arr[i] or isfv[i] >= 2:
                            need_label_110.append(i)
                        else:
                            active[i] = False
                    else:
                        step_arr[i] = new_step
                        xbb[i] = xaa[i] + (stmin_arr[i] + new_step) * d_arr[i]
                    continue

                # Curvature satisfied
                isfv[i] = 4 - isfv[i]
                if stmin_arr[i] + new_step <= steplb_arr[i]:
                    if gesa[i] < fa_arr[i] or isfv[i] >= 2:
                        need_label_110.append(i)
                    else:
                        active[i] = False
                    continue

                # --- Hessian update ---
                ga_old = ga_arr[i].copy()
                y = gb - ga_arr[i]
                denom1 = dga_arr[i]
                denom2 = stmin_arr[i] * (dgb - dga_arr[i])

                if abs(denom1) < 1e-16 or abs(denom2) < 1e-16:
                    if gesa[i] < fa_arr[i] or isfv[i] >= 2:
                        need_label_110.append(i)
                    else:
                        active[i] = False
                    continue

                h1, _, w1, ir1 = mc11a_packed(
                    hpack[i],
                    ga_old,
                    1.0 / denom1,
                    w_arr[i],
                    -n,
                    1,
                    0.0,
                    n=n,
                )
                h2, _, _, ir2 = mc11a_packed(
                    h1,
                    y,
                    1.0 / denom2,
                    y.copy(),
                    -ir1,
                    0,
                    0.0,
                    n=n,
                    alias_zw=True,
                )

                if ir2 < n:
                    # Rank deficient
                    if gesa[i] < fa_arr[i] or isfv[i] >= 2:
                        need_label_110.append(i)
                    else:
                        active[i] = False
                    continue

                # Success — accept Hessian update
                hpack[i] = h2
                w_arr[i] = w1
                ir_arr[i] = ir2
                dff[i] = fa_arr[i] - fb
                # Update base to accepted step (Fortran label 280 → 135)
                fa_arr[i] = fb
                xaa[i] = xbb[i].copy()
                ga_arr[i] = gb.copy()
                need_label_135.append(i)

            else:
                # Insufficient decrease — backtracking
                if step_arr[i] > steplb_arr[i]:
                    stepbd_arr[i] = step_arr[i]
                    ctmp = gmin_arr[i] + dgb - 3.0 * (fb - fmin_arr[i]) / step_arr[i]
                    disc = ctmp * ctmp - gmin_arr[i] * dgb
                    if disc < 0.0:
                        disc = 0.0
                    denom = ctmp + gmin_arr[i] - math.sqrt(disc)
                    if abs(denom) < 1e-16:
                        fac = 0.1
                    else:
                        fac = max(0.1, gmin_arr[i] / denom)
                    step_arr[i] *= fac
                    xbb[i] = xaa[i] + (stmin_arr[i] + step_arr[i]) * d_arr[i]
                else:
                    # Step too small
                    if gesa[i] < fa_arr[i] or isfv[i] >= 2:
                        need_label_110.append(i)
                    else:
                        active[i] = False

        # --- Label 135: compute next direction from current state ---
        # (after successful Hessian update — do NOT reset to global best)
        for i in need_label_135:
            if not active[i]:
                continue

            d_arr[i], w_arr[i] = mc11e_packed(
                hpack[i], -ga_arr[i], w_arr[i], int(ir_arr[i]), n
            )
            cmax = float(np.max(np.abs(d_arr[i])))
            dga_arr[i] = float(np.dot(ga_arr[i], d_arr[i]))

            if cmax <= 0.0 or dga_arr[i] >= 0.0:
                active[i] = False
                continue

            stmin_arr[i] = 0.0
            stepbd_arr[i] = 0.0
            steplb_arr[i] = acc / cmax
            fmin_arr[i] = fa_arr[i]
            gmin_arr[i] = dga_arr[i]

            stp = 1.0
            if dff[i] <= 0.0:
                stp = min(stp, 1.0 / cmax)
            else:
                stp = min(stp, 2.0 * dff[i] / (-dga_arr[i]))
            step_arr[i] = stp

            c = stmin_arr[i] + step_arr[i]
            xbb[i] = xaa[i] + c * d_arr[i]

        # --- Label 110: reset to global best, then compute direction ---
        # (after failed line search with isfv >= 2)
        for i in need_label_110:
            if not active[i]:
                continue

            xaa[i] = x[i]
            fa_arr[i] = gesa[i]
            isfv[i] = 1
            ga_arr[i] = g[i]

            d_arr[i], w_arr[i] = mc11e_packed(
                hpack[i], -ga_arr[i], w_arr[i], int(ir_arr[i]), n
            )
            cmax = float(np.max(np.abs(d_arr[i])))
            dga_arr[i] = float(np.dot(ga_arr[i], d_arr[i]))

            if cmax <= 0.0 or dga_arr[i] >= 0.0:
                active[i] = False
                continue

            stmin_arr[i] = 0.0
            stepbd_arr[i] = 0.0
            steplb_arr[i] = acc / cmax
            fmin_arr[i] = fa_arr[i]
            gmin_arr[i] = dga_arr[i]

            stp = 1.0
            if dff[i] <= 0.0:
                stp = min(stp, 1.0 / cmax)
            else:
                stp = min(stp, 2.0 * dff[i] / (-dga_arr[i]))
            step_arr[i] = stp

            c = stmin_arr[i] + step_arr[i]
            xbb[i] = xaa[i] + c * d_arr[i]

        _cum_python += time.time() - _t1

        # --- Trace ---
        if trace_every and tick % trace_every == 0:
            n_active = int(active.sum())
            elapsed = time.time() - t_batch
            print(
                f"  tick {tick}: {n_active}/{N} active, "
                f"mean nfev={nfev[nfev > 0].mean():.0f}, "
                f"best energy={f_best.min():.3f} ({elapsed:.1f}s) "
                f"[kernel={_cum_kernel:.1f}s python={_cum_python:.1f}s]"
            )

    return x_best, f_best, nfev


# ---------------------------------------------------------------------------
# 4x4 matrix conversion
# ---------------------------------------------------------------------------
def euler2rotmat_np(phi, ssi, rot):
    cs, cp = np.cos(ssi), np.cos(phi)
    ss, sp = np.sin(ssi), np.sin(phi)
    crot, srot = np.cos(rot), np.sin(rot)
    out = np.zeros((len(phi), 3, 3), dtype=np.float64)
    out[:, 0, 0] = crot * cs * cp + srot * sp
    out[:, 0, 1] = srot * cs * cp - crot * sp
    out[:, 0, 2] = ss * cp
    out[:, 1, 0] = crot * cs * sp - srot * cp
    out[:, 1, 1] = srot * cs * sp + crot * cp
    out[:, 1, 2] = ss * sp
    out[:, 2, 0] = -crot * ss
    out[:, 2, 1] = -srot * ss
    out[:, 2, 2] = cs
    return out


def dofs_to_mats_np(dofs, pivot):
    rot_col = euler2rotmat_np(dofs[:, 0], dofs[:, 1], dofs[:, 2])
    rot_row = np.swapaxes(rot_col, 1, 2)
    pivot_rot = np.einsum("j,bji->bi", pivot, rot_row)
    trans = dofs[:, 3:6] + pivot[None, :] - pivot_rot
    mats = np.zeros((len(dofs), 4, 4), dtype=np.float64)
    mats[:, :3, :3] = rot_row
    mats[:, 3, :3] = trans
    mats[:, 3, 3] = 1.0
    return mats


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def summarize(name, ref, cand):
    d = cand - ref
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    corr = float(np.corrcoef(ref, cand)[0, 1]) if len(ref) > 1 else float("nan")
    print(
        f"{name}: mae={mae:.4f} rmse={rmse:.4f} pearson={corr:.4f} "
        f"delta_p50={np.percentile(d, 50):.4f}"
    )


def resolve_attract_paths(test_dir):
    attractdir = os.environ.get("ATTRACTDIR", "")
    if not attractdir:
        raise RuntimeError("$ATTRACTDIR is not set")
    attract_bin = os.path.join(attractdir, "attract")
    attract_par = os.path.join(attractdir, "..", "attract.par")
    test_dir = os.path.abspath(test_dir)
    return {
        "attract_bin": attract_bin,
        "attract_par": attract_par,
        "receptor_pdb": os.path.join(test_dir, "partner1-ensemble", "model-1r.pdb"),
        "ligand_pdb": os.path.join(test_dir, "ligandr.pdb"),
        "ens_list": os.path.join(test_dir, "partner1-ensemble.list"),
        "grid_header": os.path.join(test_dir, "receptorgrid.gridheader"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Custom DFP minimizer with legacy ATTRACT --score oracle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input_dat", help="starting .dat file")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument(
        "--score",
        action="store_true",
        help="score-only mode: print legacy-style energy/gradient blocks",
    )
    ap.add_argument("--maxfun", type=int, default=150)
    ap.add_argument("--max-poses", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--pose-offset",
        type=int,
        default=0,
        help="skip this many poses from the start (for parallel chunking)",
    )
    ap.add_argument("--trace-every", type=int, default=10)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--legacy-dat", help="legacy minimized .dat for comparison")
    ap.add_argument(
        "--init-metric",
        type=float,
        default=0.01,
        help="initial diagonal Hessian (minfor default: 0.01)",
    )
    ap.add_argument(
        "--method",
        default="dfp-batch",
        choices=["dfp", "dfp-batch", "scipy-lbfgsb", "scipy-bfgs", "scipy-cg"],
        help="minimizer method (dfp-batch is batched, much faster)",
    )
    # --- JAX oracle options ---
    ap.add_argument(
        "--oracle",
        default="legacy",
        choices=["legacy", "jax"],
        help="energy oracle: 'legacy' (ATTRACT binary) or 'jax' (ATTRACT-JAX)",
    )
    ap.add_argument(
        "--grid", default=None, help="path to .grid file (required for --oracle jax)"
    )
    ap.add_argument("--attract-par-npz", default=None, help="path to attract-par.npz")
    ap.add_argument(
        "--receptor-ens-list", default=None, help="path to receptor ensemble list"
    )
    ap.add_argument("--ligand-pdb", default=None, help="path to ligand PDB")
    ap.add_argument("--epsilon", type=float, default=15.0, help="dielectric constant")
    ap.add_argument("--cdie", action="store_true", help="distance-dependent dielectric")
    ap.add_argument(
        "--energy-batch",
        type=int,
        default=256,
        help="max poses per JAX kernel call (merged-ensemble: all ensembles in one call)",
    )
    ap.add_argument(
        "--max-nb-cap",
        type=int,
        default=0,
        help="cap max_nr_neighbours for JAX oracle (0=no cap, e.g. 40 for speed)",
    )
    ap.add_argument(
        "--nb-mode",
        choices=["fixed", "bucketed"],
        default="fixed",
        help="JAX neighbour kernel mode",
    )
    ap.add_argument(
        "--nb-bucket-thresholds",
        default="8",
        help=(
            "bucket thresholds for --nb-mode bucketed. "
            "Single integer = step size (default 8); "
            "or comma-separated explicit thresholds, e.g. 8,16,24,32,64,96,128,180"
        ),
    )
    ap.add_argument(
        "--disable-jit",
        action="store_true",
        help="disable JAX JIT compilation (slower per-eval but no compilation time)",
    )
    ap.add_argument(
        "--traj",
        action="store_true",
        help="write trajectory .dat files (one per tick: {out-prefix}.traj.NNNN.dat)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    verbose = not args.score
    if not args.score and not args.out_prefix:
        raise ValueError("--out-prefix is required unless --score is used")

    if args.disable_jit and args.oracle == "jax":
        import jax

        jax.config.update("jax_disable_jit", True)
        if verbose:
            print("JAX JIT disabled (--disable-jit)")

    test_dir = args.test_dir or str(Path(args.input_dat).resolve().parent)

    # Read poses (max_poses includes offset, so read offset + max_poses total)
    total_read = (args.pose_offset + args.max_poses) if args.max_poses else 0
    header, pivots, ens, dofs0, _, centered_ligands = parse_dat_two_body(
        args.input_dat, max_poses=total_read
    )
    # Apply offset
    if args.pose_offset > 0:
        ens = ens[args.pose_offset :]
        dofs0 = dofs0[args.pose_offset :]
    n = len(dofs0)
    if verbose:
        print(
            f"Poses: {n} (offset={args.pose_offset}), Method: {args.method}, "
            f"maxfun: {args.maxfun}, oracle: {args.oracle}"
        )
        print(f"Ensemble ids: {np.unique(ens)}, centered_ligands: {centered_ligands}")

    # --- Resolve ligand pivot ---
    if 2 in pivots:
        lig_pivot = pivots[2]
    else:
        ligand_pdb_path = args.ligand_pdb or os.path.join(test_dir, "ligandr.pdb")
        coor = []
        with open(ligand_pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    coor.append(
                        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    )
        lig_pivot = np.mean(coor, axis=0)
    if verbose:
        print(f"Ligand pivot: {lig_pivot}")

    # --- Convert centered → non-centered for JAX oracle only ---
    input_centered = bool(centered_ligands) if centered_ligands is not None else False
    converted_for_oracle = input_centered and args.oracle == "jax"
    if converted_for_oracle:
        dofs0 = dofs0.copy()
        dofs0[:, 3:6] -= lig_pivot[None, :]
        if verbose:
            print(
                "Converted centered-ligand translations to non-centered (tx/ty/tz -= pivot)"
            )

    # --- Build oracle ---
    tmpdir_ctx = None
    if args.oracle == "jax":
        from jax_scorer import JaxScoreOracle

        ens_list_path = args.receptor_ens_list or os.path.join(
            test_dir, "partner1-ensemble.list"
        )
        ligand_pdb_path = args.ligand_pdb or os.path.join(test_dir, "ligandr.pdb")
        grid_path = args.grid
        par_npz = args.attract_par_npz
        if not grid_path:
            raise ValueError("--grid is required for --oracle jax")
        if not par_npz:
            raise ValueError("--attract-par-npz is required for --oracle jax")
        oracle = JaxScoreOracle(
            receptor_ens_list=ens_list_path,
            ligand_pdb=ligand_pdb_path,
            grid_file=grid_path,
            attract_par_npz=par_npz,
            lig_pivot=lig_pivot,
            epsilon=args.epsilon,
            cdie=bool(args.cdie),
            energy_batch=args.energy_batch,
            max_nb_cap=args.max_nb_cap,
            nb_mode=args.nb_mode,
            nb_bucket_thresholds=args.nb_bucket_thresholds,
        )
        if verbose:
            print(
                "JAX oracle initialized "
                f"(energy_batch={args.energy_batch}, max_nb_cap={args.max_nb_cap}, "
                f"nb_mode={args.nb_mode}, nb_bucket_thresholds={args.nb_bucket_thresholds})"
            )
    else:
        paths = resolve_attract_paths(test_dir)
        import tempfile as _tempfile

        tmpdir_ctx = _tempfile.TemporaryDirectory()
        tmpdir = tmpdir_ctx.__enter__()
        oracle = LegacyScoreOracle(
            attract_bin=paths["attract_bin"],
            attract_par=paths["attract_par"],
            receptor_pdb=paths["receptor_pdb"],
            ligand_pdb=paths["ligand_pdb"],
            ens_list=paths["ens_list"],
            grid_header=paths["grid_header"],
            header=header,
            tmpdir=tmpdir,
            cwd=test_dir,
        )

    try:
        # Score starting poses
        if verbose:
            print("Scoring starting poses...")
        start_e, start_g = oracle.score_batch(ens, dofs0)
        if args.score:
            print_legacy_score(start_e, start_g)
            return
        print(
            f"  Starting energies: min={start_e.min():.3f} mean={start_e.mean():.3f} "
            f"max={start_e.max():.3f}"
        )

        # Minimize
        t1 = time.time()

        if args.method == "dfp-batch":
            # --- Batched minimization (one oracle call per tick) ---
            traj_prefix = f"{args.out_prefix}.traj" if args.traj else None
            dofs_out, energies_out, nfev_out = minfor_minimize_batched(
                oracle,
                ens,
                dofs0,
                maxfun=args.maxfun,
                init_metric=args.init_metric,
                trace_every=args.trace_every,
                traj_prefix=traj_prefix,
                traj_header=header,
            )
        else:
            # --- Per-pose minimization ---
            dofs_out = np.zeros_like(dofs0)
            energies_out = np.full(n, np.nan, dtype=np.float64)
            nfev_out = np.zeros(n, dtype=np.int32)

            for i in range(n):
                ens_id = int(ens[i])

                if args.method == "dfp":
                    x_best, f_best, nfev = minfor_minimize(
                        oracle,
                        ens_id,
                        dofs0[i],
                        maxfun=args.maxfun,
                        init_metric=args.init_metric,
                    )
                elif args.method.startswith("scipy-"):
                    scipy_method = args.method.replace("scipy-", "").upper()
                    if scipy_method == "LBFGSB":
                        scipy_method = "L-BFGS-B"

                    def _fg(x, _eid=ens_id):
                        e, g = oracle.score_single(_eid, x)
                        return e, g.astype(np.float64)

                    from scipy.optimize import minimize as scipy_minimize

                    res = scipy_minimize(
                        _fg,
                        dofs0[i],
                        method=scipy_method,
                        jac=True,
                        options={
                            "maxfun": args.maxfun,
                            "maxiter": args.maxfun,
                            "ftol": 1e-15,
                            "gtol": 1e-10,
                        },
                    )
                    x_best, f_best, nfev = res.x, res.fun, res.nfev
                else:
                    raise ValueError(f"Unknown method: {args.method}")

                dofs_out[i] = x_best
                energies_out[i] = f_best
                nfev_out[i] = nfev

                if args.trace_every and (i + 1) % args.trace_every == 0:
                    elapsed = time.time() - t1
                    rate = (i + 1) / elapsed
                    print(
                        f"  [{i + 1}/{n}] nfev={nfev} energy={f_best:.4f} "
                        f"(start={start_e[i]:.4f} delta={start_e[i]-f_best:.4f}) "
                        f"({elapsed:.1f}s, {rate:.2f} poses/s)"
                    )

        t2 = time.time()
    finally:
        if tmpdir_ctx is not None:
            tmpdir_ctx.__exit__(None, None, None)

    # --- Convert back to centered if we converted for JAX input ---
    if converted_for_oracle:
        dofs_out = dofs_out.copy()
        dofs_out[:, 3:6] += lig_pivot[None, :]

    # Summary
    print(f"\nMinimization done in {t2 - t1:.1f}s ({n} poses)")
    print(
        f"  nfev: mean={nfev_out.mean():.1f} median={np.median(nfev_out):.0f} "
        f"min={nfev_out.min()} max={nfev_out.max()}"
    )
    print(
        f"  energy: min={energies_out.min():.3f} mean={energies_out.mean():.3f} "
        f"p1={np.percentile(energies_out, 1):.3f} p50={np.percentile(energies_out, 50):.3f}"
    )
    improv = start_e - energies_out
    print(
        f"  improvement: mean={improv.mean():.3f} median={np.median(improv):.3f} "
        f"max={improv.max():.3f}"
    )

    mats = dofs_to_mats_np(
        dofs_out if not input_centered else dofs_out.copy(), lig_pivot
    )

    np.save(args.out_prefix + ".dofs.npy", dofs_out.astype(np.float32))
    np.save(args.out_prefix + ".mat4.npy", mats.astype(np.float32))
    np.save(args.out_prefix + ".energy.npy", energies_out.astype(np.float32))
    np.save(args.out_prefix + ".ens.npy", ens.astype(np.int32))
    np.save(args.out_prefix + ".nfev.npy", nfev_out.astype(np.int32))

    out_dat = args.out_prefix + ".dat"
    write_dat_two_body(out_dat, header, ens, dofs_out, energies=energies_out)
    print(f"Saved: {args.out_prefix}.[dofs|energy|mat4|ens|nfev].npy + {out_dat}")

    # Compare with legacy
    if args.legacy_dat:
        _, _, ens_ref, _, e_ref, _ = parse_dat_two_body(args.legacy_dat, max_poses=n)
        m = min(n, len(e_ref))
        if np.isfinite(e_ref[:m]).all():
            summarize("energy_vs_legacy", e_ref[:m], energies_out[:m])
            for i in range(m):
                print(
                    f"  Pose {i+1}: legacy={e_ref[i]:.3f} ours={energies_out[i]:.3f} "
                    f"delta={energies_out[i]-e_ref[i]:.3f} nfev={nfev_out[i]}"
                )

    if hasattr(oracle, "print_stats"):
        oracle.print_stats()

    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
