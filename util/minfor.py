#!/usr/bin/env python3
"""Minimizer benchmark: batched VA13 minimizer with ATTRACT-style oracles.

Implements a quasi-Newton minimizer matching the key parameters of legacy
minfor (Harwell VA13 variable-metric method):
- Initial Hessian: H_0 = diag_value * I  (default 0.01, matching minfor)
- Armijo sufficient decrease with c1 = 0.1
- Curvature condition threshold = 0.7
- Cubic interpolation backtracking
- Step extrapolation up to 9x current step
- DFP Hessian update

Primary path uses ATTRACT-JAX (optionally with nonbon8 C++ NB kernel). The
compiled ATTRACT binary ($ATTRACTDIR) in --score mode remains available as a
fallback oracle.
"""

import argparse
import math
import os
import re
import subprocess
import sys
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


def _allclose_zero(values, atol: float = 1.0e-12) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.abs(arr) <= atol))


def parse_dat_two_body(path: str, max_poses: int = 0):
    from dat_to_npy import parse_dat_two_body_loadtxt

    (
        header,
        pivots,
        receptor_ens,
        dofs,
        energies,
        centered_ligands,
        ligand_ens,
        _receptor_has_ens,
        _ligand_has_ens,
    ) = parse_dat_two_body_loadtxt(path, max_poses=max_poses)
    return header, pivots, receptor_ens, dofs, energies, centered_ligands, ligand_ens


def write_dat_two_body(path, header, ens, dofs, energies=None, ligand_ens=None):
    with open(path, "w") as f:
        for line in header:
            f.write(line)
        for i in range(len(dofs)):
            f.write(f"#{i+1}\n")
            if energies is not None and np.isfinite(energies[i]):
                f.write(f"## Energy: {energies[i]:.15e}\n")
            f.write(f"{int(ens[i]):12d}{0:12d}{0:12d}{0:12d}{0:12d}{0:12d}{0:12d}\n")
            phi, ssi, rot, xa, ya, za = dofs[i]
            if ligand_ens is None:
                f.write(
                    f"{phi:24.16f} {ssi:24.16f} {rot:24.16f} "
                    f"{xa:24.16f} {ya:24.16f} {za:24.16f}\n"
                )
            else:
                f.write(
                    f"{int(ligand_ens[i]) + 1:12d} {phi:24.16f} {ssi:24.16f} {rot:24.16f} "
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


def print_legacy_score(energies, gradients, include_gradients=True):
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
        if include_gradients:
            print(" Gradients:" + "".join(f"{float(v):24.16E}" for v in f))


# ---------------------------------------------------------------------------
# Legacy ATTRACT --score oracle
# ---------------------------------------------------------------------------
class LegacyScoreOracle:
    def __init__(
        self,
        attract_bin,
        attract_par,
        shm_grid_bin,
        shm_clean_bin,
        receptor_pdb,
        ligand_pdb,
        ens_list,
        grid,
        grid_header,
        header,
        tmpdir,
        cwd=".",
    ):
        self.attract_bin = os.path.abspath(attract_bin)
        self.attract_par = os.path.abspath(attract_par)
        self.shm_grid_bin = os.path.abspath(shm_grid_bin)
        self.shm_clean_bin = os.path.abspath(shm_clean_bin)
        self.receptor_pdb = os.path.abspath(receptor_pdb)
        self.ligand_pdb = os.path.abspath(ligand_pdb)
        self.ens_list = os.path.abspath(ens_list)
        self.grid = os.path.abspath(grid)
        self.grid_header = os.path.abspath(grid_header)
        self.header = header
        self.tmpdir = os.path.abspath(tmpdir)
        self.cwd = os.path.abspath(cwd)
        self._call_count = 0
        # Load grid into shared memory.
        subprocess.run([self.shm_clean_bin], check=False)
        subprocess.run([self.shm_grid_bin, self.grid, self.grid_header], check=True)

    def close(self):
        subprocess.run([self.shm_clean_bin], check=False)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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

    def score_single(self, ens_id, dof, conformer=None):
        self._call_count += 1
        dat_path = os.path.join(self.tmpdir, f"_s{self._call_count}.dat")
        write_dat_two_body(
            dat_path, self.header, np.array([ens_id], np.int32), dof.reshape(1, 6)
        )
        e, g = self._run_score(dat_path, 1)
        os.unlink(dat_path)
        # ATTRACT --score prints forces (−∂E/∂x); negate to get gradient (+∂E/∂x)
        return float(e[0]), -g[0]

    def score_batch(self, ens, dofs, conformers=None):
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
# Batched VA13 minimizer — one oracle call per tick for all active poses
# ---------------------------------------------------------------------------
def minfor_minimize_batched(
    oracle,
    ens,
    dofs0,
    conformers=None,
    maxfun=150,
    init_metric=0.01,
    acc=1e-9,
    trace_every=0,
    traj_prefix=None,
    traj_header=None,
    report_step_complete=False,
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
    e0, g0 = oracle.score_batch(ens, dofs0, conformers=conformers)
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
        batch_conformers = None if conformers is None else conformers[act_idx]
        e_batch, g_batch = oracle.score_batch(
            ens[act_idx], xbb[act_idx], conformers=batch_conformers
        )
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

        if report_step_complete:
            n_active = int(active.sum())
            elapsed = time.time() - t_batch
            print(
                f"  completed tick {tick}: {n_active}/{N} active "
                f"(elapsed {elapsed:.1f}s)"
            )

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


def rotvec2rotmat_np(v):
    """Rodrigues rotvec (N,3) → rotation matrices (N,3,3) standard convention."""
    theta = np.linalg.norm(v, axis=1, keepdims=True)  # (N,1)
    safe_t = np.where(theta > 1.0e-10, theta, 1.0)
    k = v / safe_t  # (N,3)
    s = np.where(theta > 1.0e-10, np.sin(theta), theta).squeeze(1)
    c = np.where(theta > 1.0e-10, np.cos(theta), 1.0 - 0.5 * theta * theta).squeeze(1)
    omc = 1.0 - c
    k0, k1, k2 = k[:, 0], k[:, 1], k[:, 2]
    out = np.zeros((len(v), 3, 3), dtype=np.float64)
    out[:, 0, 0] = c + omc * k0 * k0
    out[:, 0, 1] = omc * k0 * k1 - s * k2
    out[:, 0, 2] = omc * k0 * k2 + s * k1
    out[:, 1, 0] = omc * k1 * k0 + s * k2
    out[:, 1, 1] = c + omc * k1 * k1
    out[:, 1, 2] = omc * k1 * k2 - s * k0
    out[:, 2, 0] = omc * k2 * k0 - s * k1
    out[:, 2, 1] = omc * k2 * k1 + s * k0
    out[:, 2, 2] = c + omc * k2 * k2
    return out


def rotvec_dofs_to_mats_np(dofs, pivot):
    rot_col = rotvec2rotmat_np(dofs[:, :3])
    rot_row = np.swapaxes(rot_col, 1, 2)
    pivot_rot = np.einsum("j,bji->bi", pivot, rot_row)
    trans = dofs[:, 3:6] + pivot[None, :] - pivot_rot
    mats = np.zeros((len(dofs), 4, 4), dtype=np.float64)
    mats[:, :3, :3] = rot_row
    mats[:, 3, :3] = trans
    mats[:, 3, 3] = 1.0
    return mats


def rotvec_world_to_attr_np(dofs, pivot):
    """Convert world-centered rotvec DOFs to ATTRACT pivot-centered DOFs.

    `pivot` is the single fixed ligand pivot used by ATTRACT semantics.
    For ligand ensembles, this is the pivot of conformer 1 / index 0.
    """
    out = np.asarray(dofs, dtype=np.float64).copy()
    rot_col = rotvec2rotmat_np(out[:, :3])
    out[:, 3:6] += np.einsum("bij,j->bi", rot_col, pivot) - pivot[None, :]
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


def _mean_pivot_from_pdb(path: str) -> np.ndarray:
    coor = []
    with open(path) as f:
        for line in f:
            if line.startswith("ATOM"):
                coor.append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                )
    if not coor:
        raise ValueError(f"No ATOM records found in ligand PDB: {path}")
    return np.mean(np.asarray(coor, dtype=np.float64), axis=0)


def _normalize_ligand_ensemble_array(coords: np.ndarray, source: str) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError(f"{source}: expected shape (N,3), got {arr.shape}")
        arr = arr[None, :, :]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{source}: expected shape (N,3) or (C,N,3), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _load_required_vector(path: str, expected_len: int, name: str, dtype) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=dtype)
    if arr.ndim != 1 or len(arr) != expected_len:
        raise ValueError(
            f"{name}: expected shape ({expected_len},), got {arr.shape} from {path}"
        )
    return arr


def _load_ligand_conformers(
    path: Optional[str], nposes: int, nconformers: int
) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
    if len(arr) != nposes:
        raise ValueError(
            f"--ligand-conformers length mismatch: expected {nposes}, got {len(arr)}"
        )
    if nconformers <= 0:
        raise ValueError("Ligand conformer library is empty")
    if arr.size == 0:
        return arr.astype(np.int32)
    if arr.min() >= 1 and arr.max() <= nconformers:
        arr = arr - 1
    elif arr.min() >= 0 and arr.max() < nconformers:
        arr = arr.copy()
    else:
        raise ValueError(
            "--ligand-conformers must be 1-based in [1, C] or 0-based in [0, C-1]"
        )
    return arr.astype(np.int32, copy=False)


def _load_ligand_pdb_list(path: str):
    from reproduce_grid_score import parse_reduced_pdb

    list_dir = Path(path).resolve().parent
    pdb_paths: List[str] = []
    coords: List[np.ndarray] = []
    atomtypes_ref: Optional[np.ndarray] = None
    charges_ref: Optional[np.ndarray] = None

    with open(path) as f:
        entries = [line.strip() for line in f if line.strip()]
    if not entries:
        raise ValueError(f"Empty ligand PDB list: {path}")

    for i, entry in enumerate(entries, start=1):
        p = Path(entry)
        if not p.is_absolute():
            p = list_dir / p
        pdb_paths.append(str(p))
        coor, atomtypes, charges, _weights = parse_reduced_pdb(str(p))
        if atomtypes_ref is None:
            atomtypes_ref = np.asarray(atomtypes, dtype=np.int32)
            charges_ref = np.asarray(charges, dtype=np.float64)
        else:
            if atomtypes.shape != atomtypes_ref.shape or not np.array_equal(
                atomtypes, atomtypes_ref
            ):
                raise ValueError(
                    f"{path}: ligand PDB #{i} has different atom types/size; "
                    "mixed ligand metadata is not supported in --ligand-pdb-list mode"
                )
            if charges.shape != charges_ref.shape or not np.allclose(
                charges, charges_ref
            ):
                raise ValueError(
                    f"{path}: ligand PDB #{i} has different charges; "
                    "mixed ligand charges are not supported in --ligand-pdb-list mode"
                )
        coords.append(np.asarray(coor, dtype=np.float64))

    assert atomtypes_ref is not None and charges_ref is not None
    return {
        "ligand_pdb_path": pdb_paths[0],
        "ligand_ensemble": np.asarray(coords, dtype=np.float64),
        "ligand_atomtypes": atomtypes_ref,
        "ligand_charges": charges_ref,
        "ligand_atomtypes_for_grid": atomtypes_ref,
        "ligand_pivot": np.mean(coords[0], axis=0),
    }


def resolve_ligand_inputs(args, test_dir: str, nposes: int):
    from reproduce_grid_score import parse_reduced_pdb

    if args.ligand_ensemble:
        lig_coords = _normalize_ligand_ensemble_array(
            np.load(args.ligand_ensemble), "--ligand-ensemble"
        )
        natoms = lig_coords.shape[1]
        lig_atomtypes = _load_required_vector(
            args.ligand_atomtypes, natoms, "--ligand-atomtypes", np.int32
        )
        if args.ligand_charges:
            lig_charges = _load_required_vector(
                args.ligand_charges, natoms, "--ligand-charges", np.float64
            )
        else:
            lig_charges = np.zeros((natoms,), dtype=np.float64)
        return {
            "mode": "ensemble",
            "ligand_pdb_path": None,
            "ligand_ensemble": lig_coords,
            "ligand_atomtypes": lig_atomtypes,
            "ligand_charges": lig_charges,
            "ligand_conformers": _load_ligand_conformers(
                args.ligand_conformers, nposes, lig_coords.shape[0]
            ),
            "ligand_atomtypes_for_grid": lig_atomtypes,
            "ligand_pivot": np.mean(lig_coords[0], axis=0),
        }

    if args.ligand_pdb_list:
        result = _load_ligand_pdb_list(args.ligand_pdb_list)
        result.update(
            {
                "mode": "pdb-list",
                "ligand_conformers": _load_ligand_conformers(
                    args.ligand_conformers,
                    nposes,
                    int(result["ligand_ensemble"].shape[0]),
                ),
            }
        )
        return result

    ligand_pdb_path = args.ligand_pdb or os.path.join(test_dir, "ligandr.pdb")
    lig_coords0, lig_atomtypes0, lig_charge0, _lig_w = parse_reduced_pdb(ligand_pdb_path)
    return {
        "mode": "pdb",
        "ligand_pdb_path": ligand_pdb_path,
        "ligand_ensemble": None,
        "ligand_atomtypes": None,
        "ligand_charges": None,
        "ligand_conformers": None,
        "ligand_atomtypes_for_grid": np.asarray(lig_atomtypes0, dtype=np.int32),
        "ligand_pivot": _mean_pivot_from_pdb(ligand_pdb_path),
    }


def resolve_receptor_ensemble_list(args, test_dir: str):
    if args.receptor_ens_list:
        return args.receptor_ens_list, None
    receptor_pdb = args.receptor_pdb
    if receptor_pdb is None:
        default_single = os.path.join(test_dir, "partner1-ensemble", "model-1r.pdb")
        if os.path.isfile(default_single):
            receptor_pdb = default_single
    if receptor_pdb is None:
        return os.path.join(test_dir, "partner1-ensemble.list"), None

    tmpdir_ctx = tempfile.TemporaryDirectory(prefix="minfor_receptor_ens_")
    list_path = os.path.join(tmpdir_ctx.name, "receptor.list")
    receptor_path_abs = os.path.abspath(receptor_pdb)
    with open(list_path, "w") as f:
        f.write(receptor_path_abs + "\n")
    return list_path, tmpdir_ctx


def resolve_receptor_inputs(args, test_dir: str):
    if args.receptor_coordinates:
        rec_coords = np.asarray(np.load(args.receptor_coordinates), dtype=np.float64)
        if rec_coords.ndim == 2:
            if rec_coords.shape[1] != 3:
                raise ValueError(
                    f"--receptor-coordinates expected shape (N,3) or (E,N,3), got {rec_coords.shape}"
                )
        elif rec_coords.ndim != 3 or rec_coords.shape[2] != 3:
            raise ValueError(
                f"--receptor-coordinates expected shape (N,3) or (E,N,3), got {rec_coords.shape}"
            )
        natoms = rec_coords.shape[-2]
        rec_atomtypes = _load_required_vector(
            args.receptor_atomtypes, natoms, "--receptor-atomtypes", np.int32
        )
        if args.receptor_charges:
            rec_charges = _load_required_vector(
                args.receptor_charges, natoms, "--receptor-charges", np.float64
            )
        else:
            rec_charges = np.zeros((natoms,), dtype=np.float64)
        return {
            "receptor_ens_list": None,
            "receptor_ensemble": rec_coords,
            "receptor_atomtypes": rec_atomtypes,
            "receptor_charges": rec_charges,
            "tmp_ctx": None,
        }

    ens_list_path, tmp_ctx = resolve_receptor_ensemble_list(args, test_dir)
    return {
        "receptor_ens_list": ens_list_path,
        "receptor_ensemble": None,
        "receptor_atomtypes": None,
        "receptor_charges": None,
        "tmp_ctx": tmp_ctx,
    }


def resolve_attract_paths(test_dir, ligand_pdb=None):
    attractdir = os.environ.get("ATTRACTDIR", "")
    if not attractdir:
        raise RuntimeError("$ATTRACTDIR is not set")
    attract_bin = os.path.join(attractdir, "attract")
    attract_par = os.path.join(attractdir, "..", "attract.par")
    test_dir = os.path.abspath(test_dir)
    ligand_pdb_path = (
        os.path.abspath(ligand_pdb)
        if ligand_pdb is not None
        else os.path.join(test_dir, "ligandr.pdb")
    )
    return {
        "attract_bin": attract_bin,
        "attract_par": attract_par,
        "shm_grid_bin": os.path.join(attractdir, "shm-grid"),
        "shm_clean_bin": os.path.join(attractdir, "shm-clean"),
        "receptor_pdb": os.path.join(test_dir, "partner1-ensemble", "model-1r.pdb"),
        "ligand_pdb": ligand_pdb_path,
        "ens_list": os.path.join(test_dir, "partner1-ensemble.list"),
        "grid": os.path.join(test_dir, "receptorgrid.grid"),
        "grid_header": os.path.join(test_dir, "receptorgrid.gridheader"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Batched VA13/DFP minimizer. Primary path: ATTRACT-JAX "
            "(default, nonbon8 NB by default). Legacy ATTRACT --score oracle "
            "is retained as fallback."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "input_dat",
        nargs="?",
        default=None,
        help="starting .dat file (optional when --input-npy is provided)",
    )
    ap.add_argument(
        "--input-npy",
        default=None,
        metavar="DOF_NPY",
        help="Nx6 float64 array: rotations+translations; interpretation set by --input-format",
    )
    ap.add_argument(
        "--input-format",
        choices=("rotvec", "euler"),
        default=None,
        help="rotation parameterization for --input-npy",
    )
    ap.add_argument(
        "--input-euler",
        default=None,
        metavar="EULER_NPY",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--input-rotvec",
        default=None,
        metavar="ROTVEC_NPY",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--input-conformers",
        default=None,
        metavar="CONF_NPY",
        help="N int32 array: 0-based ligand conformer index per pose (for --input-npy)",
    )
    ap.add_argument(
        "--input-ens",
        default=None,
        metavar="ENS_NPY",
        help="N int32 array: receptor ensemble index per pose (accepts 0-based or 1-based for --input-npy)",
    )
    ap.add_argument(
        "--input-world-centered",
        action="store_true",
        default=False,
        help=(
            "For --input-rotvec, input translations are world-frame translations "
            "(for example raw offsets from stack.py / convert_poses pre-inversion). "
            "minfor converts them to ATTRACT tx/ty/tz using the ligand pivot and pose rotation."
        ),
    )
    ap.add_argument(
        "--input-pivot-centered",
        action="store_true",
        default=False,
        help=(
            "For --input-rotvec, input translations are already ATTRACT tx/ty/tz "
            "DOFs relative to the ligand pivot. This is the default if no "
            "translation-convention flag is provided."
        ),
    )
    ap.add_argument(
        "--input-centered",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument(
        "--score",
        action="store_true",
        help="score-only mode: print legacy-style energy/gradient blocks",
    )
    ap.add_argument(
        "--energy-only",
        action="store_true",
        help="with --score: print energy-only legacy-style blocks (omit Gradients lines)",
    )
    ap.add_argument(
        "--benchmark-steady-state",
        action="store_true",
        help=(
            "benchmarking instrument for JAX --score: run one untimed warmup "
            "score pass, then time a second steady-state pass and report it to stderr"
        ),
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
    # --- JAX oracle options ---
    ap.add_argument(
        "--oracle",
        default="jax",
        choices=["jax", "legacy"],
        help=(
            "energy oracle: 'jax' (primary production path) or "
            "'legacy' (fallback ATTRACT binary)"
        ),
    )
    ap.add_argument(
        "--grid",
        default=None,
        help=(
            "path to precomputed .grid or .npz file to use as the NB potential grid. "
            "If omitted, the grid is generated just-in-time from the receptor "
            "(requires --receptor-ens-list and --attract-par-npz). "
            "With --generate-grid: output path for the freshly generated .npz."
        ),
    )
    ap.add_argument(
        "--generate-grid",
        action="store_true",
        default=False,
        help=(
            "JAX-only: generate the NB potential grid in-house and write it to "
            "--grid <output.npz>, then exit.  Requires --grid, "
            "--receptor-ens-list, --attract-par-npz.  "
            "Without --generate-grid, omitting --grid causes the grid to be "
            "generated just-in-time and used directly (not saved)."
        ),
    )
    ap.add_argument("--attract-par-npz", default=None, help="path to attract-par.npz")
    ap.add_argument(
        "--receptor-ens-list", default=None, help="path to receptor ensemble list"
    )
    ap.add_argument(
        "--receptor-pdb",
        default=None,
        help=(
            "single receptor PDB to wrap as a one-line ensemble list when "
            "--receptor-ens-list is not provided"
        ),
    )
    ap.add_argument(
        "--receptor-coordinates",
        default=None,
        help="optional receptor coordinate .npy with shape (N,3) or (E,N,3)",
    )
    ap.add_argument(
        "--receptor-atomtypes",
        default=None,
        help="required with --receptor-coordinates: per-atom ATTRACT types (.npy)",
    )
    ap.add_argument(
        "--receptor-charges",
        default=None,
        help="optional with --receptor-coordinates: per-atom charges (.npy)",
    )
    ap.add_argument(
        "--ligand-pdb",
        default=None,
        help=(
            "path to ligand PDB used by both oracles "
            "(default: {test-dir}/ligandr.pdb)"
        ),
    )
    ap.add_argument(
        "--ligand-ensemble",
        default=None,
        help="ligand coordinate .npy with shape (N,3) or (C,N,3)",
    )
    ap.add_argument(
        "--ligand-conformers",
        default=None,
        help="per-pose ligand conformer indices (.npy; accepts 1-based or 0-based)",
    )
    ap.add_argument(
        "--ligand-atomtypes",
        default=None,
        help="required with --ligand-ensemble: per-atom ATTRACT types (.npy, shape (N,))",
    )
    ap.add_argument(
        "--ligand-charges",
        default=None,
        help="optional with --ligand-ensemble: per-atom charges (.npy, shape (N,))",
    )
    ap.add_argument(
        "--ligand-pdb-list",
        default=None,
        help="optional ligand ensemble list: one reduced ligand PDB per line",
    )
    ap.add_argument(
        "--epsilon",
        type=float,
        default=15.0,
        help="JAX-only dielectric constant",
    )
    ap.add_argument(
        "--energy-batch",
        type=int,
        default=256,
        help=(
            "JAX-only: max poses per kernel call "
            "(merged-ensemble: all ensembles in one call)"
        ),
    )
    ap.add_argument(
        "--score-mode",
        default="default",
        choices=["default", "bulk"],
        help=(
            "JAX-only score scheduling mode. "
            "'default' keeps minimization-oriented batching; "
            "'bulk' uses larger score-only chunks."
        ),
    )
    ap.add_argument(
        "--score-batch-size",
        type=int,
        default=None,
        help=(
            "JAX-only: optional chunk size for --score bulk mode. "
            "If omitted, bulk mode uses an internal default."
        ),
    )
    ap.add_argument(
        "--pool-conformers",
        action="store_true",
        help=(
            "JAX-only: for score-only pooled-conformer batches, avoid pre-splitting "
            "poses by ligand conformer before scoring."
        ),
    )
    ap.add_argument(
        "--nb-kernel",
        default="nonbon8",
        choices=["jax", "nonbon8"],
        help=(
            "JAX-only NB backend: 'nonbon8' (C++ nonbon8 NB kernel, default) or "
            "'jax' (pure JAX)"
        ),
    )
    ap.add_argument(
        "--autodiff-potentials",
        action="store_true",
        help=(
            "JAX-only: derive potential-grid gradients via AD from energy channels "
            "(disables stored grid-gradient usage)"
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
    ap.add_argument(
        "--report-step-complete",
        action="store_true",
        help="print a progress line whenever a batched minimization tick completes",
    )
    args = ap.parse_args()
    argv = sys.argv[1:]

    def _opt_used(opt_name: str) -> bool:
        return any(tok == opt_name or tok.startswith(opt_name + "=") for tok in argv)

    if args.oracle == "legacy":
        jax_only_opts = [
            "--grid",
            "--generate-grid",
            "--attract-par-npz",
            "--receptor-ens-list",
            "--epsilon",
            "--energy-batch",
            "--benchmark-steady-state",
            "--score-mode",
            "--score-batch-size",
            "--pool-conformers",
            "--nb-kernel",
            "--autodiff-potentials",
            "--disable-jit",
        ]
        used = [opt for opt in jax_only_opts if _opt_used(opt)]
        if used:
            ap.error(
                "The following options are JAX-only and cannot be used with "
                "--oracle legacy: " + ", ".join(used)
            )
        if (
            args.ligand_ensemble
            or args.ligand_pdb_list
            or args.ligand_conformers
            or args.receptor_coordinates
        ):
            ap.error(
                "--oracle legacy only supports PDB/list receptor+ligand inputs; "
                "--ligand-ensemble/--ligand-pdb-list/--ligand-conformers/"
                "--receptor-coordinates require --oracle jax"
            )
    else:
        if args.generate_grid and not args.grid:
            ap.error(
                "--generate-grid requires --grid <output.npz> "
                "(the .npz path to write the generated grid to)"
            )
        if not args.attract_par_npz:
            ap.error("--attract-par-npz is required for --oracle jax")
        if args.energy_batch <= 0:
            ap.error("--energy-batch must be >= 1")
        if args.score_batch_size is not None and args.score_batch_size <= 0:
            ap.error("--score-batch-size must be >= 1")
    if args.energy_only and not args.score:
        ap.error("--energy-only is only valid together with --score")
    if args.benchmark_steady_state and not args.score:
        ap.error("--benchmark-steady-state is only valid together with --score")
    if args.benchmark_steady_state and args.oracle != "jax":
        ap.error("--benchmark-steady-state currently requires --oracle jax")
    if args.score_mode != "default" and not args.score:
        ap.error("--score-mode is only valid together with --score")
    if args.score_batch_size is not None and not args.score:
        ap.error("--score-batch-size is only valid together with --score")
    if args.pool_conformers and not args.score:
        ap.error("--pool-conformers is only valid together with --score")
    if args.pool_conformers and not args.energy_only:
        ap.error("--pool-conformers currently requires --energy-only")
    if args.pool_conformers and args.nb_kernel != "jax":
        ap.error("--pool-conformers currently requires --nb-kernel jax")
    receptor_mode_count = int(bool(args.receptor_ens_list)) + int(bool(args.receptor_pdb)) + int(
        bool(args.receptor_coordinates)
    )
    if receptor_mode_count > 1:
        ap.error(
            "--receptor-ens-list, --receptor-pdb, and --receptor-coordinates are mutually exclusive"
        )
    if args.receptor_coordinates and not args.receptor_atomtypes:
        ap.error("--receptor-coordinates requires --receptor-atomtypes")
    if args.receptor_atomtypes and not args.receptor_coordinates:
        ap.error("--receptor-atomtypes is only valid with --receptor-coordinates")
    if args.receptor_charges and not args.receptor_coordinates:
        ap.error("--receptor-charges is only valid with --receptor-coordinates")
    ligand_mode_count = int(bool(args.ligand_pdb)) + int(bool(args.ligand_ensemble)) + int(
        bool(args.ligand_pdb_list)
    )
    if ligand_mode_count > 1:
        ap.error(
            "--ligand-pdb, --ligand-ensemble, and --ligand-pdb-list are mutually exclusive"
        )
    if args.ligand_ensemble and not args.ligand_atomtypes:
        ap.error("--ligand-ensemble requires --ligand-atomtypes")
    if args.ligand_atomtypes and not args.ligand_ensemble:
        ap.error("--ligand-atomtypes is only valid with --ligand-ensemble")
    if args.ligand_charges and not args.ligand_ensemble:
        ap.error("--ligand-charges is only valid with --ligand-ensemble")
    if args.ligand_conformers and not (args.ligand_ensemble or args.ligand_pdb_list):
        ap.error(
            "--ligand-conformers requires --ligand-ensemble or --ligand-pdb-list"
        )
    nx6_inputs = int(bool(args.input_npy)) + int(bool(args.input_rotvec)) + int(bool(args.input_euler))
    if nx6_inputs > 1:
        ap.error("--input-npy, --input-rotvec, and --input-euler are mutually exclusive")
    if args.input_rotvec:
        args.input_npy = args.input_rotvec
        args.input_format = "rotvec"
    elif args.input_euler:
        args.input_npy = args.input_euler
        args.input_format = "euler"
    if args.input_npy and args.input_format is None:
        ap.error("--input-format is required with --input-npy")
    if args.input_format and not args.input_npy:
        ap.error("--input-format requires --input-npy")
    if args.input_npy:
        if args.input_dat:
            ap.error("input_dat is mutually exclusive with --input-npy")
        if args.oracle != "jax":
            ap.error("--input-npy requires --oracle jax")
    else:
        if not args.input_dat:
            ap.error("input_dat is required unless --input-npy is provided")
    if args.input_conformers and not args.input_npy:
        ap.error("--input-conformers requires --input-npy")
    if args.input_ens and not args.input_npy:
        ap.error("--input-ens requires --input-npy")
    if (
        args.input_world_centered
        or args.input_pivot_centered
        or args.input_centered
    ) and args.input_format != "rotvec":
        ap.error(
            "--input-world-centered/--input-pivot-centered currently only apply to --input-format rotvec"
        )
    if args.input_world_centered and args.input_pivot_centered:
        ap.error(
            "--input-world-centered and --input-pivot-centered are mutually exclusive"
        )
    if args.input_centered:
        if args.input_pivot_centered:
            ap.error("--input-centered conflicts with --input-pivot-centered")
        args.input_world_centered = True

    return args


def main():
    args = parse_args()
    t0 = time.time()
    verbose = not args.score
    if not args.score and not args.out_prefix and not args.generate_grid:
        raise ValueError("--out-prefix is required unless --score is used")
    ligand_ens_dat = None

    if args.disable_jit and args.oracle == "jax":
        import jax

        jax.config.update("jax_disable_jit", True)
        if verbose:
            print("JAX JIT disabled (--disable-jit)")

    # --- Nx6 input path ---
    _dof_type = "euler"
    if args.input_npy:
        test_dir = args.test_dir or str(Path(args.input_npy).resolve().parent)
        dofs0 = np.load(args.input_npy).astype(np.float64)
        if dofs0.ndim != 2 or dofs0.shape[1] != 6:
            raise ValueError(
                f"Nx6 input must have shape (N,6), got {dofs0.shape}"
            )
        if args.input_conformers:
            ligand_conformers_rotvec = np.load(args.input_conformers).astype(np.int32).reshape(-1)
        else:
            ligand_conformers_rotvec = None
        if args.input_ens:
            ens = np.load(args.input_ens).astype(np.int64).reshape(-1)
            if len(ens) != len(dofs0):
                raise ValueError(
                    f"--input-ens length mismatch: expected {len(dofs0)}, got {len(ens)}"
                )
            if ens.size:
                if ens.min() >= 1:
                    ens = ens.astype(np.int32)
                elif ens.min() >= 0:
                    ens = (ens + 1).astype(np.int32)
                else:
                    raise ValueError("--input-ens must be 0-based or 1-based integers")
            else:
                ens = ens.astype(np.int32)
        else:
            ens = np.ones(len(dofs0), dtype=np.int32)
        if args.pose_offset > 0:
            dofs0 = dofs0[args.pose_offset :]
            if ligand_conformers_rotvec is not None:
                ligand_conformers_rotvec = ligand_conformers_rotvec[args.pose_offset :]
            ens = ens[args.pose_offset :]
        if args.max_poses:
            dofs0 = dofs0[: args.max_poses]
            if ligand_conformers_rotvec is not None:
                ligand_conformers_rotvec = ligand_conformers_rotvec[: args.max_poses]
            ens = ens[: args.max_poses]
        n = len(dofs0)
        if ligand_conformers_rotvec is not None and len(ligand_conformers_rotvec) != n:
            raise ValueError(
                f"--input-conformers length mismatch: expected {n}, got {len(ligand_conformers_rotvec)}"
            )
        header = []
        pivots = {}
        centered_ligands = None
        _dof_type = str(args.input_format)
        if verbose:
            print(
                f"Poses: {n} (offset={args.pose_offset}), "
                f"maxfun: {args.maxfun}, oracle: {args.oracle}, dof_type={_dof_type}"
            )
    else:
        test_dir = args.test_dir or str(Path(args.input_dat).resolve().parent)
        # Read poses (max_poses includes offset, so read offset + max_poses total)
        total_read = (args.pose_offset + args.max_poses) if args.max_poses else 0
        header, pivots, ens, dofs0, _, centered_ligands, ligand_ens_dat = parse_dat_two_body(
            args.input_dat, max_poses=total_read
        )
        # Apply offset
        if args.pose_offset > 0:
            ens = ens[args.pose_offset :]
            dofs0 = dofs0[args.pose_offset :]
            ligand_ens_dat = ligand_ens_dat[args.pose_offset :]
        n = len(dofs0)
        ligand_conformers_rotvec = None
        if verbose:
            print(
                f"Poses: {n} (offset={args.pose_offset}), "
                f"maxfun: {args.maxfun}, oracle: {args.oracle}"
            )
            print(f"Ensemble ids: {np.unique(ens)}, centered_ligands: {centered_ligands}")

    ligand_inputs = resolve_ligand_inputs(args, test_dir, n)
    ligand_pdb_path = ligand_inputs["ligand_pdb_path"]
    ligand_conformers = (
        ligand_conformers_rotvec
        if ligand_conformers_rotvec is not None
        else ligand_inputs["ligand_conformers"]
    )
    if ligand_ens_dat is not None and np.any(ligand_ens_dat):
        if np.any(ligand_ens_dat == 0):
            raise ValueError(
                "input_dat mixes ligand lines with and without ensemble indices; this is not supported"
            )
        if ligand_conformers is not None:
            raise ValueError(
                "input_dat already contains ligand ensemble indices; remove --ligand-conformers/--input-conformers"
            )
        ligand_library = ligand_inputs["ligand_ensemble"]
        if ligand_library is None:
            raise ValueError(
                "input_dat contains ligand ensemble indices, but no ligand ensemble library was provided"
            )
        nconformers = int(ligand_library.shape[0])
        if int(ligand_ens_dat.max()) > nconformers:
            raise ValueError(
                f"input_dat ligand ensemble index {int(ligand_ens_dat.max())} exceeds "
                f"the provided ligand library size {nconformers}"
            )
        ligand_conformers = ligand_ens_dat.astype(np.int32) - 1
    if 2 in pivots:
        lig_pivot = pivots[2]
    else:
        lig_pivot = np.asarray(ligand_inputs["ligand_pivot"], dtype=np.float64)
    if verbose:
        print(f"Ligand pivot: {lig_pivot}")

    # --- Convert world-frame translations → ATTRACT DOFs (rotvec path) ---
    if _dof_type == "rotvec" and args.input_world_centered:
        dofs0 = rotvec_world_to_attr_np(dofs0, lig_pivot)
        if verbose:
            print(
                "Converted world-frame rotvec translations to ATTRACT DOFs using the ligand pivot"
            )

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
    receptor_inputs = resolve_receptor_inputs(args, test_dir)
    receptor_ens_list_ctx = receptor_inputs["tmp_ctx"]
    oracle = None
    if args.oracle == "jax":
        from jax_scorer import JaxScoreOracle

        ens_list_path = receptor_inputs["receptor_ens_list"]
        grid_path = args.grid
        par_npz = args.attract_par_npz

        # Build the grid object when no precomputed file is provided, or when
        # --generate-grid is given (generate + write + exit).
        grid_object = None
        if args.generate_grid or not grid_path:
            import math as _math
            import sys as _sys

            _util_dir = os.path.dirname(os.path.abspath(__file__))
            _jax_root = os.path.dirname(_util_dir)
            for _p in (_util_dir, _jax_root):
                if _p not in _sys.path:
                    _sys.path.insert(0, _p)
            from grid_generator import generate_grid as _gen_grid
            from reproduce_grid_score import parse_reduced_pdb as _parse_pdb
            from native.nb_kernel.forcefields.nonbon8.params import (
                load_params as _load_params,
            )

            # Load receptor coordinates and atom types via the generic path
            _rc = receptor_inputs["receptor_ensemble"]
            _rt = receptor_inputs["receptor_atomtypes"]
            _rq = receptor_inputs["receptor_charges"]
            _rec_pdb = None
            if _rc is None:
                _ens_lines = open(ens_list_path).read().splitlines()
                _ens_lines = [l.strip() for l in _ens_lines if l.strip()]
                _rec_pdb = _ens_lines[0]
                if not os.path.isabs(_rec_pdb):
                    _rec_pdb = os.path.join(os.path.dirname(ens_list_path), _rec_pdb)
                _rc, _rt, _rq, _ = _parse_pdb(_rec_pdb)
            if _rc.ndim == 3:
                _rc = _rc[0]
            _ff_params = _load_params(par_npz)
            _ffelec = _math.sqrt(332.053986 / args.epsilon)
            # Ligand alphabet: prefer explicit ligand metadata when available.
            _lig_atomtypes = ligand_inputs["ligand_atomtypes_for_grid"]
            if _lig_atomtypes is None and ligand_pdb_path and os.path.isfile(ligand_pdb_path):
                _, _lt, _, _ = _parse_pdb(ligand_pdb_path)
                _lig_atomtypes = _lt
            _rec_label = _rec_pdb if _rec_pdb is not None else ens_list_path
            if verbose:
                if _lig_atomtypes is not None:
                    print(
                        f"Generating grid in-house from {_rec_label} "
                        "(ligand alphabet restricted to ligand metadata) ..."
                    )
                else:
                    print(
                        f"Generating grid in-house from {_rec_label} (all atomtypes) ..."
                    )
            grid_object = _gen_grid(
                rec_coords=_rc,
                rec_atomtypes=_rt,
                rec_charges_raw=_rq,
                ff_params=_ff_params,
                forcefield="nonbon8",
                ffelec=_ffelec,
                lig_atomtypes=_lig_atomtypes,
            )
            if args.generate_grid:
                # --generate-grid: write to --grid <path> and exit.
                from reproduce_grid_score import write_grid_npz as _write_npz

                _write_npz(grid_object, args.grid)
                if verbose:
                    print(f"Grid written to {args.grid}")
                import sys as _sys_exit

                _sys_exit.exit(0)
            grid_path = None  # use grid_object in JaxScoreOracle

        # Dispatch --grid on extension: .npz → read_grid_npz, else legacy binary.
        if grid_path is not None and grid_path.endswith(".npz"):
            from reproduce_grid_score import read_grid_npz as _read_npz

            grid_object = _read_npz(grid_path)
            grid_path = None  # use grid_object path in JaxScoreOracle
            if verbose:
                print(
                    f"Loaded NPZ grid (n_alpha={len(grid_object.alphabet_atomtypes)})"
                )

        oracle = JaxScoreOracle(
            receptor_ens_list=ens_list_path,
            receptor_ensemble=receptor_inputs["receptor_ensemble"],
            receptor_atomtypes=receptor_inputs["receptor_atomtypes"],
            receptor_charges=receptor_inputs["receptor_charges"],
            ligand_pdb=ligand_pdb_path,
            ligand_ensemble=ligand_inputs["ligand_ensemble"],
            ligand_atomtypes=ligand_inputs["ligand_atomtypes"],
            ligand_charges=ligand_inputs["ligand_charges"],
            grid_file=grid_path,
            attract_par_npz=par_npz,
            lig_pivot=lig_pivot,
            epsilon=args.epsilon,
            energy_batch=args.energy_batch,
            score_mode=args.score_mode,
            score_batch_size=args.score_batch_size,
            pool_conformers=bool(args.pool_conformers),
            nb_kernel=args.nb_kernel,
            autodiff_potentials=bool(args.autodiff_potentials),
            energy_only=bool(args.energy_only),
            grid_object=grid_object,
            dof_type=_dof_type,
        )
        if verbose:
            print(
                "JAX oracle initialized "
                f"(energy_batch={args.energy_batch}, score_mode={args.score_mode}, "
                f"score_batch_size={args.score_batch_size}, "
                f"pool_conformers={bool(args.pool_conformers)}, "
                f"nb_kernel={args.nb_kernel}, "
                f"autodiff_potentials={bool(args.autodiff_potentials)}, "
                f"energy_only={bool(args.energy_only)})"
            )
    else:
        paths = resolve_attract_paths(test_dir, ligand_pdb=ligand_pdb_path)
        import tempfile as _tempfile

        tmpdir_ctx = _tempfile.TemporaryDirectory()
        tmpdir = tmpdir_ctx.__enter__()
        oracle = LegacyScoreOracle(
            attract_bin=paths["attract_bin"],
            attract_par=paths["attract_par"],
            shm_grid_bin=paths["shm_grid_bin"],
            shm_clean_bin=paths["shm_clean_bin"],
            receptor_pdb=paths["receptor_pdb"],
            ligand_pdb=paths["ligand_pdb"],
            ens_list=paths["ens_list"],
            grid=paths["grid"],
            grid_header=paths["grid_header"],
            header=header,
            tmpdir=tmpdir,
            cwd=test_dir,
        )

    try:
        # Score starting poses
        if verbose:
            print("Scoring starting poses...")
        if args.benchmark_steady_state:
            if verbose:
                print(
                    "Benchmarking steady-state score pass "
                    "(warmup pass + timed pass)..."
                )
            oracle.score_batch(ens, dofs0, conformers=ligand_conformers)
            kernel_time0 = getattr(oracle, "_total_kernel_time", 0.0)
            kernel_calls0 = getattr(oracle, "_total_kernel_calls", 0)
            t_score0 = time.perf_counter()
            start_e, start_g = oracle.score_batch(
                ens, dofs0, conformers=ligand_conformers
            )
            t_score1 = time.perf_counter()
            kernel_time1 = getattr(oracle, "_total_kernel_time", kernel_time0)
            kernel_calls1 = getattr(oracle, "_total_kernel_calls", kernel_calls0)
            print(
                "Benchmark steady-state "
                f"(--benchmark-steady-state): wall={t_score1 - t_score0:.6f}s "
                f"kernel={kernel_time1 - kernel_time0:.6f}s "
                f"kernel_calls={kernel_calls1 - kernel_calls0}",
                file=sys.stderr,
            )
        else:
            start_e, start_g = oracle.score_batch(
                ens, dofs0, conformers=ligand_conformers
            )
        if args.score:
            print_legacy_score(
                start_e,
                start_g,
                include_gradients=(not bool(args.energy_only)),
            )
            return
        print(
            f"  Starting energies: min={start_e.min():.3f} mean={start_e.mean():.3f} "
            f"max={start_e.max():.3f}"
        )

        # Minimize
        t1 = time.time()

        # --- Batched minimization (one oracle call per tick) ---
        traj_prefix = f"{args.out_prefix}.traj" if args.traj else None
        dofs_out, energies_out, nfev_out = minfor_minimize_batched(
            oracle,
            ens,
            dofs0,
            conformers=ligand_conformers,
            maxfun=args.maxfun,
            init_metric=args.init_metric,
            trace_every=args.trace_every,
            traj_prefix=traj_prefix,
            traj_header=header,
            report_step_complete=bool(args.report_step_complete),
        )

        t2 = time.time()
    finally:
        if receptor_ens_list_ctx is not None:
            receptor_ens_list_ctx.cleanup()
        if tmpdir_ctx is not None:
            tmpdir_ctx.__exit__(None, None, None)
        if oracle is not None and hasattr(oracle, "close"):
            oracle.close()

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

    if _dof_type == "rotvec":
        mats = rotvec_dofs_to_mats_np(dofs_out, lig_pivot)
    else:
        mats = dofs_to_mats_np(
            dofs_out if not input_centered else dofs_out.copy(), lig_pivot
        )

    np.save(args.out_prefix + ".dofs.npy", dofs_out.astype(np.float32))
    np.save(args.out_prefix + ".mat4.npy", mats.astype(np.float32))
    np.save(args.out_prefix + ".energy.npy", energies_out.astype(np.float32))
    np.save(args.out_prefix + ".ens.npy", ens.astype(np.int32))
    np.save(args.out_prefix + ".nfev.npy", nfev_out.astype(np.int32))
    if ligand_conformers is not None and _dof_type == "rotvec":
        np.save(args.out_prefix + ".conformers.npy", ligand_conformers.astype(np.int32))

    if _dof_type != "rotvec":
        out_dat = args.out_prefix + ".dat"
        write_dat_two_body(
            out_dat,
            header,
            ens,
            dofs_out,
            energies=energies_out,
            ligand_ens=ligand_conformers,
        )
        print(f"Saved: {args.out_prefix}.[dofs|energy|mat4|ens|nfev].npy + {out_dat}")
    else:
        print(f"Saved: {args.out_prefix}.[dofs|energy|mat4|ens|nfev].npy")

    # Compare with legacy
    if args.legacy_dat:
        _, _, ens_ref, _, e_ref, _, _ = parse_dat_two_body(args.legacy_dat, max_poses=n)
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
