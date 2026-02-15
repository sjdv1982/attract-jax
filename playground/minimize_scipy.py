#!/usr/bin/env python3
"""Minimizer benchmark: scipy L-BFGS-B with legacy ATTRACT --score as oracle.

Uses the compiled ATTRACT binary (via $ATTRACTDIR) in --score mode to compute
energies and gradients. This avoids JAX memory issues and gives exact energy
agreement with legacy, letting us focus on minimizer quality.

Typical usage:
    python minimize_scipy.py test/systsearch-ens1-first200.dat \
        --out-prefix test/jax_scipy_first200 --maxfun 150

Environment:
    $ATTRACTDIR must point to the attract/bin directory with the attract binary.
    The test/ directory must contain the receptor/ligand/grid files from demo-xylanase.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize as scipy_minimize

# ---------------------------------------------------------------------------
# DAT parsing
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
    """Parse ATTRACT two-body .dat file."""
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
            raise ValueError(
                f"Expected first body line with 7 fields, got {len(first)}"
            )
        ens = int(round(first[0]))
        if len(second) == 7:
            second = second[1:]
        if len(second) != 6:
            raise ValueError(
                f"Expected ligand DOF line with 6 fields, got {len(second)}"
            )
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


def write_dat_two_body(
    path: str,
    header: List[str],
    ens: np.ndarray,
    dofs: np.ndarray,
    energies: Optional[np.ndarray] = None,
):
    """Write ATTRACT two-body .dat file."""
    with open(path, "w") as f:
        for line in header:
            f.write(line)
        for i in range(len(dofs)):
            if energies is not None and np.isfinite(energies[i]):
                f.write(f"## Energy: {energies[i]:.15e}\n")
            f.write(f"#{i+1}\n")
            f.write(f"{int(ens[i]):12d}{0:12d}{0:12d}{0:12d}{0:12d}{0:12d}{0:12d}\n")
            phi, ssi, rot, xa, ya, za = dofs[i]
            f.write(
                f"{phi:24.16f} {ssi:24.16f} {rot:24.16f} "
                f"{xa:24.16f} {ya:24.16f} {za:24.16f}\n"
            )


def parse_score_output(text: str):
    """Parse legacy ATTRACT --score output into energies and gradients."""
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
    if len(g) != len(e):
        raise ValueError(f"Parsed {len(e)} energies but {len(g)} gradients")
    return e, g


# ---------------------------------------------------------------------------
# Legacy ATTRACT --score oracle
# ---------------------------------------------------------------------------
class LegacyScoreOracle:
    """Calls the legacy attract binary with --score to get energies and gradients."""

    def __init__(
        self,
        attract_bin: str,
        attract_par: str,
        receptor_pdb: str,
        ligand_pdb: str,
        ens_list: str,
        grid_header: str,
        header: List[str],
        tmpdir: str,
        cwd: str = ".",
    ):
        # All paths must be absolute so they work regardless of cwd.
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

    def _run_score(self, dat_path: str, expected_n: int):
        """Run attract --score and parse output."""
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
            raise ValueError(
                f"Expected {expected_n} energies from --score, got {len(e)}"
            )
        return e, g

    def score_single(self, ens_id: int, dof: np.ndarray) -> Tuple[float, np.ndarray]:
        """Score a single pose. Returns (energy, gradient_6d)."""
        self._call_count += 1
        dat_path = os.path.join(self.tmpdir, f"_tmp_score_{self._call_count}.dat")
        ens_arr = np.array([ens_id], dtype=np.int32)
        dof_arr = dof.reshape(1, 6)
        write_dat_two_body(dat_path, self.header, ens_arr, dof_arr)
        e, g = self._run_score(dat_path, 1)
        os.unlink(dat_path)
        return float(e[0]), g[0]

    def score_batch(
        self, ens: np.ndarray, dofs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Score a batch of poses. Returns (energies, gradients)."""
        self._call_count += 1
        dat_path = os.path.join(self.tmpdir, f"_tmp_score_{self._call_count}.dat")
        write_dat_two_body(dat_path, self.header, ens, dofs)
        e, g = self._run_score(dat_path, len(dofs))
        os.unlink(dat_path)
        return e, g


# ---------------------------------------------------------------------------
# L-BFGS-B minimization with legacy oracle
# ---------------------------------------------------------------------------
def minimize_poses_sequential(
    oracle: LegacyScoreOracle,
    ens: np.ndarray,
    dofs0: np.ndarray,
    maxfun: int = 150,
    trace_every: int = 50,
):
    """Minimize each pose one at a time using scipy L-BFGS-B.

    Each function evaluation calls the legacy attract binary.
    This is simple but slow — O(n * maxfun) subprocess calls.
    """
    n = len(dofs0)
    dofs_out = np.zeros_like(dofs0)
    energies_out = np.full(n, np.nan, dtype=np.float64)
    nfev_out = np.zeros(n, dtype=np.int32)

    for i in range(n):
        ens_id = int(ens[i])
        dof0 = dofs0[i].copy()

        def func_and_grad(x):
            e, g = oracle.score_single(ens_id, x)
            return e, g.astype(np.float64)

        result = scipy_minimize(
            func_and_grad,
            dof0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxfun": maxfun,
                "maxiter": maxfun,
                "ftol": 1e-15,
                "gtol": 1e-10,
            },
        )

        dofs_out[i] = result.x
        energies_out[i] = result.fun
        nfev_out[i] = result.nfev

        if trace_every and (i + 1) % trace_every == 0:
            print(
                f"  [{i + 1}/{n}] ens={ens_id} nfev={result.nfev} "
                f"energy={result.fun:.4f} success={result.success}"
            )

    return dofs_out, energies_out, nfev_out


def minimize_poses_batched(
    oracle: LegacyScoreOracle,
    ens: np.ndarray,
    dofs0: np.ndarray,
    maxfun: int = 150,
    trace_every: int = 10,
):
    """Minimize all poses using a round-robin batched strategy.

    Instead of running each pose's optimizer to completion sequentially,
    this interleaves poses: it collects DOFs from all active poses that
    need an energy evaluation, scores them in one batch attract --score
    call, then feeds the results back into each pose's optimizer state.

    Since scipy L-BFGS-B doesn't support this natively, we use a manual
    L-BFGS implementation.
    """
    # For now, fall through to the sequential version.
    # The batched version will be implemented if the sequential one is too slow.
    return minimize_poses_sequential(oracle, ens, dofs0, maxfun, trace_every)


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
# Utilities
# ---------------------------------------------------------------------------
def summarize(name: str, ref: np.ndarray, cand: np.ndarray):
    d = cand - ref
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    corr = float(np.corrcoef(ref, cand)[0, 1]) if len(ref) > 1 else float("nan")
    p50 = float(np.percentile(d, 50))
    p90 = float(np.percentile(d, 90))
    p99 = float(np.percentile(d, 99))
    print(
        f"{name}: mae={mae:.4f} rmse={rmse:.4f} pearson={corr:.4f} "
        f"delta[p50,p90,p99]=[{p50:.4f},{p90:.4f},{p99:.4f}]"
    )


def resolve_attract_paths(test_dir: str) -> dict:
    """Resolve paths to ATTRACT binary and data files."""
    attractdir = os.environ.get("ATTRACTDIR", "")
    if not attractdir:
        raise RuntimeError(
            "$ATTRACTDIR is not set. Please set it to the attract/bin directory."
        )
    attract_bin = os.path.join(attractdir, "attract")
    attract_par = os.path.join(attractdir, "..", "attract.par")
    if not os.path.isfile(attract_bin):
        raise RuntimeError(f"attract binary not found at {attract_bin}")
    if not os.path.isfile(attract_par):
        raise RuntimeError(f"attract.par not found at {attract_par}")

    test_dir = os.path.abspath(test_dir)
    defaults = {
        "attract_bin": attract_bin,
        "attract_par": attract_par,
        "receptor_pdb": os.path.join(test_dir, "partner1-ensemble", "model-1r.pdb"),
        "ligand_pdb": os.path.join(test_dir, "ligandr.pdb"),
        "ens_list": os.path.join(test_dir, "partner1-ensemble.list"),
        "grid_header": os.path.join(test_dir, "receptorgrid.gridheader"),
    }
    for k, v in defaults.items():
        if not os.path.isfile(v):
            raise RuntimeError(f"Required file not found: {v} ({k})")
    return defaults


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Minimize poses using scipy L-BFGS-B with legacy ATTRACT --score oracle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input_dat", help="starting .dat file")
    ap.add_argument(
        "--out-prefix", required=True, help="output prefix for .npy/.dat files"
    )
    ap.add_argument(
        "--maxfun", type=int, default=150, help="max energy evaluations per pose"
    )
    ap.add_argument(
        "--max-poses", type=int, default=0, help="cap number of poses (0 = all)"
    )
    ap.add_argument(
        "--trace-every", type=int, default=10, help="print progress every N poses"
    )
    ap.add_argument(
        "--test-dir",
        default=None,
        help="test directory with receptor/ligand/grid files (default: parent of input_dat)",
    )
    ap.add_argument(
        "--legacy-dat", help="optional legacy minimized .dat for comparison"
    )
    ap.add_argument(
        "--method",
        default="L-BFGS-B",
        choices=["L-BFGS-B", "CG", "BFGS", "Newton-CG", "trust-ncg", "trust-krylov"],
        help="scipy optimization method",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # Resolve paths
    test_dir = args.test_dir or str(Path(args.input_dat).resolve().parent)
    paths = resolve_attract_paths(test_dir)

    # Parse input
    header, pivots, ens, dofs0, starting_energies, centered_ligands = (
        parse_dat_two_body(args.input_dat, max_poses=args.max_poses)
    )
    n = len(dofs0)
    print(f"Poses: {n}")
    print(f"Ensemble ids: {np.unique(ens)}")
    print(f"Centered ligands: {centered_ligands}")
    print(f"Method: {args.method}")

    # Create oracle
    with tempfile.TemporaryDirectory() as tmpdir:
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

        # Score starting poses
        print("Scoring starting poses...")
        t_score0 = time.time()
        start_e, start_g = oracle.score_batch(ens, dofs0)
        t_score1 = time.time()
        print(
            f"  Starting energies: min={start_e.min():.3f} mean={start_e.mean():.3f} "
            f"max={start_e.max():.3f}  ({t_score1 - t_score0:.2f}s)"
        )

        # Minimize
        print(f"Starting {args.method} minimization (maxfun={args.maxfun})...")
        t1 = time.time()

        dofs_out = np.zeros_like(dofs0)
        energies_out = np.full(n, np.nan, dtype=np.float64)
        nfev_out = np.zeros(n, dtype=np.int32)

        for i in range(n):
            ens_id = int(ens[i])
            dof0 = dofs0[i].copy()

            def func_and_grad(x, _ens_id=ens_id):
                e, g = oracle.score_single(_ens_id, x)
                return e, g.astype(np.float64)

            result = scipy_minimize(
                func_and_grad,
                dof0,
                method=args.method,
                jac=True,
                options={
                    "maxfun": args.maxfun,
                    "maxiter": args.maxfun,
                    "ftol": 1e-15,
                    "gtol": 1e-10,
                },
            )

            dofs_out[i] = result.x
            energies_out[i] = result.fun
            nfev_out[i] = result.nfev

            if args.trace_every and (i + 1) % args.trace_every == 0:
                elapsed = time.time() - t1
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i + 1}/{n}] nfev={result.nfev} "
                    f"energy={result.fun:.4f} success={result.success} "
                    f"({elapsed:.1f}s, {rate:.1f} poses/s, ETA {eta:.0f}s)"
                )

        t2 = time.time()

    # Summary
    print(f"\nMinimization done in {t2 - t1:.1f}s ({n} poses)")
    print(
        f"  nfev: mean={nfev_out.mean():.1f} median={np.median(nfev_out):.0f} "
        f"min={nfev_out.min()} max={nfev_out.max()}"
    )
    print(
        f"  energy: min={energies_out.min():.3f} mean={energies_out.mean():.3f} "
        f"p1={np.percentile(energies_out, 1):.3f} p10={np.percentile(energies_out, 10):.3f}"
    )
    print(
        f"  energy improvement: mean={np.mean(start_e - energies_out):.3f} "
        f"median={np.median(start_e - energies_out):.3f}"
    )

    # Resolve pivot for output
    if 2 in pivots:
        lig_pivot = pivots[2]
    else:
        # Read from ligand PDB
        coor = []
        with open(paths["ligand_pdb"]) as f:
            for line in f:
                if line.startswith("ATOM"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coor.append((x, y, z))
        lig_pivot = np.mean(coor, axis=0)

    mats = dofs_to_mats_np(dofs_out, lig_pivot)

    # Save
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
        _, piv_ref, ens_ref, dof_ref, e_ref, _ = parse_dat_two_body(
            args.legacy_dat, max_poses=n
        )
        m = min(n, len(e_ref))
        if np.isfinite(e_ref[:m]).all():
            summarize("energy_vs_legacy", e_ref[:m], energies_out[:m])
            k = max(1, m // 100)
            leg_sort = np.sort(e_ref[:m])
            jax_sort = np.sort(energies_out[:m])
            print(
                f"  Top 1% ({k} poses): legacy mean={leg_sort[:k].mean():.3f} "
                f"scipy mean={jax_sort[:k].mean():.3f}"
            )
            # Also compare at various percentile thresholds
            for pct in [1, 5, 10, 50]:
                k2 = max(1, m * pct // 100)
                print(
                    f"  Top {pct}% ({k2} poses): legacy={leg_sort[:k2].mean():.3f} "
                    f"scipy={jax_sort[:k2].mean():.3f} "
                    f"(delta={jax_sort[:k2].mean() - leg_sort[:k2].mean():.3f})"
                )

    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
