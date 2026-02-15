#!/usr/bin/env python3
"""Compare pmap D=2,4,6 for finding optimal device count."""
import os, sys, time
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.insert(0, "/home/sjoerd/attract-namespace/attract-jax/util")
TEST_DIR = "/home/sjoerd/attract-namespace/test"

for n_dev in [1, 2, 4]:
    # Must set flags before importing jax, so we fork a subprocess
    pass

# Actually, xla_force_host_platform_device_count can't be changed after jax init.
# Test just D=4 (already running) and D=2 by restarting.

# For D=2, need a separate run. Just test the key scenarios.

# Quick test: what's the overhead of pmap reshaping?
n_dev = int(os.environ.get("PMAP_DEVICES", "4"))
os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={n_dev} "
    "--xla_cpu_multi_thread_eigen=false"
)

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax_scorer import JaxScoreOracle
from minfor import parse_dat_two_body

oracle = JaxScoreOracle(
    receptor_ens_list=os.path.join(TEST_DIR, "partner1-ensemble.list"),
    ligand_pdb=os.path.join(TEST_DIR, "ligandr.pdb"),
    grid_file=os.path.join(TEST_DIR, "receptorgrid.grid"),
    attract_par_npz="/home/sjoerd/attract-namespace/attract-jax/attract-par.npz",
    lig_pivot=np.array([34.915, 19.239, 0.784]),
    epsilon=15.0,
    energy_batch=256,
    max_nb_cap=40,
)

kernel_ad = oracle._kernel_main.ad


def mk_single_energy():
    def _single_energy(dof_1d, rec_coor, rec_charge_scaled):
        dof_2d = dof_1d[None, :]
        _, per_pose = kernel_ad(
            dof_2d,
            rec_coor,
            oracle._rec_atomtypes_ff_j,
            rec_charge_scaled,
            oracle._coor_lig_j,
            oracle._lig_atomtypes_ff_j,
            oracle._lig_vdw_channel_idx_j,
            oracle._lig_charge_raw_j,
            oracle._lig_charge_scaled_j,
            oracle._ff,
            oracle._grid_j,
            oracle._lig_pivot_j,
        )
        return per_pose[0]

    return _single_energy


_se = mk_single_energy()
_vg = jax.value_and_grad(_se)

if n_dev > 1:
    _vg_fn = jax.pmap(jax.vmap(_vg, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
    label = f"pmap D={n_dev}"
else:
    _vg_fn = jax.jit(jax.vmap(_vg, in_axes=(0, 0, 0)))
    label = "vmap-only"


def score_fn(ens, dofs):
    n = len(dofs)
    ens0 = np.asarray(ens, dtype=np.intp) - 1
    rc = oracle._rec_coor_ens_np[ens0]
    rq = oracle._rec_charge_ens_np[ens0]

    if n_dev > 1:
        per_dev = (n + n_dev - 1) // n_dev
        total = per_dev * n_dev
        dp = np.zeros((total, 6), np.float64)
        dp[:n] = dofs
        rcp = np.zeros((total,) + rc.shape[1:], np.float64)
        rcp[:n] = rc
        rqp = np.zeros((total,) + rq.shape[1:], np.float64)
        rqp[:n] = rq
        d = jnp.array(dp.reshape(n_dev, per_dev, 6))
        r = jnp.array(rcp.reshape(n_dev, per_dev, *rc.shape[1:]))
        q = jnp.array(rqp.reshape(n_dev, per_dev, *rq.shape[1:]))
        e, g = _vg_fn(d, r, q)
        return np.asarray(e.reshape(-1)[:n]), np.asarray(g.reshape(-1, 6)[:n])
    else:
        from jax_scorer import _round_up_batch

        pad = _round_up_batch(n)
        d = jnp.array(dofs)
        r = jnp.array(rc)
        q = jnp.array(rq)
        if n < pad:
            d = jnp.pad(d, ((0, pad - n), (0, 0)))
            r = jnp.pad(r, ((0, pad - n), (0, 0), (0, 0)))
            q = jnp.pad(q, ((0, pad - n), (0, 0)))
        e, g = _vg_fn(d, r, q)
        return np.asarray(e[:n]), np.asarray(g[:n])


for n_poses in [200, 500]:
    _, _, ens, dofs, _, _ = parse_dat_two_body(
        os.path.join(TEST_DIR, "systsearch-ens1.dat"), max_poses=n_poses
    )
    # Warmup
    score_fn(ens, dofs)

    times = []
    for _ in range(20):
        dt = dofs + np.random.normal(0, 0.01, dofs.shape)
        ts = time.monotonic()
        score_fn(ens, dt)
        te = time.monotonic()
        times.append(te - ts)

    avg = np.mean(times) * 1000
    print(
        f"{label}, {n_poses} poses: {avg:.1f} ms/tick → {n_poses/np.mean(times):.1f} poses/s"
    )
