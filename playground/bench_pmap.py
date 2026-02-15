#!/usr/bin/env python3
"""Test pmap-based parallelism: split batch across multiple CPU 'devices'."""
import os, sys, time
import numpy as np

N_DEVICES = 4

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={N_DEVICES} "
    "--xla_cpu_multi_thread_eigen=false"
)

sys.path.insert(0, "/home/sjoerd/attract-namespace/attract-jax/util")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

print(f"Devices: {jax.devices()}")

from reproduce_grid_score import build_kernel, parse_reduced_pdb, read_grid_with_electro
from minfor import parse_dat_two_body
from jax_scorer import JaxScoreOracle, _round_up_batch

TEST_DIR = "/home/sjoerd/attract-namespace/test"

# Build oracle to get all the data structures
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

# Build pmap'd value_and_grad
kernel_ad = oracle._kernel_main.ad


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


_vg_single = jax.value_and_grad(_single_energy)
# pmap over devices, vmap within each device
_vg_pmap = jax.pmap(
    jax.vmap(_vg_single, in_axes=(0, 0, 0)),
    in_axes=(0, 0, 0),
)


def score_batch_pmap(ens, dofs):
    n = len(dofs)
    ens0 = np.asarray(ens, dtype=np.intp) - 1
    rec_coor_np = oracle._rec_coor_ens_np[ens0]
    rec_charge_np = oracle._rec_charge_ens_np[ens0]

    # Pad to multiple of N_DEVICES
    per_dev = (n + N_DEVICES - 1) // N_DEVICES
    total = per_dev * N_DEVICES

    dofs_pad = np.zeros((total, 6), dtype=np.float64)
    rc_pad = np.zeros((total,) + rec_coor_np.shape[1:], dtype=np.float64)
    rq_pad = np.zeros((total,) + rec_charge_np.shape[1:], dtype=np.float64)
    dofs_pad[:n] = dofs
    rc_pad[:n] = rec_coor_np
    rq_pad[:n] = rec_charge_np

    # Reshape to (D, per_dev, ...)
    dofs_r = jnp.array(dofs_pad.reshape(N_DEVICES, per_dev, 6))
    rc_r = jnp.array(rc_pad.reshape(N_DEVICES, per_dev, *rec_coor_np.shape[1:]))
    rq_r = jnp.array(rq_pad.reshape(N_DEVICES, per_dev, *rec_charge_np.shape[1:]))

    e, g = _vg_pmap(dofs_r, rc_r, rq_r)
    e = np.asarray(e.reshape(-1)[:n])
    g = np.asarray(g.reshape(-1, 6)[:n])
    return e, g


# Load poses
for n_poses in [32, 200, 500]:
    _, _, ens, dofs, _, _ = parse_dat_two_body(
        os.path.join(TEST_DIR, "systsearch-ens1.dat"), max_poses=n_poses
    )
    print(f"\n--- {n_poses} poses, pmap D={N_DEVICES} ---")

    # Warmup
    t0 = time.monotonic()
    e0, g0 = score_batch_pmap(ens, dofs)
    t_warmup = time.monotonic() - t0
    print(f"  Warmup: {t_warmup:.2f}s")

    # Timed runs
    times = []
    for i in range(20):
        dofs_t = dofs + np.random.normal(0, 0.01, dofs.shape)
        ts = time.monotonic()
        e, g = score_batch_pmap(ens, dofs_t)
        te = time.monotonic()
        times.append(te - ts)

    avg = np.mean(times) * 1000
    print(f"  Per-tick: {avg:.1f} ms  →  {n_poses/np.mean(times):.1f} poses/s")

    # Verify correctness against standard oracle
    e_ref, g_ref = oracle.score_batch(ens, dofs)
    e_err = float(np.max(np.abs(e0 - e_ref)))
    g_err = float(np.max(np.abs(g0 - g_ref)))
    print(f"  Energy max-err: {e_err:.2e}, Grad max-err: {g_err:.2e}")
