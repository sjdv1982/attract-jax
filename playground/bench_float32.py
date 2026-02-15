#!/usr/bin/env python3
"""Quick float32 test: disable x64, rebuild oracle, measure speedup."""
import os, sys, time
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

sys.path.insert(0, "/home/sjoerd/attract-namespace/attract-jax/util")

# BEFORE importing jax or the oracle, we need to not set jax_enable_x64.
# The oracle module sets it. We override.
import jax

# Don't enable x64 — let everything be float32
jax.config.update("jax_enable_x64", False)
print(f"x64 enabled: {jax.config.jax_enable_x64}")
print(f"Default float: {jax.numpy.array(1.0).dtype}")

from jax_scorer import JaxScoreOracle
from minfor import parse_dat_two_body

TEST_DIR = "/home/sjoerd/attract-namespace/test"

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

for n_poses in [30, 200, 500]:
    _, _, ens, dofs, _, _ = parse_dat_two_body(
        os.path.join(TEST_DIR, "systsearch-ens1.dat"), max_poses=n_poses
    )
    # Warmup
    e0, g0 = oracle.score_batch(ens, dofs)
    oracle._total_kernel_time = 0.0
    oracle._total_kernel_calls = 0

    print(f"\n--- {n_poses} poses (float32 mode) ---")
    print(f"  Energy dtype: {type(e0[0])}, sample: {e0[:3]}")
    print(f"  Grad dtype: {g0.dtype}")

    times = []
    for i in range(30):
        dofs_t = dofs + np.random.normal(0, 0.01, dofs.shape)
        ts = time.monotonic()
        e, g = oracle.score_batch(ens, dofs_t)
        te = time.monotonic()
        times.append(te - ts)

    avg = np.mean(times) * 1000
    print(f"  Per-tick: {avg:.1f} ms  →  {n_poses/np.mean(times):.1f} poses/s")
