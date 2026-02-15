#!/usr/bin/env python3
"""Test merged-ensemble with threading explicitly disabled.
Sets XLA_FLAGS AFTER the oracle module override, to force single-thread."""
import os, sys, time
import numpy as np

# Force single-threaded BEFORE any jax import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
# Override what the oracle module does by re-setting AFTER import
sys.path.insert(0, "/home/sjoerd/attract-namespace/attract-jax/util")

# The oracle import will try to modify XLA_FLAGS.  We need to import jax
# first to lock in the flags.
import jax  # noqa: locks XLA_FLAGS

print(f"XLA_FLAGS after jax init: {os.environ.get('XLA_FLAGS', '')}")
print(f"Devices: {jax.devices()}")

from jax_scorer import JaxScoreOracle
from minfor import parse_dat_two_body

TEST_DIR = "/home/sjoerd/attract-namespace/test"

for n_poses in [30, 200, 500]:
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
    _, _, ens, dofs, _, _ = parse_dat_two_body(
        os.path.join(TEST_DIR, "systsearch-ens1.dat"), max_poses=n_poses
    )

    # Warmup
    e0, g0 = oracle.score_batch(ens, dofs)
    oracle._total_kernel_time = 0.0
    oracle._total_kernel_calls = 0

    times = []
    for i in range(30):
        dofs_t = dofs + np.random.normal(0, 0.01, dofs.shape)
        ts = time.monotonic()
        e, g = oracle.score_batch(ens, dofs_t)
        te = time.monotonic()
        times.append(te - ts)

    avg = np.mean(times) * 1000
    print(
        f"  {n_poses:4d} poses: {avg:.1f} ms/tick  →  {n_poses/np.mean(times):.1f} poses/s  "
        f"(1 thread, merged-ensemble)"
    )
