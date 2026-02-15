#!/usr/bin/env python3
"""Profile with multi-threading enabled."""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
# Don't limit OMP threads
if "OMP_NUM_THREADS" in os.environ:
    del os.environ["OMP_NUM_THREADS"]

import sys
import time
import numpy as np

sys.path.insert(0, "/home/sjoerd/attract-namespace/attract-jax/util")
from jax_scorer import JaxScoreOracle
from minfor import parse_dat_two_body

test_dir = "/home/sjoerd/attract-namespace/test"

for batch_size in [8, 30, 64]:
    oracle = JaxScoreOracle(
        receptor_ens_list=os.path.join(test_dir, "partner1-ensemble.list"),
        ligand_pdb=os.path.join(test_dir, "ligandr.pdb"),
        grid_file=os.path.join(test_dir, "receptorgrid.grid"),
        attract_par_npz="/home/sjoerd/attract-namespace/attract-jax/attract-par.npz",
        lig_pivot=np.array([34.915, 19.239, 0.784]),
        epsilon=15.0,
        energy_batch=batch_size,
        max_nb_cap=40,
    )

    _, _, ens, dofs, _, _ = parse_dat_two_body(
        os.path.join(test_dir, "systsearch-ens1.dat"), max_poses=30
    )

    print(f"\n=== Multi-threaded, energy_batch={batch_size} ===")
    # Warmup
    t0 = time.monotonic()
    e, g = oracle.score_batch(ens, dofs)
    t1 = time.monotonic()
    print(f"  Warmup: {t1-t0:.2f}s")
    oracle._total_kernel_time = 0.0
    oracle._total_kernel_calls = 0

    # Timed runs
    times = []
    for i in range(20):
        dofs_t = dofs + np.random.normal(0, 0.01, dofs.shape)
        ts = time.monotonic()
        e, g = oracle.score_batch(ens, dofs_t)
        te = time.monotonic()
        times.append(te - ts)

    avg = np.mean(times) * 1000
    kernel_avg = (
        oracle._total_kernel_time / oracle._total_kernel_calls * 1000
        if oracle._total_kernel_calls
        else 0
    )
    calls = oracle._total_kernel_calls / 20
    print(
        f"  Per-tick: {avg:.1f} ms  |  kernel/call: {kernel_avg:.1f} ms  |  calls/tick: {calls:.1f}"
    )
    print(f"  Poses/s: {30 / np.mean(times):.1f}")
