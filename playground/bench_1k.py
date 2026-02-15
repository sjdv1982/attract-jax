#!/usr/bin/env python3
"""Larger mini-minimize: 1000 poses, maxfun=50."""
import os, sys, time
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

sys.path.insert(0, "/home/sjoerd/attract-namespace/attract-jax/util")
from minfor import parse_dat_two_body, minfor_minimize_batched
from jax_scorer import JaxScoreOracle

TEST_DIR = "/home/sjoerd/attract-namespace/test"

for N, EB in [(200, 256), (1000, 256), (1000, 1024)]:
    oracle = JaxScoreOracle(
        receptor_ens_list=os.path.join(TEST_DIR, "partner1-ensemble.list"),
        ligand_pdb=os.path.join(TEST_DIR, "ligandr.pdb"),
        grid_file=os.path.join(TEST_DIR, "receptorgrid.grid"),
        attract_par_npz="/home/sjoerd/attract-namespace/attract-jax/attract-par.npz",
        lig_pivot=np.array([34.915, 19.239, 0.784]),
        epsilon=15.0,
        energy_batch=EB,
        max_nb_cap=40,
    )
    _, _, ens, dofs0, _, _ = parse_dat_two_body(
        os.path.join(TEST_DIR, "systsearch-ens1.dat"), max_poses=N
    )

    # Warmup
    oracle.score_batch(ens, dofs0)
    oracle._total_kernel_time = 0.0
    oracle._total_kernel_calls = 0

    t1 = time.monotonic()
    x_best, f_best, nfev = minfor_minimize_batched(
        oracle, ens, dofs0, maxfun=50, trace_every=0
    )
    t2 = time.monotonic()
    rate = N / (t2 - t1)
    kavg = oracle._total_kernel_time / oracle._total_kernel_calls * 1000

    print(
        f"N={N:5d} EB={EB:5d}: {t2-t1:.1f}s → {rate:.1f} poses/s "
        f"  (kernel: {oracle._total_kernel_calls} calls, avg {kavg:.0f}ms)"
    )
