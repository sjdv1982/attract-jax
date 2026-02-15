#!/usr/bin/env python3
"""Quick mini-minimization benchmark: 200 poses, 50 maxfun.
Compares old vs new oracle speed in realistic minimizer conditions."""
import os, sys, time
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

sys.path.insert(0, "/home/sjoerd/attract-namespace/attract-jax/util")
from minfor import parse_dat_two_body, minfor_minimize_batched
from jax_scorer import JaxScoreOracle

TEST_DIR = "/home/sjoerd/attract-namespace/test"
N_POSES = 200
MAXFUN = 50

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

_, _, ens, dofs0, _, _ = parse_dat_two_body(
    os.path.join(TEST_DIR, "systsearch-ens1.dat"), max_poses=N_POSES
)

print(f"Minimizing {N_POSES} poses, maxfun={MAXFUN}")
print(f"Warmup (JIT compile)...")
# Warmup with a single score_batch call
e0, g0 = oracle.score_batch(ens, dofs0)
oracle._total_kernel_time = 0.0
oracle._total_kernel_calls = 0

t_start = time.monotonic()
x_best, f_best, nfev = minfor_minimize_batched(
    oracle, ens, dofs0, maxfun=MAXFUN, trace_every=10
)
t_end = time.monotonic()

elapsed = t_end - t_start
poses_per_s = N_POSES / elapsed
avg_nfev = nfev.mean()

print(f"\n{'='*60}")
print(f"Results: {elapsed:.1f}s for {N_POSES} poses = {poses_per_s:.2f} poses/s")
print(f"  avg nfev={avg_nfev:.0f}, median nfev={np.median(nfev):.0f}")
print(f"  energy: min={f_best.min():.3f} mean={f_best.mean():.3f}")
oracle.print_stats()
print(f"  Estimated full-run rate: ~{poses_per_s:.1f} poses/s")
