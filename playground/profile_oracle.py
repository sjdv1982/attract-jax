#!/usr/bin/env python3
"""Quick profiling of the JAX oracle to understand bottleneck split.

Runs 30 poses through 20 ticks of the batched minimizer, then 30 poses
with energy_batch=30 (single call per ensemble) to compare.
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
import time
import numpy as np


def profile(label, energy_batch, xla_flags, n_poses=30, n_ticks=20):
    """Run a mini minimization and report timing."""
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["OMP_NUM_THREADS"] = "1" if "false" in xla_flags else "10"

    # Must reload JAX with new flags (only works for first call realistically)
    # For subsequent calls, flags are already baked in.

    from jax_scorer import JaxScoreOracle

    test_dir = "/home/sjoerd/attract-namespace/test"
    oracle = JaxScoreOracle(
        receptor_ens_list=os.path.join(test_dir, "partner1-ensemble.list"),
        ligand_pdb=os.path.join(test_dir, "ligandr.pdb"),
        grid_file=os.path.join(test_dir, "receptorgrid.grid"),
        attract_par_npz="/home/sjoerd/attract-namespace/attract-jax/attract-par.npz",
        lig_pivot=np.array([34.915, 19.239, 0.784]),
        epsilon=15.0,
        energy_batch=energy_batch,
        max_nb_cap=40,
    )

    # Load poses
    sys.path.insert(0, "/home/sjoerd/attract-namespace/attract-jax/util")
    from minfor import parse_dat_two_body

    _, _, ens, dofs, _, _ = parse_dat_two_body(
        os.path.join(test_dir, "systsearch-ens1.dat"), max_poses=n_poses
    )

    # Warmup (first call triggers JIT compilation)
    print(f"\n=== {label} (energy_batch={energy_batch}) ===")
    print("Warmup JIT compilation...")
    t0 = time.monotonic()
    e, g = oracle.score_batch(ens, dofs)
    t1 = time.monotonic()
    print(f"  Warmup (incl JIT): {t1-t0:.2f}s")

    # Timed calls
    kernel_times = []
    total_times = []
    for tick in range(n_ticks):
        # Perturb dofs slightly (like a real minimization step)
        dofs_trial = dofs + np.random.normal(0, 0.01, dofs.shape)
        t_start = time.monotonic()
        e, g = oracle.score_batch(ens, dofs_trial)
        t_end = time.monotonic()
        total_times.append(t_end - t_start)

    # Report
    avg_total = np.mean(total_times) * 1000
    std_total = np.std(total_times) * 1000
    kernel_avg = oracle._total_kernel_time / oracle._total_kernel_calls * 1000
    n_kernel_calls = oracle._total_kernel_calls - 1  # exclude warmup
    calls_per_tick = n_kernel_calls / n_ticks

    print(f"  Per-tick wall time: {avg_total:.1f} ± {std_total:.1f} ms")
    print(f"  Kernel calls per tick: {calls_per_tick:.1f}")
    print(f"  Avg kernel call time: {kernel_avg:.1f} ms")
    print(
        f"  Total kernel time: {oracle._total_kernel_time:.3f}s "
        f"({oracle._total_kernel_calls} calls)"
    )
    print(
        f"  Poses/s estimate (from {n_poses} poses, {n_ticks} ticks): "
        f"{n_poses / (sum(total_times) / n_ticks):.1f}"
    )

    return avg_total


if __name__ == "__main__":
    # Test with current settings (single-thread, energy_batch=8)
    profile(
        "Baseline (1 thread, batch=8)",
        energy_batch=8,
        xla_flags="--xla_cpu_multi_thread_eigen=false",
        n_poses=30,
        n_ticks=20,
    )

    # Note: can't change XLA_FLAGS after JAX init.
    # For multi-thread test, run separately.
    print(
        "\n[NOTE: To test multi-threading, run with different XLA_FLAGS in a fresh process]"
    )
