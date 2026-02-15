#!/usr/bin/env python3
"""Benchmark the optimized merged-ensemble oracle.

Tests:
1. Warm-up JIT compilation time
2. Per-tick wall time for score_batch (the critical path for minimization)
3. Comparison at different pose counts (30, 200, 500)
"""
import os, sys, time
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# The oracle module now overrides --xla_cpu_multi_thread_eigen=false → true

sys.path.insert(0, "/home/sjoerd/attract-namespace/attract-jax/util")
from minfor import parse_dat_two_body

TEST_DIR = "/home/sjoerd/attract-namespace/test"


def run_benchmark(label, n_poses, n_ticks=30, energy_batch=256):
    """Run benchmark and report timing."""
    from jax_scorer import JaxScoreOracle

    oracle = JaxScoreOracle(
        receptor_ens_list=os.path.join(TEST_DIR, "partner1-ensemble.list"),
        ligand_pdb=os.path.join(TEST_DIR, "ligandr.pdb"),
        grid_file=os.path.join(TEST_DIR, "receptorgrid.grid"),
        attract_par_npz="/home/sjoerd/attract-namespace/attract-jax/attract-par.npz",
        lig_pivot=np.array([34.915, 19.239, 0.784]),
        epsilon=15.0,
        energy_batch=energy_batch,
        max_nb_cap=40,
    )

    _, _, ens, dofs, _, _ = parse_dat_two_body(
        os.path.join(TEST_DIR, "systsearch-ens1.dat"), max_poses=n_poses
    )

    print(f"\n{'='*60}")
    print(f"{label}  (n_poses={n_poses}, energy_batch={energy_batch})")
    print(f"  Ensembles: {np.unique(ens)}, poses/ens: ", end="")
    for e in np.unique(ens):
        print(f"{e}:{(ens==e).sum()} ", end="")
    print()

    # Warmup (first call triggers JIT)
    t0 = time.monotonic()
    e0, g0 = oracle.score_batch(ens, dofs)
    t_warmup = time.monotonic() - t0
    print(f"  Warmup (incl JIT): {t_warmup:.2f}s")

    # Reset stats after warmup
    oracle._total_kernel_time = 0.0
    oracle._total_kernel_calls = 0

    # Timed runs (simulate minimizer ticks)
    tick_times = []
    for i in range(n_ticks):
        dofs_trial = dofs + np.random.normal(0, 0.01, dofs.shape)
        ts = time.monotonic()
        e, g = oracle.score_batch(ens, dofs_trial)
        te = time.monotonic()
        tick_times.append(te - ts)

    avg_tick = np.mean(tick_times) * 1000
    std_tick = np.std(tick_times) * 1000
    p50 = np.percentile(tick_times, 50) * 1000
    kernel_total = oracle._total_kernel_time
    kernel_calls = oracle._total_kernel_calls
    kernel_avg = kernel_total / kernel_calls * 1000 if kernel_calls else 0
    calls_per_tick = kernel_calls / n_ticks
    poses_per_s = n_poses / np.mean(tick_times)

    print(f"  Per-tick: {avg_tick:.1f} ± {std_tick:.1f} ms  (p50={p50:.1f})")
    print(
        f"  Kernel calls/tick: {calls_per_tick:.1f}  avg kernel call: {kernel_avg:.1f} ms"
    )
    print(f"  ⟹ Poses/s: {poses_per_s:.1f}")

    # Verify correctness (energies should be finite and reasonable)
    assert np.all(np.isfinite(e0)), "Non-finite energies!"
    assert np.all(np.isfinite(g0)), "Non-finite gradients!"
    print(f"  Energy range: [{e0.min():.2f}, {e0.max():.2f}]")

    return poses_per_s


if __name__ == "__main__":
    # Test at different pose counts
    results = {}
    for n in [30, 200, 500]:
        r = run_benchmark(f"Merged-ensemble oracle", n_poses=n)
        results[n] = r

    print(f"\n{'='*60}")
    print("SUMMARY (merged-ensemble, multi-threaded):")
    for n, r in results.items():
        print(f"  {n:4d} poses → {r:.1f} poses/s")

    # Compare with baseline (single-thread, per-ensemble, batch=8)
    print(f"\nBaseline was ~144 poses/s (30 poses) → ~4.9 poses/s (165k full run)")
    print(f"Legacy single-thread: 47.0 poses/s")
