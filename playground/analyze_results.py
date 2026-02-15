#!/usr/bin/env python3
"""Analyze minimize_custom results vs legacy minfor."""
import argparse
import sys
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from minfor import parse_dat_two_body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("our_prefix", help="e.g. test/custom_va13_first200")
    ap.add_argument("legacy_dat", help="legacy minfor .dat file")
    ap.add_argument("--max-poses", type=int, default=0)
    args = ap.parse_args()

    e_ours = np.load(args.our_prefix + ".energy.npy").astype(np.float64)
    nfev = np.load(args.our_prefix + ".nfev.npy")
    n = len(e_ours)

    _, _, _, _, e_legacy, _ = parse_dat_two_body(
        args.legacy_dat, max_poses=args.max_poses or n
    )
    m = min(n, len(e_legacy))
    e_ours = e_ours[:m]
    e_legacy = e_legacy[:m]
    nfev = nfev[:m]

    delta = e_ours - e_legacy

    print(f"=== {m}-pose VA13 vs Legacy minfor ===")
    print(f"Our mean energy:    {e_ours.mean():.3f}")
    print(f"Legacy mean energy: {e_legacy.mean():.3f}")
    print(f"Delta: mean={delta.mean():.3f} median={np.median(delta):.3f}")
    print()

    exact = np.sum(np.abs(delta) < 0.01)
    ours_better = np.sum(delta < -0.01)
    legacy_better = np.sum(delta > 0.01)
    print(f"Exact match (<0.01): {exact}/{m} ({100*exact/m:.0f}%)")
    print(f"Our minimizer better: {ours_better}/{m} ({100*ours_better/m:.0f}%)")
    print(f"Legacy better: {legacy_better}/{m} ({100*legacy_better/m:.0f}%)")
    print()

    for k in [1, 2, 3, 5, 10]:
        if k > m:
            break
        our_topk = np.sort(e_ours)[:k]
        leg_topk = np.sort(e_legacy)[:k]
        print(
            f"Top-{k}: ours={our_topk.mean():.3f} legacy={leg_topk.mean():.3f}"
            f" (delta={our_topk.mean()-leg_topk.mean():.3f})"
        )

    print()
    print(
        f"nfev: mean={nfev.mean():.1f} median={np.median(nfev):.0f} "
        f"min={nfev.min()} max={nfev.max()}"
    )
    print(f"Poses hitting maxfun=150: {np.sum(nfev >= 150)}/{m}")
    print()

    # Correlation
    corr = np.corrcoef(e_ours, e_legacy)[0, 1]
    print(f"Pearson correlation: {corr:.4f}")

    # RMSE
    rmse = np.sqrt(np.mean(delta**2))
    print(f"RMSE: {rmse:.3f}")

    # Energy distributions
    for pct in [1, 5, 10, 25, 50, 75, 90]:
        print(
            f"  p{pct}: ours={np.percentile(e_ours, pct):.3f} "
            f"legacy={np.percentile(e_legacy, pct):.3f}"
        )


if __name__ == "__main__":
    main()
