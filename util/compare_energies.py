#!/usr/bin/env python3
"""Compare two energy arrays with docking-oriented summary metrics."""

import argparse
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="reference energies (.npy)")
    parser.add_argument("candidate", help="candidate energies (.npy)")
    parser.add_argument("--topk", type=int, default=100, help="top-K low-energy overlap")
    parser.add_argument(
        "--truncate-to-min",
        action="store_true",
        help="if shapes differ, compare only the first min(len(reference), len(candidate)) values",
    )
    parser.add_argument("--out-json", help="optional JSON output")
    args = parser.parse_args()

    ref = np.load(args.reference).astype(np.float64)
    cand = np.load(args.candidate).astype(np.float64)
    if ref.shape != cand.shape:
        if not args.truncate_to_min:
            raise ValueError(f"shape mismatch: {ref.shape} vs {cand.shape}")
        n = min(len(ref), len(cand))
        ref = ref[:n]
        cand = cand[:n]

    delta = cand - ref
    q = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    def spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    topk = min(args.topk, len(ref))
    ref_top = set(np.argsort(ref)[:topk].tolist())
    cand_top = set(np.argsort(cand)[:topk].tolist())

    metrics = {
        "n": int(len(ref)),
        "ref_min": float(ref.min()),
        "cand_min": float(cand.min()),
        "ref_median": float(np.median(ref)),
        "cand_median": float(np.median(cand)),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "pearson": float(np.corrcoef(ref, cand)[0, 1]),
        "spearman": spearman(ref, cand),
        "delta_quantiles": {str(qq): float(np.percentile(delta, qq)) for qq in q},
        "ref_quantiles": {str(qq): float(np.percentile(ref, qq)) for qq in q},
        "cand_quantiles": {str(qq): float(np.percentile(cand, qq)) for qq in q},
        "topk": int(topk),
        "topk_overlap": float(len(ref_top & cand_top) / topk),
    }

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
