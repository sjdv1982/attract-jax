#!/usr/bin/env python3
"""Build a cherry-picked test set: N poses from systsearch-ens1.dat
that include the K poses with lowest RMSD from the docking benchmark.

Usage:
    python build_cherrypicked_set.py \
        --sorted-dr-dat test/out_demo_xylanase-sorted-dr.dat \
        --lrmsd test/out_demo_xylanase-sorted-dr.lrmsd \
        --systsearch test/systsearch-ens1.dat \
        --total 5000 --cherry 50 \
        --out-dat test/systsearch-ens1-cherry5000.dat \
        --out-map test/cherry5000_map.npz
"""

import argparse
import re
import numpy as np
import sys


def parse_sort_indices_from_sorted_dr(dat_path):
    """Extract '=> sort' pose indices (1-based in scored/systsearch dat) from sorted-dr.dat."""
    indices = []
    with open(dat_path) as f:
        for line in f:
            m = re.match(r"^## (\d+) => sort", line)
            if m:
                indices.append(int(m.group(1)))
    return np.array(indices, dtype=np.int64)


def parse_lrmsd(lrmsd_path):
    """Parse l-RMSD values from an lrmsd file."""
    rmsds = []
    with open(lrmsd_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rmsds.append(float(parts[1]))
    return np.array(rmsds, dtype=np.float64)


def count_poses_in_dat(dat_path):
    """Count total poses in a dat file by counting #<number> lines."""
    count = 0
    with open(dat_path) as f:
        for line in f:
            if re.match(r"^#\d+\s*$", line):
                count += 1
    return count


def extract_poses_by_seed(systsearch_path, wanted_seeds, out_path):
    """Extract specific poses from systsearch-ens1.dat by their 1-based
    pose number (SEED), and also fill with consecutive poses from the start.

    wanted_seeds: set of 1-based pose indices to include
    Returns: list of (new_index, original_seed) tuples in output order
    """
    # We need to read the file twice: first to get the header, then to extract poses
    # Actually, we can do it in one pass if we stream

    # A pose in the dat file looks like:
    #   #<number>
    #   <ens_line>  (e.g. "  7 0.000 0.000 0.000 0.0000 0.0000 0.0000")
    #   <dof_line>  (e.g. "  0.000 0.000 0.000 0.0000 0.0000 0.0000")

    # Read header lines (before first #1)
    header_lines = []
    pose_blocks = {}  # seed -> list of lines (including #N line)

    current_seed = None
    current_lines = []

    with open(systsearch_path) as f:
        for line in f:
            if re.match(r"^#(\d+)\s*$", line):
                # Save previous pose
                if current_seed is not None:
                    pose_blocks[current_seed] = current_lines
                current_seed = int(re.match(r"^#(\d+)", line).group(1))
                current_lines = [line]
            elif current_seed is None:
                header_lines.append(line)
            else:
                current_lines.append(line)
        # Save last pose
        if current_seed is not None:
            pose_blocks[current_seed] = current_lines

    return header_lines, pose_blocks


def main():
    parser = argparse.ArgumentParser(description="Build cherry-picked pose set")
    parser.add_argument("--sorted-dr-dat", required=True, help="Path to sorted-dr.dat")
    parser.add_argument("--lrmsd", required=True, help="Path to sorted-dr.lrmsd")
    parser.add_argument(
        "--systsearch", required=True, help="Path to systsearch-ens1.dat"
    )
    parser.add_argument("--total", type=int, default=5000, help="Total poses in output")
    parser.add_argument(
        "--cherry", type=int, default=50, help="Number of low-RMSD poses to include"
    )
    parser.add_argument("--out-dat", required=True, help="Output dat file path")
    parser.add_argument("--out-map", required=True, help="Output mapping npz file")
    args = parser.parse_args()

    print(f"Parsing RMSD values from {args.lrmsd}...")
    rmsds = parse_lrmsd(args.lrmsd)
    print(
        f"  {len(rmsds)} RMSD values, min={rmsds.min():.3f}, median={np.median(rmsds):.3f}"
    )

    print(f"Parsing '=> sort' indices from {args.sorted_dr_dat}...")
    sort_indices = parse_sort_indices_from_sorted_dr(args.sorted_dr_dat)
    print(
        f"  {len(sort_indices)} poses, index range: {sort_indices.min()}-{sort_indices.max()}"
    )
    assert len(sort_indices) == len(
        rmsds
    ), f"Mismatch: {len(sort_indices)} indices vs {len(rmsds)} rmsds"

    # Find the cherry lowest-RMSD poses
    lowest_idx = np.argsort(rmsds)[: args.cherry]
    cherry_seeds = set(
        sort_indices[lowest_idx].tolist()
    )  # 1-based pose nums in systsearch
    cherry_rmsds = rmsds[lowest_idx]

    print(f"\n{args.cherry} lowest-RMSD poses:")
    print(f"  RMSD range: {cherry_rmsds.min():.3f} - {cherry_rmsds.max():.3f}")
    print(
        f"  Pose numbers (1-based): {sorted(cherry_seeds)[:10]}... (showing first 10)"
    )

    # Count total poses in systsearch
    print(f"\nCounting poses in {args.systsearch}...")
    total_in_systsearch = count_poses_in_dat(args.systsearch)
    print(f"  {total_in_systsearch} total poses")

    # Build the output set: cherry-picked seeds + fill from beginning
    # Strategy: take consecutive poses from start, then append any cherry
    # seeds that weren't in that range
    fill_seeds = []
    for i in range(1, total_in_systsearch + 1):
        if len(fill_seeds) + len(cherry_seeds - set(fill_seeds)) >= args.total:
            break
        fill_seeds.append(i)

    # Combine: fill_seeds + any cherry seeds not yet included
    output_seeds = list(fill_seeds)
    missing_cherry = cherry_seeds - set(output_seeds)
    output_seeds.extend(sorted(missing_cherry))

    # Trim to exact total if needed
    # Actually, let's be more precise: we want exactly args.total poses.
    # Include all cherry seeds, fill the rest from the start.
    output_seeds_set = set()
    output_seeds_ordered = []

    # First, add cherry seeds (they must be included)
    for s in sorted(cherry_seeds):
        output_seeds_set.add(s)
        output_seeds_ordered.append(s)

    # Fill from the start up to total
    for i in range(1, total_in_systsearch + 1):
        if len(output_seeds_ordered) >= args.total:
            break
        if i not in output_seeds_set:
            output_seeds_set.add(i)
            output_seeds_ordered.append(i)

    # Sort the final list so poses are in order (makes dat file cleaner)
    output_seeds_ordered.sort()

    print(f"\nOutput set: {len(output_seeds_ordered)} poses")
    print(f"  Cherry-picked: {len(cherry_seeds)} (all included)")
    n_overlap = len(cherry_seeds & set(range(1, args.total + 1)))
    print(f"  Cherry seeds in first {args.total}: {n_overlap}")
    print(
        f"  Extra cherry seeds beyond first {args.total}: {len(cherry_seeds) - n_overlap}"
    )

    # Create mapping arrays
    # is_cherry[i] = True if output_seeds_ordered[i] is a cherry-picked pose
    is_cherry = np.array([s in cherry_seeds for s in output_seeds_ordered], dtype=bool)
    original_seeds = np.array(output_seeds_ordered, dtype=np.int64)

    # Also store the RMSD for cherry-picked poses
    seed_to_rmsd = dict(zip(sort_indices[lowest_idx].tolist(), cherry_rmsds.tolist()))
    cherry_rmsd_in_output = np.array(
        [seed_to_rmsd.get(s, np.nan) for s in output_seeds_ordered]
    )

    print(f"\nReading and extracting poses from {args.systsearch}...")
    # Stream through the file, extracting only wanted poses
    wanted_set = set(output_seeds_ordered)
    header_lines = []
    extracted = {}  # seed -> lines
    current_seed = None
    current_lines = []

    with open(args.systsearch) as f:
        for line in f:
            m = re.match(r"^#(\d+)\s*$", line)
            if m:
                if current_seed is not None and current_seed in wanted_set:
                    extracted[current_seed] = current_lines
                current_seed = int(m.group(1))
                current_lines = [line]
            elif current_seed is None:
                header_lines.append(line)
            else:
                current_lines.append(line)
        if current_seed is not None and current_seed in wanted_set:
            extracted[current_seed] = current_lines

    print(f"  Extracted {len(extracted)} poses")
    assert len(extracted) == len(
        output_seeds_ordered
    ), f"Expected {len(output_seeds_ordered)} but got {len(extracted)}"

    # Write output dat file with renumbered poses
    print(f"Writing {args.out_dat}...")
    with open(args.out_dat, "w") as f:
        for line in header_lines:
            f.write(line)
        for new_idx, orig_seed in enumerate(output_seeds_ordered, 1):
            lines = extracted[orig_seed]
            # Replace the pose number
            f.write(f"#{new_idx}\n")
            for line in lines[1:]:  # skip original #N line
                f.write(line)

    # Save mapping
    print(f"Saving mapping to {args.out_map}...")
    np.savez(
        args.out_map,
        original_seeds=original_seeds,
        is_cherry=is_cherry,
        cherry_rmsd=cherry_rmsd_in_output,
        cherry_seed_list=np.array(sorted(cherry_seeds), dtype=np.int64),
    )

    print(f"\nDone! Output: {len(output_seeds_ordered)} poses in {args.out_dat}")
    print(f"  {is_cherry.sum()} cherry-picked (low-RMSD) poses")
    print(f"  {(~is_cherry).sum()} filler poses")

    # Print summary of cherry-picked poses
    cherry_indices_in_output = np.where(is_cherry)[0]
    print(
        f"\nCherry-picked pose indices in output (0-based): {cherry_indices_in_output[:20]}..."
    )
    print(
        f"Cherry-picked pose indices in output (1-based): {(cherry_indices_in_output + 1)[:20]}..."
    )


if __name__ == "__main__":
    main()
