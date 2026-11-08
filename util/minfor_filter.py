"""Stream score-only filtering to white/grey arrays of original pose indices."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def filter_poses(oracle, dofs, ens, conformers, *, threshold, tolerance,
                 white_output, grey_output, pose_indices=None, offset=0,
                 batch_size=8192, transform=None, precision_safe=False, report_path=None):
    """White: E < T-m. Grey: T-m <= E < T+m. Rejected poses are omitted.

    Optional precision safety keeps grid values with inadequate float64 spacing
    grey. The report records those grey positions as packed bits, allowing a
    cascade to reserve them for exact scoring without another pose-index file.
    """
    threshold, tolerance = float(threshold), float(tolerance)
    if not math.isfinite(threshold) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("filter threshold must be finite; tolerance must be finite and non-negative")
    if batch_size <= 0:
        raise ValueError("filter batch size must be positive")
    selection = None
    if pose_indices is not None:
        selection = np.load(pose_indices, mmap_mode="r", allow_pickle=False)
        if selection.ndim != 1 or not np.issubdtype(selection.dtype, np.integer):
            raise ValueError("pose indices must be a one-dimensional integer array")
        if np.any(selection < 0) or np.any(selection >= len(dofs)):
            raise ValueError("pose index outside input pose array")
    n = len(dofs) if selection is None else len(selection)
    whites, greys, precision_parts = [], [], []
    started = time.monotonic()
    report_every = max(batch_size, max(1, (n + 19) // 20))
    next_report = report_every
    white_count = grey_count = precision_count = 0
    with np.errstate(over="ignore", invalid="ignore"):
        threshold_ulp = abs(float(np.spacing(np.float64(abs(threshold)))))
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        gather = slice(start, stop) if selection is None else np.asarray(selection[start:stop], dtype=np.intp)
        batch = dofs[gather]
        if transform is not None:
            batch = transform(batch)
        energies, _ = oracle.score_batch(ens[gather], batch,
                                        conformers=None if conformers is None else conformers[gather])
        energies = np.asarray(energies, dtype=np.float64)
        if energies.shape != (stop - start,) or not np.all(np.isfinite(energies)):
            raise ValueError("filter scorer must return one finite float64 energy per pose")
        white = energies < threshold - tolerance
        grey = (energies >= threshold - tolerance) & (energies < threshold + tolerance)
        unsafe = np.zeros(len(energies), dtype=bool)
        if precision_safe:
            with np.errstate(over="ignore", invalid="ignore"):
                unsafe = (np.spacing(np.abs(energies)) >= tolerance / 2.0) | (threshold_ulp >= tolerance / 2.0)
            white &= ~unsafe
            grey |= unsafe
        ids = (np.arange(start + offset, stop + offset, dtype=np.uint64)
               if selection is None else np.asarray(selection[start:stop], dtype=np.uint64))
        whites.append(ids[white])
        greys.append(ids[grey])
        if precision_safe:
            precision_parts.append(unsafe[grey])
        white_count += int(np.count_nonzero(white))
        grey_count += int(np.count_nonzero(grey))
        precision_count += int(np.count_nonzero(unsafe))
        if stop >= next_report or stop == n:
            print(f"Filter progress: {stop}/{n} poses ({stop / n:.1%}) "
                  f"white={white_count} grey={grey_count} elapsed={time.monotonic() - started:.1f}s",
                  file=sys.stderr, flush=True)
            next_report = stop + report_every
    if n == 0:
        print("Filter progress: 0/0 poses (100.0%) white=0 grey=0", file=sys.stderr, flush=True)
    outputs = (np.concatenate(whites) if whites else np.empty(0, dtype=np.uint64),
               np.concatenate(greys) if greys else np.empty(0, dtype=np.uint64))
    staged = []
    try:
        for name, values in zip((white_output, grey_output), outputs):
            path = Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + ".partial")
            staged.append((partial, path))
            with partial.open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
        for partial, path in staged:
            partial.replace(path)
    finally:
        for partial, _ in staged:
            partial.unlink(missing_ok=True)
    report = {"schema": 1, "input_count": n, "white_count": white_count,
              "grey_count": grey_count, "precision_fallback_count": precision_count,
              "backend": "compiled", "threshold": threshold, "tolerance": tolerance}
    if precision_safe:
        flags = np.concatenate(precision_parts) if precision_parts else np.empty(0, dtype=bool)
        report["precision_fallback_bits"] = base64.b64encode(np.packbits(flags).tobytes()).decode("ascii")
    if report_path is not None:
        Path(report_path).write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
    return report
