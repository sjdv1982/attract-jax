# Compiled energy filtering with minfor

`--filter` implies scoring and energy-only mode. It never minimizes poses and
requires the compiled backend (`--oracle jax --nb-kernel compiled`, the defaults).
It compares the float64 scorer results directly and writes two NPY arrays of
0-based uint64 pose indices:

- `--output-white`: energies strictly below `threshold - tolerance`.
- `--output-grey`: energies at or above `threshold - tolerance` and strictly
  below `threshold + tolerance`.

All other poses are rejected. Lower energies, including more negative energies,
are better. For threshold -16.59 and tolerance 1, white means E < -17.59; grey
means -17.59 <= E < -15.59. An energy of -15.59 is rejected.

```bash
python util/minfor.py \
  --filter --nb-kernel compiled \
  --input-rotvec poses.rotvec.npy --input-conformers poses.conformers.npy \
  --input-world-centered \
  --receptor-coordinates receptor-coor.npy \
  --receptor-atomtypes receptor-atomtypes.npy \
  --ligand-ensemble ligand-ensemble.npy --ligand-atomtypes ligand-atomtypes.npy \
  --attract-par-npz attract-original/attract-par.npz \
  --plateaudis 5 --neighbourdis 7 \
  --energy-threshold -16.59 --energy-tolerance 1 \
  --output-white white.npy --output-grey grey.npy
```

Use `--pose-indices grey.npy` on the next invocation to score that sparse subset
against the same original input arrays. The output retains the original source
indices and follows the selection's order. `--pose-indices` cannot be combined
with `--pose-offset` or `--max-poses`. Without an index selection, output indices
include any `--pose-offset` applied to the original input.

`--no-grid` evaluates all receptor-ligand pairs with the compiled nonbon8 kernel,
without building or loading any grid. Use it with `--energy-tolerance 0` for the
final cascade pass: white is strictly E < threshold, and grey is empty. The
NumPy gridless reference remains available for ordinary scoring, but filter mode
requires compiled scoring.

`--score-batch-size` controls scorer batches and progress updates on stderr.
Filtering writes indices only; `--output-npy` cannot be combined with `--filter`.
`--save-grid grid.npz` saves a just-in-time generated grid while continuing the
filter pass. Reuse it through `--grid grid.npz` on a later pass with the same
grid settings.

For Alaric's cascade, `--filter-precision-safe --filter-report report.json` also
keeps numerically unsafe grid values grey. The JSON contains counts and packed
bits identifying those grey positions, so Alaric can reserve them for the final
gridless pass. This preserves its float64 precision fallback and global
`min_cascade` counting while using only the two pose-index NPY outputs.
