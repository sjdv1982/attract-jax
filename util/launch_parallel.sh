#!/bin/bash
# Parallel minimization of all 165,528 poses using 6 single-threaded processes.
# Each process handles ~27,588 poses.
#
# Estimated wall clock: ~80-100 minutes on i5-1235U (10 cores, 30GB RAM)
# Memory per process: ~2GB → total ~12GB
set -e

NPROC=6
TOTAL=165528
CHUNK=$(( (TOTAL + NPROC - 1) / NPROC ))  # 27588

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DAT="/home/sjoerd/attract-namespace/test/systsearch-ens1.dat"
GRID="/home/sjoerd/attract-namespace/test/receptorgrid.grid"
PAR="/home/sjoerd/attract-namespace/attract-jax/attract-par.npz"
REC_LIST="/home/sjoerd/attract-namespace/test/partner1-ensemble.list"
LIG="/home/sjoerd/attract-namespace/test/ligandr.pdb"
OUT_DIR="/home/sjoerd/attract-namespace/test/jax_full_run"

mkdir -p "$OUT_DIR"

echo "=== Parallel minimization: $TOTAL poses in $NPROC chunks of $CHUNK ==="
echo "Output dir: $OUT_DIR"
echo ""

PIDS=()
for i in $(seq 0 $((NPROC - 1))); do
    OFFSET=$((i * CHUNK))
    PREFIX="$OUT_DIR/chunk_${i}"
    LOG="$OUT_DIR/chunk_${i}.log"

    echo "Launching chunk $i: offset=$OFFSET, max-poses=$CHUNK → $PREFIX"

    # Each process: single-threaded XLA, no memory preallocation
    env \
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
    conda run --no-capture-output -n jax python -u "$SCRIPT_DIR/minfor.py" \
        "$DAT" \
        --grid "$GRID" \
        --attract-par-npz "$PAR" \
        --receptor-ens-list "$REC_LIST" \
        --ligand-pdb "$LIG" \
        --oracle jax --epsilon 15.0 --energy-batch 8 --max-nb-cap 40 \
        --pose-offset "$OFFSET" --max-poses "$CHUNK" \
        --trace-every 20 \
        --out-prefix "$PREFIX" \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All $NPROC processes launched. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f $OUT_DIR/chunk_*.log"
echo ""

# Wait for all processes
FAIL=0
for i in $(seq 0 $((NPROC - 1))); do
    PID=${PIDS[$i]}
    if wait "$PID"; then
        echo "✓ Chunk $i (PID $PID) finished successfully"
    else
        EXIT_CODE=$?
        echo "✗ Chunk $i (PID $PID) failed with exit code $EXIT_CODE"
        FAIL=1
    fi
done

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "=== All chunks completed successfully ==="
    echo "Collecting results..."

    # Merge results with a small Python script
    conda run --no-capture-output -n jax python -u -c "
import numpy as np, os, sys

out_dir = '$OUT_DIR'
nproc = $NPROC

all_dofs, all_energy, all_ens, all_nfev = [], [], [], []
for i in range(nproc):
    prefix = os.path.join(out_dir, f'chunk_{i}')
    all_dofs.append(np.load(prefix + '.dofs.npy'))
    all_energy.append(np.load(prefix + '.energy.npy'))
    all_ens.append(np.load(prefix + '.ens.npy'))
    all_nfev.append(np.load(prefix + '.nfev.npy'))

dofs = np.concatenate(all_dofs, axis=0)
energy = np.concatenate(all_energy)
ens = np.concatenate(all_ens)
nfev = np.concatenate(all_nfev)

merged = os.path.join(out_dir, 'merged')
np.save(merged + '.dofs.npy', dofs)
np.save(merged + '.energy.npy', energy)
np.save(merged + '.ens.npy', ens)
np.save(merged + '.nfev.npy', nfev)

print(f'Merged {len(energy)} poses')
print(f'  energy: min={energy.min():.3f} mean={energy.mean():.3f} '
      f'p1={np.percentile(energy,1):.3f} p50={np.median(energy):.3f}')
print(f'  nfev:   mean={nfev.mean():.1f} median={np.median(nfev):.0f} '
      f'min={nfev.min()} max={nfev.max()}')

# Top 10 poses by energy
idx = np.argsort(energy)[:10]
print(f'  Top 10 energies: {energy[idx]}')
print(f'  Top 10 ensemble: {ens[idx]}')
print(f'Saved: {merged}.[dofs|energy|ens|nfev].npy')
"
else
    echo ""
    echo "=== Some chunks FAILED — check logs ==="
fi
