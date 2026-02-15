# Legacy minfor Energy-Call Budget (Instrumentation)

## Scope
- Legacy executable: `/home/sjoerd/attract/bin/attract` with `ATTRACT_MINFOR_TRACE=1`.
- Input poses: first **200** structures from `test/systsearch-ens1.dat` (`test/systsearch-ens1-first200.dat`).
- Docking setup: same test grid/protocol (`--grid 1 receptorgrid.gridheader`, `--fix-receptor --ens 1 partner1-ensemble.list`).
- Metric counted: number of `MINFOR_REQUEST PRE` lines per pose.
- By construction: `total_calls = INIT + LINE + FINAL = line_calls + 2`.

## How It Was Captured
A sweep over `vmax in {1,2,5,10,20,50,100,200,500,1000}` was executed with legacy instrumentation enabled. Data files:
- `test/legacy_minfor_call_budget_first200.csv`
- `test/legacy_minfor_call_budget_first200.json` (per-pose raw counts)

## Summary Table (200 poses)

| vmax | mean total calls | p50 | p90 | p99 | max | mean line calls | frac(line==vmax) | wall time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.000 | 3.0 | 3.0 | 3.0 | 3 | 1.000 | 1.000 | 0.34 |
| 2 | 4.000 | 4.0 | 4.0 | 4.0 | 4 | 2.000 | 1.000 | 0.35 |
| 5 | 7.000 | 7.0 | 7.0 | 7.0 | 7 | 5.000 | 1.000 | 0.47 |
| 10 | 12.000 | 12.0 | 12.0 | 12.0 | 12 | 10.000 | 1.000 | 0.73 |
| 20 | 22.000 | 22.0 | 22.0 | 22.0 | 22 | 20.000 | 1.000 | 1.23 |
| 50 | 51.605 | 52.0 | 52.0 | 52.0 | 52 | 49.605 | 0.975 | 2.50 |
| 100 | 89.060 | 98.5 | 102.0 | 102.0 | 102 | 87.060 | 0.460 | 3.95 |
| 200 | 105.355 | 98.5 | 163.5 | 202.0 | 202 | 103.355 | 0.045 | 4.40 |
| 500 | 116.315 | 98.5 | 163.5 | 502.0 | 502 | 114.315 | 0.030 | 4.64 |
| 1000 | 131.315 | 98.5 | 163.5 | 1002.0 | 1002 | 129.315 | 0.030 | 5.07 |

## Observations
- For `vmax <= 20`, every pose in this sample uses the full line-search budget (`frac(line==vmax)=1.0`).
- At higher `vmax`, many poses exit early. On this sample:
  - `vmax=100`: mean line calls `87.060` (46% hit the cap).
  - `vmax=1000`: mean line calls `129.315` (3% hit the cap).
- In this dataset, increasing `vmax` from `500` to `1000` increases mean total calls from `116.315` to `131.315` (modest average increase, but with a long tail up to the cap).

## Practical Budget Numbers
- At mean `131.315` total calls/pose (`vmax=1000` on this sample), expected energy-call count is ~`1,313,150` for `10,000` poses.
- At mean `131.315` total calls/pose (`vmax=1000` on this sample), expected energy-call count is ~`13,131,500` for `100,000` poses.

## Supplemental Check (1000 poses)
A second instrumentation run was executed on `test/systsearch-ens1-first1000.dat` for `vmax={10,100,1000}`:

| vmax | nposes | mean total calls | p50 | p90 | p99 | max | frac(line==vmax) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1000 | 12.000 | 12.0 | 12.0 | 12.0 | 12 | 1.000 |
| 100 | 1000 | 89.239 | 96.0 | 102.0 | 102.0 | 102 | 0.432 |
| 1000 | 1000 | 111.857 | 96.0 | 148.0 | 612.9 | 1002 | 0.010 |

This confirms that `vmax=1000` average call budget can vary by workload slice, but still sits around `~1e2` calls/pose rather than `~1e3` calls/pose for this protocol.

## Caveats
- This is a **single workload slice** (first 200 start poses of `systsearch-ens1`). Other datasets/protocol stages can shift the distribution.
- `--vmax` is a line-search call budget; total calls include fixed `INIT` and `FINAL` requests.
