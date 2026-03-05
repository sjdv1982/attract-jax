# Milestone 4 Validation Report

**Implementation date:** 2026-03-04  
**Baseline:** Milestone 3 (`VALIDATION_M3.md`)  
**Plan target:** `plan-nonbonded-m1-m2.md` Milestone 4

---

## 1. Milestone 4 scope

Milestone 4 requires:

1. Capability-based wrapper emission from forcefield headers:
- `lj_grad.h` + `elec_grad.h` present => emit grad+energy wrappers.
- Either gradient header missing => emit energy-only wrappers.

2. Python symbol probing/dispatch selects the available wrapper family.

3. Remove remaining manual wrapper fallback in maintained paths.

4. Regression check: where both variants exist, energy-only scoring matches the energy component of grad+energy with strict tolerance.

---

## 2. Implementation summary

### 2.1 Capability-driven dispatch helpers

Updated `native/nb_kernel/forcefields/__init__.py`:

- Added `KernelDispatch` dataclass.
- Added `select_kernel_dispatch(available_symbols, rotation="euler")`.
- Added `bind_kernel_dispatch(lib, rotation="euler")`.

Behavior:
- Chooses `family="grad_energy"` when both `nb_kernel_euler_grad` and `nb_kernel_euler_energy` are present.
- Chooses `family="energy_only"` when only `nb_kernel_euler_energy` is present.
- Raises a runtime error when no callable wrapper exists.

### 2.2 Maintained runtime path now uses capability dispatch

Updated `util/jax_scorer.py` (`JaxScoreOracle._init_nonbon8_backend` and `_score_nb_nonbon8_batch`):

- Removed manual source-file fallback probing/compilation from ad-hoc `nb_kernel.cpp` paths.
- Uses generated `nb_kernel_nonbon8.so` from `native/nb_kernel` (`make nb_kernel_nonbon8.so` when missing).
- Uses `forcefields.bind_kernel_dispatch(...)` to choose wrapper family.
- If only energy wrapper is available, scorer now executes an explicit central finite-difference fallback over `nb_kernel_euler_energy` to produce NB gradients.

This removes the remaining manual wrapper fallback in the maintained nonbon8 runtime path.

### 2.3 Test expansion for Milestone 4

Extended `native/nb_kernel/tests/test_codegen_ff.py` with:

- `TestCodegenPartialGradHeaders`
  - Missing `lj_grad.h` only => no grad wrapper.
  - Missing `elec_grad.h` only => no grad wrapper.

- `TestKernelDispatchAndParity`
  - Dispatch selects `grad_energy` for `nb_kernel_nonbon8.so`.
  - Dispatch selects `energy_only` for a generated energy-only FF (`energyonly_dispatch`).
  - Direct ctypes invocation validates `nb_kernel_euler_energy` output equals `nb_kernel_euler_grad` energy output (`atol=1e-12`).

---

## 3. Validation evidence

Command run:

```bash
python -m unittest native.nb_kernel.tests.test_codegen_ff -v
```

Result:

- **82 tests run**
- **78 passed**
- **4 skipped** (expected JAX/environment-dependent skips)
- **0 failures / 0 errors**

Milestone-4-specific checks included in passing set:

- `test_missing_lj_grad_header_disables_grad_wrapper`
- `test_missing_elec_grad_header_disables_grad_wrapper`
- `test_dispatch_selects_grad_energy_when_both_symbols_exist`
- `test_dispatch_selects_energy_only_when_grad_symbol_missing`
- `test_energy_wrapper_matches_grad_energy_component`

---

## 4. Files changed

- `native/nb_kernel/forcefields/__init__.py`
- `util/jax_scorer.py`
- `native/nb_kernel/tests/test_codegen_ff.py`
- `VALIDATION_M4.md` (this report)

---

## 5. Conclusion

Milestone 4 is implemented and validated:

- Capability-based emission behavior is verified for both complete-grad and partial-grad-header scenarios.
- Python probing/dispatch now selects wrapper families explicitly.
- Maintained runtime path no longer depends on manual wrapper fallback sourcing.
- Energy-only wrapper output matches grad-wrapper energy output within strict tolerance when both are available.
