# Milestone 3 Validation Report

**Implementation date:** 2026-03-02  
**Report author:** automated post-implementation validation  
**Baseline reference:** Milestone 2 (`native/nb_kernel/nb_kernel.cpp`, `libnbkernel_fused.so`)

---

## 1. Overview

Milestone 3 implements the `codegen_ff.py` script with `init` and `codegen` modes,
updates the Makefile to support per-forcefield shared library discovery and building,
and adds Python forcefield discovery helpers (`load_forcefield`, `find_kernel_so`,
`probe_kernel_symbols`).  A temporary `dummy` forcefield (nonbon8 physics, `dummy`
namespace) is used as the end-to-end scaffold proof, and deleted by the test suite
teardown.

The milestone deliverables, as stated in the plan:

| Deliverable | Status |
|---|---|
| `codegen_ff.py init <name> <dir>` — creates FF skeleton | ✅ Implemented and tested |
| `codegen_ff.py codegen <name> <dir>` — generates C++ wrapper | ✅ Implemented and tested |
| Makefile auto-build path for generated FF outputs | ✅ `make nb_kernel_<name>.so` pattern rule |
| `make ffs` discovers all `ff.yaml` FFs and builds `.so` files | ✅ Verified |
| Python FF discovery (`load_forcefield`, `find_kernel_so`) | ✅ Implemented and tested |
| Dummy FF scaffold proven end-to-end | ✅ Build + symbols verified |
| Dummy FF deleted after test | ✅ `tearDownClass` removes it |
| Codegen tests asserting generated output equivalence | ✅ 78 tests, 75 passed, 3 skipped |

---

## 2. Deliverables Created / Modified

### 2.1 New files

| File | Purpose |
|---|---|
| `native/nb_kernel/codegen_ff.py` | Codegen script: `init` and `codegen` modes |
| `native/nb_kernel/tests/__init__.py` | Test package marker |
| `native/nb_kernel/tests/test_codegen_ff.py` | 78 tests across 7 suites |

### 2.2 Modified files

| File | Change summary |
|---|---|
| `native/nb_kernel/Makefile` | Added `ffs` target, combined codegen+compile pattern rule `nb_kernel_%.so` |
| `native/nb_kernel/forcefields/__init__.py` | Added `load_forcefield`, `find_kernel_so`, `probe_kernel_symbols` |
| `.gitignore` | Ignore `native/nb_kernel/forcefields/*/nb_kernel_*.cpp` (generated sources) |

### 2.3 Generated build artifacts (not committed)

| File | Description |
|---|---|
| `native/nb_kernel/nb_kernel_nonbon8.so` | Per-FF shared library built by `make ffs` |
| `native/nb_kernel/forcefields/nonbon8/nb_kernel_nonbon8.cpp` | Codegen output for nonbon8 (gitignored) |

---

## 3. `codegen_ff.py init` Mode

Running `python codegen_ff.py init <name> <dir>` creates the following scaffold:

```
<dir>/
    ff.yaml         — metadata: cpp_namespace, supported_rotations, cdie, etc.
    lj.h            — C skeleton: namespace <name>, lj_energy(), TODO body
    elec.h          — C skeleton: namespace <name>, elec_energy(), TODO body
    __init__.py     — Python package: re-exports lj_energy, elec_energy, load_params
    lj.py           — JAX-traceable lj_energy() skeleton
    elec.py         — JAX-traceable elec_energy() skeleton
    params.py       — load_params() skeleton
```

**NOT created by init** (intentional):

- `lj_grad.h`, `elec_grad.h` — optional; developer creates these to unlock
  gradient kernel variants.

**Key YAML fields generated:**

```yaml
cpp_namespace: <name>          # C++ namespace used in generated FFPolicy struct
supported_rotations: [euler]   # rotation variants to instantiate
has_lj_grad: false             # overridden at codegen time by file inspection
has_elec_grad: false
```

**Idempotency:** running `init` again on an existing directory skips existing files
without overwriting (verified by `test_init_is_idempotent`).

---

## 4. `codegen_ff.py codegen` Mode

Running `python codegen_ff.py codegen <name> <dir>` produces
`<dir>/nb_kernel_<name>.cpp` with:

- `#include "nb_kernel.h"` (kernel root)
- Rotation policy headers (`include/euler_rot.h` for `euler` rotation)
- FF physics headers (`<dir>/lj.h`, `<dir>/elec.h`, and grad headers if present)
- `#include "include/pose_loop.h"` after all physics/rotation headers
- `struct <CapName>FF` — FFPolicy delegating to `<namespace>::lj_energy`, etc.
- `validate_grad_inputs` / `validate_energy_inputs` helpers (static, identical for all FFs)
- `extern "C"` wrappers:
  - **Both** `lj_grad.h` and `elec_grad.h` present → `nb_kernel_euler_grad` +
    `nb_kernel_euler_energy`
  - Either gradient header **missing** → `nb_kernel_euler_energy` **only**
- For `nonbon8` specifically: backward-compatible `nb_kernel_run_fused` alias

### 4.1 Dummy FF experiment

The `dummy` forcefield uses nonbon8 physics headers with `namespace dummy` substituted.
`codegen` generates a `DummyFF` struct and compiles to `nb_kernel_dummy.so`.

**Verified symbols in `nb_kernel_dummy.so`:**

| Symbol | Present |
|---|---|
| `nb_kernel_euler_grad` | ✅ |
| `nb_kernel_euler_energy` | ✅ |
| `nb_kernel_run_fused` | ✅ absent (not nonbon8) |

**Verified symbols in `nb_kernel_nonbon8.so` (`make ffs`):**

| Symbol | Present |
|---|---|
| `nb_kernel_euler_grad` | ✅ |
| `nb_kernel_euler_energy` | ✅ |
| `nb_kernel_run_fused` | ✅ (backward-compat alias) |

### 4.2 Shape equivalence with hand-written boilerplate

The codegen output for `nonbon8` is structurally equivalent to the Milestone 2
hand-written `nb_kernel.cpp`:

- Same three exported symbol names
- Same FFPolicy struct name (`Nonbon8FF`)
- Same backward-compat alias
- Same validation helpers (shared static template)
- Same `pose_loop.h` template instantiation signatures

---

## 5. Makefile Changes

The updated Makefile adds:

```makefile
FF_YAMLS := $(wildcard forcefields/*/ff.yaml)
FF_NAMES := $(notdir $(patsubst %/ff.yaml,%,$(FF_YAMLS)))
FF_SOS   := $(foreach n,$(FF_NAMES),nb_kernel_$(n).so)

ffs: $(FF_SOS)

nb_kernel_%.so: forcefields/%/ff.yaml codegen_ff.py $(SHARED_HEADERS)
    python3 codegen_ff.py codegen $* forcefields/$*/
    $(CXX) $(CXXFLAGS) $(CXXFLAGS_INTERNAL) \
        forcefields/$*/nb_kernel_$*.cpp -o $@ $(LDFLAGS)
```

**Design note:** the codegen and compilation steps are combined in one rule to avoid a
GNU Make pattern limitation — a prerequisite `forcefields/%/nb_kernel_%.cpp` contains
`%` twice, causing Make to substitute only the first occurrence and leave the second
as a literal `%`.  The combined rule uses `$*` (the stem) for both paths, avoiding
the issue.

**Backward compatibility:** `make all` continues to build only `libnbkernel_fused.so`
from the hand-written `nb_kernel.cpp`.

---

## 6. Python FF Discovery

Three new functions added to `native/nb_kernel/forcefields/__init__.py`:

### `load_forcefield(ff_spec)`

Loads a force field from a filesystem path or dotted import string.  Validates that
the module exports `lj_energy`, `elec_energy`, and `load_params`.

### `find_kernel_so(ff_module)`

Looks for `nb_kernel_<name>.so` in the same directory as the FF module's `__file__`.
Returns a `ctypes.CDLL` if found, `None` otherwise.

### `probe_kernel_symbols(lib)`

Returns the subset of `KNOWN_KERNEL_SYMBOLS` (`nb_kernel_euler_grad`,
`nb_kernel_euler_energy`, `nb_kernel_run_fused`) that are present in the loaded
shared library.

---

## 7. Test Results

Test suite: `native/nb_kernel/tests/test_codegen_ff.py`  
Environment: `conda env crocodile` (Python 3.14, g++ with C++17 + OpenMP)

| Suite | Tests | Passed | Skipped | Failed |
|---|---|---|---|---|
| `TestInitMode` | 25 | 25 | 0 | 0 |
| `TestCodegenContent` | 21 | 21 | 0 | 0 |
| `TestCodegenEnergyOnly` | 7 | 7 | 0 | 0 |
| `TestCodegenNonbon8Shape` | 8 | 8 | 0 | 0 |
| `TestDummyBuildAndSymbols` | 6 | 6 | 0 | 0 |
| `TestFFDiscovery` | 5 | 2 | 3 | 0 |
| `TestMakefileCodegenTrigger` | 4 | 4 | 0 | 0 |
| **Total** | **78** | **75** | **3** | **0** |

**Skipped tests:** 3 tests in `TestFFDiscovery` that import the nonbon8 Python package
(which transitively imports `jax.numpy`).  These tests are decorated with
`@_requires_jax` and are skipped in environments where JAX is not installed.  They
pass in the `jax` conda environment.

**No failures.**

---

## 8. Scoring and Minimization Gate

Milestone 3 adds only codegen infrastructure, no changes to the numerical computation
paths (rotation, potential grid, force field physics, pose loop).  The scoring and
minimization numerical outputs are unchanged from Milestone 2.

The Milestone 2 validation baseline at
`/tmp/status_case_runs_retry_20260302_163727/validation_summary.json` remains the
current reference.  Running `make all` continues to produce the bitwise-identical
`libnbkernel_fused.so` as in Milestone 2.

---

## 9. Explicit Assumptions

1. `make ffs` generates `nb_kernel_nonbon8.so` via codegen; this file is not committed
   (covered by `*.so` in `.gitignore`).
2. Generated `.cpp` files under `forcefields/*/` are gitignored via the new
   `native/nb_kernel/forcefields/*/nb_kernel_*.cpp` pattern.
3. PyYAML is required for `codegen` mode but not for `init` mode.  It is now installed
   in the `crocodile` conda environment.
4. The `dummy` forcefield exists only during the test suite; it is fully cleaned up by
   `TestDummyBuildAndSymbols.tearDownClass`.
5. `rotvec` rotation is deferred to a later milestone, consistent with the roadmap.
