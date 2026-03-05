"""
test_codegen_ff.py — Tests for codegen_ff.py (Milestone 3).

Tests cover:
  1. 'init' mode: correct files created, with expected namespace and YAML structure.
  2. 'codegen' mode: generated .cpp has correct structure, symbols, and headers.
  3. 'codegen' capability detection: grad wrappers emitted iff both *_grad.h present.
  3a. Milestone 4a: correction/clamp plateau mode wrappers are emitted.
  4. Full build test: dummy FF (nonbon8 physics, dummy namespace) builds to a working .so
     with the expected extern "C" symbols.
  5. Python FF discovery: load_forcefield / find_kernel_so / probe_kernel_symbols
     work for the built dummy .so.

Teardown removes the dummy FF directory and .so after the full suite.
"""

import ctypes
import importlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# JAX availability guard (some discovery tests load nonbon8 which imports jax)
# ---------------------------------------------------------------------------
try:
    import jax  # noqa: F401

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

_requires_jax = unittest.skipUnless(
    _JAX_AVAILABLE, "jax not installed in this environment"
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KERNEL_ROOT = Path(__file__).parent.parent.resolve()
CODEGEN_SCRIPT = KERNEL_ROOT / "codegen_ff.py"
FF_DIR = KERNEL_ROOT / "forcefields"
NONBON8_DIR = FF_DIR / "nonbon8"
DUMMY_DIR = FF_DIR / "dummy"
DUMMY_SO = KERNEL_ROOT / "nb_kernel_dummy.so"

PYTHON = sys.executable


# ---------------------------------------------------------------------------
# ctypes structures for direct wrapper-call validation
# ---------------------------------------------------------------------------
class NbFusedStepData(ctypes.Structure):
    _fields_ = [
        ("nposes", ctypes.c_int32),
        ("natoms", ctypes.c_int32),
        ("dofs", ctypes.POINTER(ctypes.c_double)),
        ("ens", ctypes.POINTER(ctypes.c_int16)),
        ("lig_coords", ctypes.POINTER(ctypes.c_double)),
        ("lig_pivot", ctypes.POINTER(ctypes.c_double)),
        ("lig_type", ctypes.POINTER(ctypes.c_int16)),
        ("lig_charge", ctypes.POINTER(ctypes.c_double)),
    ]


class NbFusedGridData(ctypes.Structure):
    _fields_ = [
        ("dim", ctypes.c_int32 * 3),
        ("origin", ctypes.c_double * 3),
        ("spacing", ctypes.c_double),
        ("nr_neigh", ctypes.POINTER(ctypes.c_int32)),
        ("nb_start", ctypes.POINTER(ctypes.c_int64)),
    ]


class NbGlobalData(ctypes.Structure):
    _fields_ = [
        ("nrec", ctypes.c_int32),
        ("nens", ctypes.c_int32),
        ("nb_concat_len", ctypes.c_int64),
        ("nb_concat", ctypes.POINTER(ctypes.c_int32)),
        ("rec_coord", ctypes.POINTER(ctypes.c_double)),
        ("rec_type", ctypes.POINTER(ctypes.c_int16)),
        ("rec_charge", ctypes.POINTER(ctypes.c_double)),
        ("nrec_types", ctypes.c_int32),
        ("nlig_types", ctypes.c_int32),
        ("rc", ctypes.POINTER(ctypes.c_double)),
        ("ac", ctypes.POINTER(ctypes.c_double)),
        ("emin", ctypes.POINTER(ctypes.c_double)),
        ("rmin2", ctypes.POINTER(ctypes.c_double)),
        ("ivor", ctypes.POINTER(ctypes.c_int8)),
        ("plateaudissq", ctypes.c_double),
    ]


class NbRunConfig(ctypes.Structure):
    _fields_ = [("num_threads", ctypes.c_int32), ("kernel_variant", ctypes.c_int32)]


def _as_ptr(arr, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


# ---------------------------------------------------------------------------
# Helper: run codegen_ff.py as subprocess
# ---------------------------------------------------------------------------
def run_codegen(mode: str, name: str, directory: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PYTHON, str(CODEGEN_SCRIPT), mode, name, directory],
        capture_output=True,
        text=True,
        cwd=str(KERNEL_ROOT),
    )


# ---------------------------------------------------------------------------
# Helper: copy nonbon8 physics .h files to dummy/ with namespace substitution
# ---------------------------------------------------------------------------
def _copy_nonbon8_headers_with_namespace(dst_dir: Path, namespace: str) -> None:
    """Copy nonbon8 .h files into dst_dir, substituting namespace to *namespace*."""
    ns_upper = namespace.upper()
    for fname in ("lj.h", "elec.h", "lj_grad.h", "elec_grad.h"):
        src = NONBON8_DIR / fname
        if not src.exists():
            continue
        text = src.read_text()
        # Rename both the namespace declaration and calls within it.
        text = text.replace("namespace nonbon8", f"namespace {namespace}")
        text = text.replace("nonbon8::", f"{namespace}::")
        # Update include guards
        text = text.replace("NONBON8_", f"{ns_upper}_")
        (dst_dir / fname).write_text(text)


def _copy_nonbon8_headers_to_dummy(dst_dir: Path) -> None:
    _copy_nonbon8_headers_with_namespace(dst_dir, namespace="dummy")


# ===========================================================================
# Test suite 1: 'init' mode
# ===========================================================================
class TestInitMode(unittest.TestCase):
    """Tests for 'codegen_ff.py init' — skeleton creation."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="test_codegen_init_"))
        cls.name = "testff"
        cls.ff_dir = cls.tmpdir / cls.name
        result = run_codegen("init", cls.name, str(cls.ff_dir))
        cls.result = result

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    # ---- basic success ----

    def test_init_exits_zero(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)

    def test_init_creates_ff_yaml(self):
        self.assertTrue((self.ff_dir / "ff.yaml").exists())

    def test_init_creates_lj_h(self):
        self.assertTrue((self.ff_dir / "lj.h").exists())

    def test_init_creates_elec_h(self):
        self.assertTrue((self.ff_dir / "elec.h").exists())

    def test_init_creates_python_init(self):
        self.assertTrue((self.ff_dir / "__init__.py").exists())

    def test_init_creates_lj_py(self):
        self.assertTrue((self.ff_dir / "lj.py").exists())

    def test_init_creates_elec_py(self):
        self.assertTrue((self.ff_dir / "elec.py").exists())

    def test_init_creates_params_py(self):
        self.assertTrue((self.ff_dir / "params.py").exists())

    def test_init_does_not_create_grad_headers(self):
        """Gradient headers are optional; init must NOT create them."""
        self.assertFalse((self.ff_dir / "lj_grad.h").exists())
        self.assertFalse((self.ff_dir / "elec_grad.h").exists())

    # ---- C header content ----

    def test_lj_h_uses_ff_namespace(self):
        text = (self.ff_dir / "lj.h").read_text()
        self.assertIn(f"namespace {self.name}", text)

    def test_elec_h_uses_ff_namespace(self):
        text = (self.ff_dir / "elec.h").read_text()
        self.assertIn(f"namespace {self.name}", text)

    def test_lj_h_has_lj_energy_signature(self):
        text = (self.ff_dir / "lj.h").read_text()
        self.assertIn("lj_energy", text)

    def test_elec_h_has_elec_energy_signature(self):
        text = (self.ff_dir / "elec.h").read_text()
        self.assertIn("elec_energy", text)

    def test_lj_h_has_include_guard(self):
        text = (self.ff_dir / "lj.h").read_text()
        self.assertIn("#ifndef", text)
        self.assertIn("#define", text)

    # ---- YAML content ----

    def test_ff_yaml_has_name_field(self):
        import yaml

        cfg = yaml.safe_load((self.ff_dir / "ff.yaml").read_text())
        self.assertEqual(cfg["name"], self.name)

    def test_ff_yaml_has_cpp_namespace(self):
        import yaml

        cfg = yaml.safe_load((self.ff_dir / "ff.yaml").read_text())
        self.assertIn("cpp_namespace", cfg)
        self.assertEqual(cfg["cpp_namespace"], self.name)

    def test_ff_yaml_has_supported_rotations(self):
        import yaml

        cfg = yaml.safe_load((self.ff_dir / "ff.yaml").read_text())
        self.assertIn("supported_rotations", cfg)
        self.assertIn("euler", cfg["supported_rotations"])

    def test_ff_yaml_grad_flags_default_false(self):
        import yaml

        cfg = yaml.safe_load((self.ff_dir / "ff.yaml").read_text())
        self.assertFalse(cfg.get("has_lj_grad", True))
        self.assertFalse(cfg.get("has_elec_grad", True))

    # ---- Python package ----

    def test_python_init_exports_lj_energy(self):
        text = (self.ff_dir / "__init__.py").read_text()
        self.assertIn("lj_energy", text)

    def test_python_init_exports_elec_energy(self):
        text = (self.ff_dir / "__init__.py").read_text()
        self.assertIn("elec_energy", text)

    def test_python_init_exports_load_params(self):
        text = (self.ff_dir / "__init__.py").read_text()
        self.assertIn("load_params", text)

    def test_python_lj_py_has_lj_energy_def(self):
        text = (self.ff_dir / "lj.py").read_text()
        self.assertIn("def lj_energy", text)

    def test_python_elec_py_has_elec_energy_def(self):
        text = (self.ff_dir / "elec.py").read_text()
        self.assertIn("def elec_energy", text)

    def test_python_params_py_has_load_params_def(self):
        text = (self.ff_dir / "params.py").read_text()
        self.assertIn("def load_params", text)

    def test_init_is_idempotent(self):
        """Running init a second time must skip existing files (not overwrite)."""
        # Note the timestamp of ff.yaml before re-running
        yaml_mtime = (self.ff_dir / "ff.yaml").stat().st_mtime
        run_codegen("init", self.name, str(self.ff_dir))
        self.assertEqual((self.ff_dir / "ff.yaml").stat().st_mtime, yaml_mtime)


# ===========================================================================
# Test suite 2: 'codegen' mode — content checks (no build)
# ===========================================================================
class TestCodegenContent(unittest.TestCase):
    """Tests for 'codegen_ff.py codegen' output content."""

    @classmethod
    def setUpClass(cls):
        if DUMMY_DIR.exists():
            shutil.rmtree(DUMMY_DIR)
        # Create the dummy FF skeleton
        run_codegen("init", "dummy", str(DUMMY_DIR))
        # Copy nonbon8 physics headers with namespace substitution
        _copy_nonbon8_headers_to_dummy(DUMMY_DIR)
        # Run codegen
        cls.result = run_codegen("codegen", "dummy", str(DUMMY_DIR))
        cls.cpp_path = DUMMY_DIR / "nb_kernel_dummy.cpp"
        cls.cpp_text = cls.cpp_path.read_text() if cls.cpp_path.exists() else ""

    @classmethod
    def tearDownClass(cls):
        # Clean up generated .cpp but keep directory for the build test suite
        if cls.cpp_path.exists():
            cls.cpp_path.unlink()

    def test_codegen_exits_zero(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)

    def test_codegen_creates_cpp_file(self):
        self.assertTrue(self.cpp_path.exists())

    # ---- Required includes ----

    def test_generated_includes_nb_kernel_h(self):
        self.assertIn('#include "nb_kernel.h"', self.cpp_text)

    def test_generated_includes_euler_rot_h(self):
        self.assertIn('#include "include/euler_rot.h"', self.cpp_text)

    def test_generated_includes_pose_loop_h(self):
        self.assertIn('#include "include/pose_loop.h"', self.cpp_text)

    def test_generated_includes_lj_h(self):
        self.assertIn("forcefields/dummy/lj.h", self.cpp_text)

    def test_generated_includes_elec_h(self):
        self.assertIn("forcefields/dummy/elec.h", self.cpp_text)

    def test_generated_includes_lj_grad_h(self):
        """lj_grad.h was copied from nonbon8, so grad path should be active."""
        self.assertIn("forcefields/dummy/lj_grad.h", self.cpp_text)

    def test_generated_includes_elec_grad_h(self):
        self.assertIn("forcefields/dummy/elec_grad.h", self.cpp_text)

    # ---- FFPolicy struct ----

    def test_ffpolicy_struct_present(self):
        self.assertIn("struct DummyFF", self.cpp_text)

    def test_ffpolicy_delegates_lj_energy(self):
        self.assertIn("dummy::lj_energy", self.cpp_text)

    def test_ffpolicy_delegates_elec_energy(self):
        self.assertIn("dummy::elec_energy", self.cpp_text)

    def test_ffpolicy_delegates_lj_grad(self):
        self.assertIn("dummy::lj_grad", self.cpp_text)

    def test_ffpolicy_delegates_elec_grad(self):
        self.assertIn("dummy::elec_grad", self.cpp_text)

    # ---- Exported wrappers ----

    def test_grad_wrapper_present(self):
        self.assertIn("nb_kernel_euler_grad", self.cpp_text)

    def test_energy_wrapper_present(self):
        self.assertIn("nb_kernel_euler_energy", self.cpp_text)

    def test_correction_grad_wrapper_present(self):
        self.assertIn("nb_kernel_euler_correction_grad", self.cpp_text)

    def test_correction_energy_wrapper_present(self):
        self.assertIn("nb_kernel_euler_correction_energy", self.cpp_text)

    def test_clamp_grad_wrapper_present(self):
        self.assertIn("nb_kernel_euler_clamp_grad", self.cpp_text)

    def test_clamp_energy_wrapper_present(self):
        self.assertIn("nb_kernel_euler_clamp_energy", self.cpp_text)

    def test_wrappers_are_extern_c(self):
        self.assertIn('extern "C"', self.cpp_text)

    def test_grad_wrapper_uses_euler_rot(self):
        self.assertIn("EulerRot", self.cpp_text)

    def test_grad_wrapper_instantiates_compute_grad_true(self):
        self.assertIn("DummyFF, true, PlateauMode::", self.cpp_text)

    def test_energy_wrapper_instantiates_compute_grad_false(self):
        self.assertIn("DummyFF, false, PlateauMode::", self.cpp_text)

    def test_validation_helpers_present(self):
        self.assertIn("validate_grad_inputs", self.cpp_text)
        self.assertIn("validate_energy_inputs", self.cpp_text)

    def test_no_backward_compat_alias_for_dummy(self):
        """nb_kernel_run_fused alias is not emitted (removed in nonbon8 renaming)."""
        self.assertNotIn("nb_kernel_run_fused", self.cpp_text)

    def test_generated_file_has_do_not_edit_comment(self):
        self.assertIn("Do not edit manually", self.cpp_text)


# ===========================================================================
# Test suite 3: energy-only emission when gradient headers are absent
# ===========================================================================
class TestCodegenEnergyOnly(unittest.TestCase):
    """Verify energy-only wrappers when grad headers are missing."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="test_codegen_energy_only_"))
        cls.name = "energyonlyff"
        cls.ff_dir = cls.tmpdir / cls.name
        run_codegen("init", cls.name, str(cls.ff_dir))
        # Do NOT copy grad headers — energy-only scenario
        cls.result = run_codegen("codegen", cls.name, str(cls.ff_dir))
        cls.cpp_path = cls.ff_dir / f"nb_kernel_{cls.name}.cpp"
        cls.cpp_text = cls.cpp_path.read_text() if cls.cpp_path.exists() else ""

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_codegen_exits_zero(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)

    def test_energy_wrapper_present(self):
        self.assertIn("nb_kernel_euler_energy", self.cpp_text)
        self.assertIn("nb_kernel_euler_correction_energy", self.cpp_text)
        self.assertIn("nb_kernel_euler_clamp_energy", self.cpp_text)

    def test_grad_wrapper_absent(self):
        """No grad wrapper when either grad header is missing."""
        self.assertNotIn("nb_kernel_euler_grad", self.cpp_text)
        self.assertNotIn("nb_kernel_euler_correction_grad", self.cpp_text)
        self.assertNotIn("nb_kernel_euler_clamp_grad", self.cpp_text)

    def test_no_lj_grad_include(self):
        self.assertNotIn("lj_grad.h", self.cpp_text)

    def test_no_elec_grad_include(self):
        self.assertNotIn("elec_grad.h", self.cpp_text)

    def test_ffpolicy_has_no_lj_grad(self):
        self.assertNotIn("lj_grad", self.cpp_text)

    def test_energy_wrapper_instantiates_compute_grad_false(self):
        self.assertIn("false, PlateauMode::", self.cpp_text)


# ===========================================================================
# Test suite 3b: either missing grad header => energy-only wrappers
# ===========================================================================
class TestCodegenPartialGradHeaders(unittest.TestCase):
    """Either missing gradient header must disable grad wrappers."""

    def _run_partial_case(self, case_name: str, keep_lj_grad: bool, keep_elec_grad: bool):
        tmpdir = Path(tempfile.mkdtemp(prefix=f"test_codegen_partial_{case_name}_"))
        ff_dir = tmpdir / case_name
        try:
            run_codegen("init", case_name, str(ff_dir))
            # Copy energy headers (always required).
            shutil.copy(NONBON8_DIR / "lj.h", ff_dir / "lj.h")
            shutil.copy(NONBON8_DIR / "elec.h", ff_dir / "elec.h")
            if keep_lj_grad:
                shutil.copy(NONBON8_DIR / "lj_grad.h", ff_dir / "lj_grad.h")
            if keep_elec_grad:
                shutil.copy(NONBON8_DIR / "elec_grad.h", ff_dir / "elec_grad.h")

            result = run_codegen("codegen", case_name, str(ff_dir))
            self.assertEqual(result.returncode, 0, result.stderr)
            cpp_text = (ff_dir / f"nb_kernel_{case_name}.cpp").read_text()
            self.assertIn("nb_kernel_euler_energy", cpp_text)
            self.assertIn("nb_kernel_euler_correction_energy", cpp_text)
            self.assertIn("nb_kernel_euler_clamp_energy", cpp_text)
            self.assertNotIn("nb_kernel_euler_grad", cpp_text)
            self.assertNotIn("nb_kernel_euler_correction_grad", cpp_text)
            self.assertNotIn("nb_kernel_euler_clamp_grad", cpp_text)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_missing_lj_grad_header_disables_grad_wrapper(self):
        self._run_partial_case(
            case_name="missing_lj_grad",
            keep_lj_grad=False,
            keep_elec_grad=True,
        )

    def test_missing_elec_grad_header_disables_grad_wrapper(self):
        self._run_partial_case(
            case_name="missing_elec_grad",
            keep_lj_grad=True,
            keep_elec_grad=False,
        )


# ===========================================================================
# Test suite 4: nonbon8 codegen — struct shape matches hand-written boilerplate
# ===========================================================================
class TestCodegenNonbon8Shape(unittest.TestCase):
    """codegen on nonbon8 must reproduce the same wrapper shape as nb_kernel.cpp."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="test_codegen_nonbon8_"))
        cls.out_dir = cls.tmpdir / "nonbon8"
        cls.out_dir.mkdir()
        # Copy ff.yaml from the real nonbon8 directory
        shutil.copy(NONBON8_DIR / "ff.yaml", cls.out_dir / "ff.yaml")
        # Copy all .h files
        for h in NONBON8_DIR.glob("*.h"):
            shutil.copy(h, cls.out_dir / h.name)
        # Run codegen
        cls.result = run_codegen("codegen", "nonbon8", str(cls.out_dir))
        cls.cpp_path = cls.out_dir / "nb_kernel_nonbon8.cpp"
        cls.cpp_text = cls.cpp_path.read_text() if cls.cpp_path.exists() else ""
        # Reference: the real hand-written file
        cls.ref_text = (KERNEL_ROOT / "nb_kernel.cpp").read_text()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_codegen_exits_zero(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)

    def test_generated_has_euler_grad_symbol(self):
        self.assertIn("nb_kernel_euler_grad", self.cpp_text)

    def test_generated_has_euler_energy_symbol(self):
        self.assertIn("nb_kernel_euler_energy", self.cpp_text)

    def test_generated_has_plateau_mode_symbols(self):
        for sym in (
            "nb_kernel_euler_correction_grad",
            "nb_kernel_euler_correction_energy",
            "nb_kernel_euler_clamp_grad",
            "nb_kernel_euler_clamp_energy",
        ):
            self.assertIn(sym, self.cpp_text)

    def test_reference_has_same_symbols(self):
        """The reference nb_kernel.cpp must contain both canonical symbols."""
        for sym in ("nb_kernel_euler_grad", "nb_kernel_euler_energy"):
            self.assertIn(sym, self.ref_text)

    def test_struct_name_matches(self):
        self.assertIn("struct Nonbon8FF", self.cpp_text)

    def test_generated_includes_backward_compat_lj_grad_h(self):
        self.assertIn("lj_grad.h", self.cpp_text)

    def test_generated_includes_backward_compat_elec_grad_h(self):
        self.assertIn("elec_grad.h", self.cpp_text)


# ===========================================================================
# Test suite 5: full build + symbol probe (dummy FF)
# ===========================================================================
class TestDummyBuildAndSymbols(unittest.TestCase):
    """End-to-end: build dummy FF .so and probe extern C symbols.

    This test requires:
      - g++ with C++17 and OpenMP
      - 'make' available
      - The kernel has been built at least once (libnbkernel_nonbon8.so exists,
        confirming the build environment works)

    The dummy FF directory DUMMY_DIR is set up in TestCodegenContent.setUpClass.
    After the full suite its .so and directory are cleaned up by tearDownClass.
    """

    @classmethod
    def setUpClass(cls):
        cls.skip_reason = None
        cls.build_ok = False

        # Ensure the dummy FF directory is set up with real physics headers.
        if not DUMMY_DIR.exists() or not (DUMMY_DIR / "lj.h").exists():
            if DUMMY_DIR.exists():
                shutil.rmtree(DUMMY_DIR)
            run_codegen("init", "dummy", str(DUMMY_DIR))
            _copy_nonbon8_headers_to_dummy(DUMMY_DIR)

        # Remove any stale .so to force Make to rebuild.
        if DUMMY_SO.exists():
            DUMMY_SO.unlink()

        # 'make nb_kernel_dummy.so' handles codegen + compilation via the
        # combined pattern rule (Milestone 3 Makefile: codegen_ff.py codegen
        # is invoked as part of the .so recipe).
        make_result = subprocess.run(
            ["make", "nb_kernel_dummy.so"],
            capture_output=True,
            text=True,
            cwd=str(KERNEL_ROOT),
        )
        if make_result.returncode != 0:
            cls.skip_reason = f"make failed:\n{make_result.stderr}"
            return

        cls.build_ok = True

    @classmethod
    def tearDownClass(cls):
        """Remove dummy FF directory and .so after all tests in this suite."""
        if DUMMY_SO.exists():
            DUMMY_SO.unlink()
        if DUMMY_DIR.exists():
            shutil.rmtree(DUMMY_DIR)

    def _skip_if_no_build(self):
        if self.skip_reason:
            self.skipTest(self.skip_reason)

    def test_so_file_created(self):
        self._skip_if_no_build()
        self.assertTrue(DUMMY_SO.exists(), f"{DUMMY_SO} not found after make")

    def test_so_is_loadable_by_ctypes(self):
        self._skip_if_no_build()
        lib = ctypes.CDLL(str(DUMMY_SO))
        self.assertIsNotNone(lib)

    def test_symbol_euler_grad_present(self):
        self._skip_if_no_build()
        lib = ctypes.CDLL(str(DUMMY_SO))
        sym = lib.nb_kernel_euler_grad
        self.assertIsNotNone(sym)

    def test_symbol_euler_energy_present(self):
        self._skip_if_no_build()
        lib = ctypes.CDLL(str(DUMMY_SO))
        sym = lib.nb_kernel_euler_energy
        self.assertIsNotNone(sym)

    def test_symbol_plateau_mode_wrappers_present(self):
        self._skip_if_no_build()
        lib = ctypes.CDLL(str(DUMMY_SO))
        self.assertIsNotNone(lib.nb_kernel_euler_correction_grad)
        self.assertIsNotNone(lib.nb_kernel_euler_correction_energy)
        self.assertIsNotNone(lib.nb_kernel_euler_clamp_grad)
        self.assertIsNotNone(lib.nb_kernel_euler_clamp_energy)

    def test_no_run_fused_alias_in_dummy_so(self):
        """nb_kernel_run_fused alias no longer exists in any .so."""
        self._skip_if_no_build()
        lib = ctypes.CDLL(str(DUMMY_SO))
        try:
            _ = lib.nb_kernel_run_fused
            found = True
        except AttributeError:
            found = False
        self.assertFalse(found)

    def test_probe_kernel_symbols_returns_expected_set(self):
        self._skip_if_no_build()
        sys.path.insert(0, str(KERNEL_ROOT))
        from forcefields import probe_kernel_symbols

        lib = ctypes.CDLL(str(DUMMY_SO))
        syms = probe_kernel_symbols(lib)
        self.assertIn("nb_kernel_euler_correction_grad", syms)
        self.assertIn("nb_kernel_euler_correction_energy", syms)
        self.assertIn("nb_kernel_euler_clamp_grad", syms)
        self.assertIn("nb_kernel_euler_clamp_energy", syms)
        self.assertIn("nb_kernel_euler_grad", syms)
        self.assertIn("nb_kernel_euler_energy", syms)
        self.assertNotIn("nb_kernel_run_fused", syms)


# ===========================================================================
# Test suite 6: Python FF discovery helpers
# ===========================================================================
class TestFFDiscovery(unittest.TestCase):
    """Tests for load_forcefield, find_kernel_so, probe_kernel_symbols."""

    @classmethod
    def setUpClass(cls):
        # Ensure the forcefields package root is on sys.path
        if str(KERNEL_ROOT) not in sys.path:
            sys.path.insert(0, str(KERNEL_ROOT))
        from forcefields import load_forcefield, find_kernel_so, probe_kernel_symbols

        cls.load_forcefield = staticmethod(load_forcefield)
        cls.find_kernel_so = staticmethod(find_kernel_so)
        cls.probe_kernel_symbols = staticmethod(probe_kernel_symbols)

    @_requires_jax
    def test_load_nonbon8_by_import_path(self):
        ff = self.load_forcefield("forcefields.nonbon8")
        self.assertTrue(hasattr(ff, "lj_energy"))
        self.assertTrue(hasattr(ff, "elec_energy"))
        self.assertTrue(hasattr(ff, "load_params"))

    @_requires_jax
    def test_load_nonbon8_by_filesystem_path(self):
        ff = self.load_forcefield(str(NONBON8_DIR))
        self.assertTrue(hasattr(ff, "lj_energy"))

    def test_load_missing_ff_raises_import_error(self):
        with self.assertRaises((ImportError, ModuleNotFoundError)):
            self.load_forcefield("forcefields.doesnotexist")

    @_requires_jax
    def test_find_kernel_so_returns_none_when_absent(self):
        """Before building, no .so should be found for nonbon8 (unless it already exists)."""
        ff = self.load_forcefield("forcefields.nonbon8")
        so_path = NONBON8_DIR / "nb_kernel_nonbon8.so"
        result = self.find_kernel_so(ff)
        if so_path.exists():
            self.assertIsNotNone(result)
        else:
            self.assertIsNone(result)

    def test_probe_symbols_on_libnbkernel_nonbon8(self):
        """The main .so must contain the canonical nonbon8 symbols."""
        legacy_so = KERNEL_ROOT / "libnbkernel_nonbon8.so"
        if not legacy_so.exists():
            self.skipTest("libnbkernel_nonbon8.so not built yet")
        lib = ctypes.CDLL(str(legacy_so))
        syms = self.probe_kernel_symbols(lib)
        self.assertIn("nb_kernel_euler_grad", syms)
        self.assertIn("nb_kernel_euler_energy", syms)


# ===========================================================================
# Test suite 6b: capability-based dispatch + wrapper consistency
# ===========================================================================
class TestKernelDispatchAndParity(unittest.TestCase):
    """Milestone 4: select wrapper family and validate grad-vs-energy consistency."""

    @classmethod
    def setUpClass(cls):
        if str(KERNEL_ROOT) not in sys.path:
            sys.path.insert(0, str(KERNEL_ROOT))
        from forcefields import bind_kernel_dispatch, probe_kernel_symbols

        cls.bind_kernel_dispatch = staticmethod(bind_kernel_dispatch)
        cls.probe_kernel_symbols = staticmethod(probe_kernel_symbols)

        # Build a grad-capable library (nonbon8).
        subprocess.run(
            ["make", "nb_kernel_nonbon8.so"],
            capture_output=True,
            text=True,
            cwd=str(KERNEL_ROOT),
            check=True,
        )
        cls.nonbon8_so = KERNEL_ROOT / "nb_kernel_nonbon8.so"
        cls.nonbon8_lib = ctypes.CDLL(str(cls.nonbon8_so))

        # Build an energy-only FF to validate dispatch downgrade.
        cls.energyonly_name = "energyonly_dispatch"
        cls.energyonly_dir = FF_DIR / cls.energyonly_name
        if cls.energyonly_dir.exists():
            shutil.rmtree(cls.energyonly_dir)
        run_codegen("init", cls.energyonly_name, str(cls.energyonly_dir))
        _copy_nonbon8_headers_with_namespace(
            cls.energyonly_dir, namespace=cls.energyonly_name
        )
        # Intentionally do not copy *_grad.h files.
        for grad_h in ("lj_grad.h", "elec_grad.h"):
            p = cls.energyonly_dir / grad_h
            if p.exists():
                p.unlink()
        subprocess.run(
            ["make", f"nb_kernel_{cls.energyonly_name}.so"],
            capture_output=True,
            text=True,
            cwd=str(KERNEL_ROOT),
            check=True,
        )
        cls.energyonly_so = KERNEL_ROOT / f"nb_kernel_{cls.energyonly_name}.so"
        cls.energyonly_lib = ctypes.CDLL(str(cls.energyonly_so))

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "energyonly_so") and cls.energyonly_so.exists():
            cls.energyonly_so.unlink()
        if hasattr(cls, "energyonly_dir") and cls.energyonly_dir.exists():
            shutil.rmtree(cls.energyonly_dir, ignore_errors=True)

    def test_dispatch_selects_grad_energy_when_both_symbols_exist(self):
        dispatch, grad_fn, energy_fn = self.bind_kernel_dispatch(self.nonbon8_lib)
        self.assertEqual(dispatch.family, "grad_energy")
        self.assertEqual(dispatch.plateau_mode, "correction")
        self.assertEqual(dispatch.grad_symbol, "nb_kernel_euler_correction_grad")
        self.assertEqual(dispatch.energy_symbol, "nb_kernel_euler_correction_energy")
        self.assertFalse(dispatch.uses_legacy_alias)
        self.assertIsNotNone(grad_fn)
        self.assertIsNotNone(energy_fn)

    def test_dispatch_selects_energy_only_when_grad_symbol_missing(self):
        dispatch, grad_fn, energy_fn = self.bind_kernel_dispatch(self.energyonly_lib)
        self.assertEqual(dispatch.family, "energy_only")
        self.assertEqual(dispatch.plateau_mode, "correction")
        self.assertIsNone(dispatch.grad_symbol)
        self.assertEqual(dispatch.energy_symbol, "nb_kernel_euler_correction_energy")
        self.assertFalse(dispatch.uses_legacy_alias)
        self.assertIsNone(grad_fn)
        self.assertIsNotNone(energy_fn)

    def test_dispatch_can_select_clamp_mode(self):
        dispatch, grad_fn, energy_fn = self.bind_kernel_dispatch(
            self.nonbon8_lib, plateau_mode="clamp"
        )
        self.assertEqual(dispatch.family, "grad_energy")
        self.assertEqual(dispatch.plateau_mode, "clamp")
        self.assertEqual(dispatch.grad_symbol, "nb_kernel_euler_clamp_grad")
        self.assertEqual(dispatch.energy_symbol, "nb_kernel_euler_clamp_energy")
        self.assertIsNotNone(grad_fn)
        self.assertIsNotNone(energy_fn)

    def test_energy_wrapper_matches_grad_energy_component(self):
        """For grad-capable kernels, energy-only output must equal grad path energy."""
        lib = self.nonbon8_lib
        fn_grad = lib.nb_kernel_euler_correction_grad
        fn_energy = lib.nb_kernel_euler_correction_energy
        for fn in (fn_grad, fn_energy):
            fn.argtypes = [
                ctypes.POINTER(NbFusedStepData),
                ctypes.POINTER(NbFusedGridData),
                ctypes.POINTER(NbGlobalData),
                ctypes.POINTER(NbRunConfig),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
            fn.restype = ctypes.c_int

        dofs = np.zeros((1, 6), dtype=np.float64)
        ens = np.array([0], dtype=np.int16)
        lig_coords = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        lig_pivot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        lig_type = np.array([0], dtype=np.int16)
        lig_charge = np.array([0.0], dtype=np.float64)

        dim = (ctypes.c_int32 * 3)(1, 1, 1)
        origin = (ctypes.c_double * 3)(0.0, 0.0, 0.0)
        nr_neigh = np.array([1], dtype=np.int32)
        nb_start = np.array([0], dtype=np.int64)
        nb_concat = np.array([0], dtype=np.int32)
        rec_coord = np.array([2.0, 0.0, 0.0], dtype=np.float64)
        rec_type = np.array([0], dtype=np.int16)
        rec_charge = np.array([0.0], dtype=np.float64)
        rc = np.array([1.0], dtype=np.float64)
        ac = np.array([1.0], dtype=np.float64)
        emin = np.array([-27.0 / 256.0], dtype=np.float64)
        rmin2 = np.array([4.0 / 3.0], dtype=np.float64)
        ivor = np.array([1], dtype=np.int8)

        step = NbFusedStepData(
            nposes=np.int32(1),
            natoms=np.int32(1),
            dofs=_as_ptr(dofs.reshape(-1), ctypes.c_double),
            ens=_as_ptr(ens, ctypes.c_int16),
            lig_coords=_as_ptr(lig_coords, ctypes.c_double),
            lig_pivot=_as_ptr(lig_pivot, ctypes.c_double),
            lig_type=_as_ptr(lig_type, ctypes.c_int16),
            lig_charge=_as_ptr(lig_charge, ctypes.c_double),
        )
        grid = NbFusedGridData(
            dim=dim,
            origin=origin,
            spacing=ctypes.c_double(1.0),
            nr_neigh=_as_ptr(nr_neigh, ctypes.c_int32),
            nb_start=_as_ptr(nb_start, ctypes.c_int64),
        )
        global_data = NbGlobalData(
            nrec=np.int32(1),
            nens=np.int32(1),
            nb_concat_len=np.int64(1),
            nb_concat=_as_ptr(nb_concat, ctypes.c_int32),
            rec_coord=_as_ptr(rec_coord, ctypes.c_double),
            rec_type=_as_ptr(rec_type, ctypes.c_int16),
            rec_charge=_as_ptr(rec_charge, ctypes.c_double),
            nrec_types=np.int32(1),
            nlig_types=np.int32(1),
            rc=_as_ptr(rc, ctypes.c_double),
            ac=_as_ptr(ac, ctypes.c_double),
            emin=_as_ptr(emin, ctypes.c_double),
            rmin2=_as_ptr(rmin2, ctypes.c_double),
            ivor=_as_ptr(ivor, ctypes.c_int8),
            plateaudissq=ctypes.c_double(9.0),
        )
        cfg = NbRunConfig(num_threads=np.int32(1), kernel_variant=np.int32(1))

        e_grad = np.zeros((1,), dtype=np.float64)
        g_grad = np.zeros((1, 6), dtype=np.float64)
        e_energy = np.zeros((1,), dtype=np.float64)

        rc_grad = fn_grad(
            ctypes.byref(step),
            ctypes.byref(grid),
            ctypes.byref(global_data),
            ctypes.byref(cfg),
            _as_ptr(e_grad, ctypes.c_double),
            _as_ptr(g_grad.reshape(-1), ctypes.c_double),
        )
        rc_energy = fn_energy(
            ctypes.byref(step),
            ctypes.byref(grid),
            ctypes.byref(global_data),
            ctypes.byref(cfg),
            _as_ptr(e_energy, ctypes.c_double),
            None,
        )
        self.assertEqual(rc_grad, 0)
        self.assertEqual(rc_energy, 0)
        np.testing.assert_allclose(e_energy, e_grad, rtol=0.0, atol=1e-12)


# ===========================================================================
# Test suite 7: Makefile codegen auto-trigger
# ===========================================================================
class TestMakefileCodegenTrigger(unittest.TestCase):
    """Make rules: forcefields/%/nb_kernel_%.cpp depends on ff.yaml + codegen_ff.py."""

    def test_makefile_has_pattern_rule_for_cpp(self):
        makefile = (KERNEL_ROOT / "Makefile").read_text()
        # The combined rule invokes codegen_ff.py codegen as a make recipe step.
        self.assertIn("codegen_ff.py codegen", makefile)

    def test_makefile_has_ffs_target(self):
        makefile = (KERNEL_ROOT / "Makefile").read_text()
        self.assertIn("ffs", makefile)

    def test_makefile_has_pattern_rule_for_so(self):
        makefile = (KERNEL_ROOT / "Makefile").read_text()
        self.assertIn("nb_kernel_%.so", makefile)

    def test_makefile_legacy_target_unchanged(self):
        makefile = (KERNEL_ROOT / "Makefile").read_text()
        self.assertIn("libnbkernel_nonbon8.so", makefile)
        self.assertIn("nb_kernel.cpp", makefile)


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
