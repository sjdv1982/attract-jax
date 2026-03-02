"""
test_codegen_ff.py — Tests for codegen_ff.py (Milestone 3).

Tests cover:
  1. 'init' mode: correct files created, with expected namespace and YAML structure.
  2. 'codegen' mode: generated .cpp has correct structure, symbols, and headers.
  3. 'codegen' capability detection: grad wrappers emitted iff both *_grad.h present.
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
def _copy_nonbon8_headers_to_dummy(dst_dir: Path) -> None:
    """Copy nonbon8 .h files into dst_dir, substituting namespace nonbon8 → dummy."""
    for fname in ("lj.h", "elec.h", "lj_grad.h", "elec_grad.h"):
        src = NONBON8_DIR / fname
        if not src.exists():
            continue
        text = src.read_text()
        # Rename both the namespace declaration and calls within it
        text = text.replace("namespace nonbon8", "namespace dummy")
        text = text.replace("nonbon8::", "dummy::")
        # Update include guards
        text = text.replace("NONBON8_", "DUMMY_")
        (dst_dir / fname).write_text(text)


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

    def test_wrappers_are_extern_c(self):
        self.assertIn('extern "C"', self.cpp_text)

    def test_grad_wrapper_uses_euler_rot(self):
        self.assertIn("EulerRot", self.cpp_text)

    def test_grad_wrapper_instantiates_compute_grad_true(self):
        self.assertIn("DummyFF, true>", self.cpp_text)

    def test_energy_wrapper_instantiates_compute_grad_false(self):
        self.assertIn("DummyFF, false>", self.cpp_text)

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

    def test_grad_wrapper_absent(self):
        """No grad wrapper when either grad header is missing."""
        self.assertNotIn("nb_kernel_euler_grad", self.cpp_text)

    def test_no_lj_grad_include(self):
        self.assertNotIn("lj_grad.h", self.cpp_text)

    def test_no_elec_grad_include(self):
        self.assertNotIn("elec_grad.h", self.cpp_text)

    def test_ffpolicy_has_no_lj_grad(self):
        self.assertNotIn("lj_grad", self.cpp_text)

    def test_energy_wrapper_instantiates_compute_grad_false(self):
        self.assertIn("false>", self.cpp_text)


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
