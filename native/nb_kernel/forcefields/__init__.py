"""
forcefields — Force field implementations for the nb_kernel.

Each subdirectory is a Python package implementing the Section 5.1 contract:
  - lj_energy(rc, ac, emin, rmin2, ivor, dsq, rr2) -> float
  - elec_energy(charge, rr2, dsq=None)              -> float
  - load_params(npz_path)                           -> FFParams

Available built-in force fields:
  - nonbon8: 8/6 Lennard-Jones with rdie (distance-dependent dielectric),
             the classic ATTRACT potential.

Discovery helpers (Section 5.6 of PLAN_NONBONDED_FORCEFIELD.md):
  - load_forcefield(ff_spec)  — load a FF from a filesystem path or import path
  - find_kernel_so(ff_module) — locate nb_kernel_<name>.so next to the FF module
  - probe_kernel_symbols(lib) — return set of available extern "C" symbol names
"""

import ctypes
import importlib
import sys
from pathlib import Path

# Required attributes for a valid force field module.
_REQUIRED_ATTRS = ("lj_energy", "elec_energy", "load_params")

# Canonical extern "C" symbols emitted by codegen_ff.py.
KNOWN_KERNEL_SYMBOLS = (
    "nb_kernel_euler_grad",
    "nb_kernel_euler_energy",
)


def load_forcefield(ff_spec: str):
    """Load a force field module from a filesystem path or Python import path.

    Parameters
    ----------
    ff_spec : str
        Either:
        - A filesystem path to a directory containing ``__init__.py``
          (parent directory is temporarily added to ``sys.path``), or
        - A dotted Python import path (e.g. ``"forcefields.nonbon8"``).

    Returns
    -------
    module
        The loaded force field module.  Guaranteed to have ``lj_energy``,
        ``elec_energy``, and ``load_params`` attributes.

    Raises
    ------
    ImportError
        If the module cannot be found or is missing a required attribute.
    """
    path = Path(ff_spec)
    if path.is_dir() and (path / "__init__.py").exists():
        parent = str(path.parent.resolve())
        if parent not in sys.path:
            sys.path.insert(0, parent)
        module = importlib.import_module(path.name)
    else:
        module = importlib.import_module(ff_spec)

    for attr in _REQUIRED_ATTRS:
        if not hasattr(module, attr):
            raise ImportError(
                f"Force field '{ff_spec}' is missing required attribute '{attr}'"
            )
    return module


def find_kernel_so(ff_module) -> "ctypes.CDLL | None":
    """Look for nb_kernel_<name>.so next to the force field module.

    The convention (Section 5.6) is that the shared library lives in the same
    directory as the force field Python package, named ``nb_kernel_<name>.so``
    where *name* is the last component of the module's ``__name__``.

    Parameters
    ----------
    ff_module : module
        A loaded force field module (returned by :func:`load_forcefield`).

    Returns
    -------
    ctypes.CDLL or None
        Loaded shared library, or ``None`` if no ``.so`` is found.
    """
    ff_dir = Path(ff_module.__file__).parent
    ff_name = ff_module.__name__.split(".")[-1]
    so_path = ff_dir / f"nb_kernel_{ff_name}.so"
    if so_path.exists():
        return ctypes.CDLL(str(so_path))
    return None


def probe_kernel_symbols(lib: "ctypes.CDLL") -> set:
    """Return the set of known kernel symbols available in *lib*.

    Parameters
    ----------
    lib : ctypes.CDLL
        A loaded shared library (from :func:`find_kernel_so`).

    Returns
    -------
    set of str
        Subset of :data:`KNOWN_KERNEL_SYMBOLS` found in *lib*.
    """
    available = set()
    for sym in KNOWN_KERNEL_SYMBOLS:
        try:
            getattr(lib, sym)
            available.add(sym)
        except AttributeError:
            pass
    return available
