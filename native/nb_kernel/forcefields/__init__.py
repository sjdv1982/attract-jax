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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# Required attributes for a valid force field module.
_REQUIRED_ATTRS = ("lj_energy", "elec_energy", "load_params")

# Canonical extern "C" symbols emitted by codegen_ff.py.
KNOWN_KERNEL_SYMBOLS = (
    "nb_kernel_euler_correction_grad",
    "nb_kernel_euler_correction_energy",
    "nb_kernel_euler_clamp_grad",
    "nb_kernel_euler_clamp_energy",
    "nb_kernel_euler_grad",
    "nb_kernel_euler_energy",
)


@dataclass(frozen=True)
class KernelDispatch:
    """Selected wrapper family for one rotation scheme."""

    family: str
    plateau_mode: str
    grad_symbol: Optional[str]
    energy_symbol: str
    uses_legacy_alias: bool = False


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
    """Look for nb_kernel_<name>.so near the force field module.

    Searches from the FF package directory upward (up to 4 levels) for a file
    named ``nb_kernel_<name>.so``, where *name* is the last component of the
    module's ``__name__``.  This accommodates layouts where the compiled kernel
    sits in a parent directory of the force field Python package.

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
    so_name = f"nb_kernel_{ff_name}.so"
    candidate = ff_dir
    for _ in range(4):
        so_path = candidate / so_name
        if so_path.exists():
            return ctypes.CDLL(str(so_path))
        candidate = candidate.parent
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


def select_kernel_dispatch(
    available_symbols: set, rotation: str = "euler", plateau_mode: str = "correction"
) -> KernelDispatch:
    """Pick grad+energy or energy-only wrapper family from available symbols."""
    if plateau_mode not in {"correction", "clamp"}:
        raise ValueError(
            f"Unsupported plateau_mode={plateau_mode!r}; expected 'correction' or 'clamp'"
        )
    grad_sym = f"nb_kernel_{rotation}_{plateau_mode}_grad"
    energy_sym = f"nb_kernel_{rotation}_{plateau_mode}_energy"

    has_grad = grad_sym in available_symbols
    has_energy = energy_sym in available_symbols

    if has_grad and has_energy:
        return KernelDispatch(
            family="grad_energy",
            plateau_mode=plateau_mode,
            grad_symbol=grad_sym,
            energy_symbol=energy_sym,
        )
    if has_energy:
        return KernelDispatch(
            family="energy_only",
            plateau_mode=plateau_mode,
            grad_symbol=None,
            energy_symbol=energy_sym,
        )

    # Backward-compatible fallback for old correction-only symbol names.
    if plateau_mode == "correction":
        legacy_grad = f"nb_kernel_{rotation}_grad"
        legacy_energy = f"nb_kernel_{rotation}_energy"
        has_legacy_grad = legacy_grad in available_symbols
        has_legacy_energy = legacy_energy in available_symbols
        if has_legacy_grad and has_legacy_energy:
            return KernelDispatch(
                family="grad_energy",
                plateau_mode=plateau_mode,
                grad_symbol=legacy_grad,
                energy_symbol=legacy_energy,
                uses_legacy_alias=True,
            )
        if has_legacy_energy:
            return KernelDispatch(
                family="energy_only",
                plateau_mode=plateau_mode,
                grad_symbol=None,
                energy_symbol=legacy_energy,
                uses_legacy_alias=True,
            )

    raise RuntimeError(
        f"No callable wrappers for rotation '{rotation}', plateau_mode '{plateau_mode}'. "
        f"Expected '{energy_sym}' (and optionally '{grad_sym}')."
    )


def bind_kernel_dispatch(
    lib: "ctypes.CDLL", rotation: str = "euler", plateau_mode: str = "correction"
) -> Tuple[KernelDispatch, Optional[object], object]:
    """Probe and bind C wrapper functions from a loaded kernel library."""
    available = probe_kernel_symbols(lib)
    dispatch = select_kernel_dispatch(
        available, rotation=rotation, plateau_mode=plateau_mode
    )
    grad_fn = getattr(lib, dispatch.grad_symbol) if dispatch.grad_symbol else None
    energy_fn = getattr(lib, dispatch.energy_symbol)
    return dispatch, grad_fn, energy_fn
