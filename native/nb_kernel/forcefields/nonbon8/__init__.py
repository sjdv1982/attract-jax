"""
nonbon8 — Python forcefield module for ATTRACT nonbon8 (8/6 LJ + rdie electrostatics).

Follows the Section 5.1 force field module contract:
  - lj_energy(rc, ac, emin, rmin2, ivor, dsq, rr2) -> float
  - elec_energy(charge, rr2, dsq) -> float
  - load_params(npz_path) -> FFParams

This package co-locates with the C kernel headers (forcefields/nonbon8/).
The Python functions are the reference implementations that the C .h files
in this directory are ports of.

Discovery (Section 5.6 pattern):
    from forcefields import nonbon8
    ff = nonbon8
    ff.load_params(attract_par_npz)   # returns FFParams namedtuple
    ff.lj_energy(...)                # JAX-traceable single-pair energy
    ff.elec_energy(...)              # JAX-traceable single-pair energy
"""

from .lj import lj_energy
from .elec import elec_energy
from .params import load_params, FFParams

__all__ = [
    "lj_energy",
    "elec_energy",
    "load_params",
    "FFParams",
]
