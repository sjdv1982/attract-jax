"""
forcefields — Force field implementations for the nb_kernel.

Each subdirectory is a Python package implementing the Section 5.1 contract:
  - lj_energy(rc, ac, emin, rmin2, ivor, dsq, rr2) -> float
  - elec_energy(charge, rr2, dsq) -> float
  - load_params(npz_path) -> FFParams

Available force fields:
  - nonbon8: 8/6 Lennard-Jones with rdie (distance-dependent dielectric).
             The classic ATTRACT potential.
"""
