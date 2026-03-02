"""
elec.py — Energy-only electrostatics for nonbon8 (rdie: distance-dependent dielectric).

JAX-traceable Python reference implementation that matches the C kernel
in forcefields/nonbon8/elec.h.

Physics (rdie model, identical to elec.h):
    dd = max(0, rr2 - (1/50)^2)
    E  = charge * dd
where charge = q_i * q_j * felec^2  (prescaled by the caller).
The 50 Å hard cutoff means dd=0 for r>50 Å.
"""

import jax.numpy as jnp

_RDIE_THRESH = (1.0 / 50.0) ** 2  # 1 / r_cut^2 = 4e-4 Å^{-2}


def elec_energy(charge, rr2, dsq=None):
    """Energy-only rdie electrostatics for a single atom pair (nonbon8).

    Parameters
    ----------
    charge : float — product of the two prescaled atomic charges
                     (q_i * q_j * (332.054 / epsilon))
    rr2    : float — 1/dsq, precomputed
    dsq    : float, optional — squared distance; not used by this implementation
             (accepted for API compatibility with the Python contract signature)

    Returns
    -------
    energy : float — pairwise electrostatic energy contribution (kcal/mol)

    Notes
    -----
    Uses jnp.where for JAX-traceability; equivalent to the C branch in elec.h.
    """
    dd = rr2 - _RDIE_THRESH
    return charge * jnp.where(dd > 0.0, dd, 0.0)
